//! Host CPU backend arithmetic, not a full decoder or a performance claim.

use engine_gemma4::{
    AttentionKind, CpuParameter, Gemma4Config, Gemma4CpuWeights, Gemma4Source, LayerParameter,
    ParameterId, add, apply_rotary, bf16, factors, gelu_tanh, linear, linear_f32, multiply,
    ple_combine, rms_norm, scale, softcap,
};
use ribn_hf::LocalModelPackage;
use ribn_safetensors::SafeTensorArtifact;
use std::io::Read;
use std::path::PathBuf;

fn value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn bits(value: f32) -> u16 {
    bf16(value)
}

#[test]
fn linear_rounds_once_after_fp32_accumulation() {
    // weight [2,3]; row-major source order makes every sum exactly representable.
    let weight = engine_gemma4::CpuParameter::from_parts(
        &[2, 3],
        &[
            bits(1.0),
            bits(2.0),
            bits(4.0),
            bits(0.5),
            bits(-1.0),
            bits(2.0),
        ],
    )
    .unwrap();
    let input = [bits(1.0), bits(1.0), bits(1.0)];
    let output = linear(&input, 1, &weight).unwrap();
    assert_eq!(output, [bits(7.0), bits(1.5)]);
    // A non-dyadic sum still rounds only once, at the end.
    let weight =
        engine_gemma4::CpuParameter::from_parts(&[1, 3], &[bits(0.1), bits(0.2), bits(0.3)])
            .unwrap();
    let input = [bits(1.0), bits(1.0), bits(1.0)];
    assert_eq!(
        linear(&input, 1, &weight).unwrap(),
        [bits(value(bits(0.1)) + value(bits(0.2)) + value(bits(0.3)))]
    );
}

#[test]
fn rms_normalizes_with_fp32_statistics_and_direct_scale() {
    let scale = engine_gemma4::CpuParameter::from_parts(&[2], &[bits(1.0), bits(1.0)]).unwrap();
    let input = [bits(3.0), bits(4.0)];
    let output = rms_norm(&input, 1, 2, 1e-6, &scale).unwrap();
    let mean = f32::midpoint(9.0_f32, 16.0);
    let inverse = (mean + 1e-6).powf(-0.5);
    assert_eq!(output, [bits(3.0 * inverse), bits(4.0 * inverse)]);
}

#[test]
fn rotary_rounds_each_product_and_reads_both_factor_halves() {
    // Identity factors preserve magnitudes, but the pinned eager addition of a
    // positive-zero partner turns an input negative zero into positive zero.
    let cosine = [bits(1.0), bits(1.0)];
    let sine = [bits(0.0), bits(0.0)];
    let input = [0x8000_u16, bits(-2.0)];
    assert_eq!(
        apply_rotary(&input, 1, 1, 2, &cosine, &sine).unwrap(),
        [0x0000, bits(-2.0)]
    );
    // Non-identical halves are read independently: the second element uses the
    // second cosine/sine entries, not the first pair.
    let cosine = [bits(1.0), bits(0.5)];
    let sine = [bits(0.0), bits(0.25)];
    let pair = [bits(2.0), bits(4.0)];
    let expected = [
        bf16(value(bits(2.0)) * 1.0 + (-value(bits(4.0))) * 0.0),
        bf16(value(bits(4.0)) * 0.5 + value(bits(2.0)) * 0.25),
    ];
    assert_eq!(
        apply_rotary(&pair, 1, 1, 2, &cosine, &sine).unwrap(),
        expected
    );

    // A quarter turn: x0' = -x1, x1' = x0, with each product BF16-rounded first.
    let cosine = [bits(0.0), bits(0.0)];
    let sine = [bits(1.0), bits(1.0)];
    let pair = [bits(3.0), bits(-4.0)];
    assert_eq!(
        apply_rotary(&pair, 1, 1, 2, &cosine, &sine).unwrap(),
        [bits(4.0), bits(3.0)]
    );
}

#[test]
fn factors_duplicate_halves_and_keep_padded_frequencies_identity() {
    let root: serde_json::Value = serde_json::from_str(include_str!(
        "../../../benchmarks/gemma4-reference/config.json"
    ))
    .unwrap();
    let config = Gemma4Config::from_json(&root).unwrap();
    let layer = config
        .layers()
        .iter()
        .position(|geometry| geometry.kind == AttentionKind::Full)
        .expect("the pinned config has global attention");
    let width = config.layers()[layer].head_dim;
    let active = width / 4;
    let (cosine, sine) = factors(&config, layer, &[0, 5]).unwrap();
    assert_eq!(cosine.len(), 2 * width);
    assert_eq!(sine.len(), 2 * width);
    for row in 0..2 {
        for channel in 0..width / 2 {
            assert_eq!(
                cosine[row * width + channel],
                cosine[row * width + channel + width / 2]
            );
            assert_eq!(
                sine[row * width + channel],
                sine[row * width + channel + width / 2]
            );
        }
    }
    // Position zero is exactly one and positive zero for every frequency.
    for channel in 0..width {
        assert_eq!((cosine[channel], sine[channel]), (0x3f80, 0x0000));
    }
    // Proportional padding keeps identity at nonzero positions too.
    for channel in active..(width / 2) {
        assert_eq!(
            (cosine[width + channel], sine[width + channel]),
            (0x3f80, 0x0000)
        );
    }
}

fn environment(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name)
            .unwrap_or_else(|| panic!("{name} must name the archived evidence file")),
    )
}

fn tap(artifact: &SafeTensorArtifact, name: &str) -> Vec<u16> {
    let tensor = artifact.tensor(name).unwrap();
    assert_eq!(
        *tensor.scalar_type(),
        ribn_foundation::ScalarType::Bf16,
        "{name}"
    );
    let mut bytes = vec![0_u8; tensor.len()];
    tensor.reader().read_exact(&mut bytes).unwrap();
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

/// Spacing to the next BF16 encoding in magnitude, the reference's rounding slack.
///
/// Encoding increases with magnitude on both sides of zero, so the successor works
/// for negative zero as well; non-finite successors fall back to an FP32 estimate.
fn ulp(bits: u16) -> f32 {
    let difference = (value(bits.wrapping_add(1)) - value(bits)).abs();
    if difference.is_finite() && difference > 0.0 {
        difference
    } else {
        value(bits).abs() * 2.0 * f32::EPSILON
    }
}

fn compare_linear(
    input: &[u16],
    weight: &CpuParameter,
    expected: &[u16],
    verify_rounding: bool,
) -> f32 {
    let (sums, magnitudes) = linear_f32(input, 1, weight).unwrap();
    assert_eq!(sums.len(), expected.len(), "linear reference length");
    if verify_rounding {
        assert_eq!(
            linear(input, 1, weight).unwrap(),
            sums.iter().map(|&sum| bf16(sum)).collect::<Vec<_>>(),
            "linear must round the audited FP32 sums exactly once"
        );
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "test shapes are small model dimensions"
    )]
    let width = weight.shape()[1] as f32;
    // First-order bound for two FP32 summation orders, plus the reference's own
    // BF16 rounding slack.
    let gamma = width * f32::EPSILON / (1.0 - width * f32::EPSILON);
    let mut worst = 0.0_f32;
    for (index, (&sum, &magnitude)) in sums.iter().zip(&magnitudes).enumerate() {
        let reference_bits = expected[index];
        let reference = value(reference_bits);
        assert!(
            sum.is_finite() && magnitude.is_finite() && reference.is_finite(),
            "non-finite linear operand"
        );
        let bound = gamma * magnitude + ulp(reference_bits);
        let error = (sum - reference).abs();
        worst = worst.max(error / bound);
        assert!(
            error <= bound,
            "linear error {error:e} exceeds the FP32 summation bound {bound:e}"
        );
    }
    worst
}

fn compare_norm(actual: &[u16], expected: &[u16], width: usize) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut worst = 0.0_f32;
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let (a, b) = (value(a), value(b));
        #[expect(
            clippy::cast_precision_loss,
            reason = "normalization widths are small model dimensions"
        )]
        let bound = width as f32 * f32::EPSILON * a.abs().max(b.abs()) + 2.0 * ulp(expected[index]);
        let error = (a - b).abs();
        worst = worst.max(error / bound);
        assert!(
            error <= bound,
            "normalization error {error:e} exceeds the FP32 variance bound {bound:e}"
        );
    }
    worst
}

fn compare_gelu(input: &[u16], expected: &[u16]) -> f32 {
    assert_eq!(input.len(), expected.len(), "GELU reference length");
    let actual = gelu_tanh(input).unwrap();
    let mut worst = 0.0_f32;
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let (a, b) = (value(a), value(b));
        assert!(a.is_finite() && b.is_finite(), "non-finite GELU operand");
        let bound = 16.0 * f32::EPSILON * value(input[index]).abs() + ulp(expected[index]);
        let error = (a - b).abs();
        worst = worst.max(error / bound);
        assert!(
            error <= bound,
            "GELU error {error:e} exceeds the FP32 evaluation bound {bound:e}"
        );
    }
    worst
}

fn compare_softcap(input: &[u16], expected: &[u16], cap: f32) -> f32 {
    assert_eq!(input.len(), expected.len(), "soft-cap reference length");
    let actual = softcap(input, cap).unwrap();
    let mut worst = 0.0_f32;
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        let (a, b) = (value(a), value(b));
        assert!(
            a.is_finite() && b.is_finite(),
            "non-finite soft-cap operand"
        );
        let bound = 8.0 * f32::EPSILON * cap + 2.0 * ulp(expected[index]);
        let error = (a - b).abs();
        worst = worst.max(error / bound);
        assert!(
            error <= bound,
            "soft cap error {error:e} exceeds the tanh bound {bound:e}"
        );
    }
    worst
}

fn compare_layer(
    config: &Gemma4Config,
    weights: &Gemma4CpuWeights,
    stages: &SafeTensorArtifact,
    components: &SafeTensorArtifact,
    layer: usize,
) -> (f32, f32, f32) {
    let prefix = format!("chat.0.layer.{layer}");
    let width = config.hidden_size();
    let epsilon = config.rms_norm_eps();
    let hidden = tap(stages, &format!("{prefix}.input"));
    let normed = rms_norm(
        &hidden,
        1,
        width,
        epsilon,
        weights.layer(layer, LayerParameter::InputNorm).unwrap(),
    )
    .unwrap();
    let mut norm_worst = compare_norm(
        &normed,
        &tap(stages, &format!("{prefix}.input_norm")),
        width,
    );
    let residual = tap(stages, &format!("{prefix}.post_attention_residual"));
    let ffn_reference = tap(stages, &format!("{prefix}.pre_feedforward_norm"));
    let ffn = rms_norm(
        &residual,
        1,
        width,
        epsilon,
        weights
            .layer(layer, LayerParameter::PreFeedforwardNorm)
            .unwrap(),
    )
    .unwrap();
    norm_worst = norm_worst.max(compare_norm(&ffn, &ffn_reference, width));
    let mut linear_worst = 0.0_f32;
    for (role, name) in [
        (LayerParameter::Gate, "mlp_gate"),
        (LayerParameter::Up, "mlp_up"),
    ] {
        linear_worst = linear_worst.max(compare_linear(
            &ffn_reference,
            weights.layer(layer, role).unwrap(),
            &tap(stages, &format!("{prefix}.{name}")),
            true,
        ));
    }
    let product = tap(stages, &format!("{prefix}.mlp_gate_product"));
    linear_worst = linear_worst.max(compare_linear(
        &product,
        weights.layer(layer, LayerParameter::Down).unwrap(),
        &tap(stages, &format!("{prefix}.mlp_output")),
        true,
    ));
    let ple = rms_norm(
        &tap(stages, &format!("{prefix}.ple_projection_raw")),
        1,
        width,
        epsilon,
        weights.layer(layer, LayerParameter::PostPleNorm).unwrap(),
    )
    .unwrap();
    norm_worst = norm_worst.max(compare_norm(
        &ple,
        &tap(stages, &format!("{prefix}.ple_norm")),
        width,
    ));
    let gelu = compare_layer_gating(stages, layer, &hidden, &residual, &ple);
    compare_layer_rotary(config, components, layer);
    (linear_worst, norm_worst, gelu)
}

/// Exact BF16 residual/scale algebra plus the GELU-saturation bound.
fn compare_layer_gating(
    stages: &SafeTensorArtifact,
    layer: usize,
    hidden: &[u16],
    residual: &[u16],
    ple: &[u16],
) -> f32 {
    let prefix = format!("chat.0.layer.{layer}");
    let attention_norm = tap(stages, &format!("{prefix}.post_attention_norm"));
    assert_eq!(
        add(hidden, &attention_norm).unwrap(),
        residual,
        "layer {layer} attention residual"
    );
    let ffn_norm = tap(stages, &format!("{prefix}.post_feedforward_norm"));
    let ffn_residual = tap(stages, &format!("{prefix}.post_feedforward_residual"));
    assert_eq!(
        add(residual, &ffn_norm).unwrap(),
        ffn_residual,
        "layer {layer} feed-forward residual"
    );
    let scalar = tap(stages, &format!("{prefix}.layer_scalar"));
    assert_eq!(
        scale(&add(&ffn_residual, ple).unwrap(), value(scalar[0])).unwrap(),
        tap(stages, &format!("{prefix}.output")),
        "layer {layer} layer-scalar output"
    );
    let gate = tap(stages, &format!("{prefix}.mlp_gate"));
    let up = tap(stages, &format!("{prefix}.mlp_up"));
    let activated = tap(stages, &format!("{prefix}.mlp_gate_activation"));
    let gelu = compare_gelu(&gate, &activated);
    assert_eq!(
        multiply(&activated, &up).unwrap(),
        tap(stages, &format!("{prefix}.mlp_gate_product")),
        "layer {layer} gated product"
    );
    let projection = tap(stages, "chat.0.ple_projection_norm");
    let lookup = tap(stages, "chat.0.ple_lookup");
    let per_layer = tap(stages, &format!("{prefix}.per_layer_input"));
    let width = per_layer.len();
    assert_eq!(
        ple_combine(
            &projection[layer * width..(layer + 1) * width],
            &lookup[layer * width..(layer + 1) * width],
        )
        .unwrap(),
        per_layer,
        "layer {layer} per-layer combine"
    );
    gelu
}

/// Exact split-half application with the captured stock factors.
fn compare_layer_rotary(config: &Gemma4Config, components: &SafeTensorArtifact, layer: usize) {
    let prefix = format!("chat.0.layer.{layer}");
    let cosine = tap(components, &format!("{prefix}.cos"));
    let sine = tap(components, &format!("{prefix}.sin"));
    let width = cosine.len();
    assert_eq!(config.layers()[layer].head_dim, width);
    for (stored, rotated) in [("q_norm", "rotated_query"), ("k_norm", "rotated_key")] {
        let input = tap(components, &format!("{prefix}.{stored}"));
        let actual = apply_rotary(&input, 1, input.len() / width, width, &cosine, &sine).unwrap();
        assert_eq!(
            actual,
            tap(components, &format!("{prefix}.{rotated}")),
            "layer {layer} {rotated}"
        );
    }
}

#[test]
#[ignore = "requires the pinned Gemma package and the archived CPU references"]
fn host_weights_and_ops_match_the_captured_cpu_reference() {
    let package = LocalModelPackage::open(environment("RIBN_GEMMA_PACKAGE")).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    let config = Gemma4Config::from_json(
        &serde_json::from_slice(&std::fs::read(package.root().join("config.json")).unwrap())
            .unwrap(),
    )
    .unwrap();
    let weights = Gemma4CpuWeights::load(&source).unwrap();
    assert_eq!(weights.len(), 665);
    assert_eq!(
        weights.bytes(),
        u64::try_from(source.encoded_bytes()).unwrap()
    );

    let stages = SafeTensorArtifact::open(environment("RIBN_GEMMA_STAGES")).unwrap();
    let components = SafeTensorArtifact::open(environment("RIBN_GEMMA_COMPONENTS")).unwrap();
    let mut linear_worst = 0.0_f32;
    let mut norm_worst = 0.0_f32;
    let mut gelu_worst = 0.0_f32;
    for layer in [0_usize, 5, 22, 23] {
        let (linear, norm, gelu) = compare_layer(&config, &weights, &stages, &components, layer);
        linear_worst = linear_worst.max(linear);
        norm_worst = norm_worst.max(norm);
        gelu_worst = gelu_worst.max(gelu);
    }
    let final_norm = rms_norm(
        &tap(&stages, "chat.0.final_norm_input"),
        1,
        config.hidden_size(),
        config.rms_norm_eps(),
        weights.parameter(ParameterId::FinalNorm).unwrap(),
    )
    .unwrap();
    norm_worst = norm_worst.max(compare_norm(
        &final_norm,
        &tap(&stages, "chat.0.final_norm"),
        config.hidden_size(),
    ));
    let head = weights.parameter(ParameterId::TokenEmbedding).unwrap();
    let raw = tap(&stages, "chat.0.raw_logits");
    let head_worst = compare_linear(&final_norm, head, &raw, false);
    let cap = config.final_logit_softcapping();
    let capped = tap(&stages, "chat.0.logits");
    let softcap_worst = compare_softcap(&raw, &capped, cap);
    assert!(linear_worst <= 1.0, "linear bound usage {linear_worst}");
    assert!(norm_worst <= 1.0, "normalization bound usage {norm_worst}");
    assert!(gelu_worst <= 1.0, "GELU bound usage {gelu_worst}");
    assert!(head_worst <= 1.0, "head linear bound usage {head_worst}");
    assert!(softcap_worst <= 1.0, "soft-cap bound usage {softcap_worst}");
    println!(
        "linear {linear_worst}, norm {norm_worst}, gelu {gelu_worst}, head {head_worst}, softcap {softcap_worst}"
    );
}
