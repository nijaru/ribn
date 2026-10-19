//! Host CPU backend arithmetic, not a full decoder or a performance claim.

use engine_gemma4::{
    AttentionKind, CpuParameter, Gemma4Config, Gemma4CpuWeights, Gemma4Source, LayerParameter,
    ParameterId, apply_rotary, bf16, factors, linear, linear_f32, rms_norm,
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

fn compare_linear(input: &[u16], weight: &CpuParameter, expected: &[u16]) -> f32 {
    let (sums, magnitudes) = linear_f32(input, 1, weight).unwrap();
    assert_eq!(sums.len(), expected.len(), "linear reference length");
    assert_eq!(
        linear(input, 1, weight).unwrap(),
        sums.iter().map(|&sum| bf16(sum)).collect::<Vec<_>>(),
        "linear must round the audited FP32 sums exactly once"
    );
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

fn compare_layer(
    config: &Gemma4Config,
    weights: &Gemma4CpuWeights,
    stages: &SafeTensorArtifact,
    components: &SafeTensorArtifact,
    layer: usize,
) -> (f32, f32) {
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
    norm_worst = norm_worst.max(compare_norm(
        &ffn,
        &tap(stages, &format!("{prefix}.pre_feedforward_norm")),
        width,
    ));
    let ffn_reference = tap(stages, &format!("{prefix}.pre_feedforward_norm"));
    let mut linear_worst = 0.0_f32;
    for (role, name) in [
        (LayerParameter::Gate, "mlp_gate"),
        (LayerParameter::Up, "mlp_up"),
    ] {
        linear_worst = linear_worst.max(compare_linear(
            &ffn_reference,
            weights.layer(layer, role).unwrap(),
            &tap(stages, &format!("{prefix}.{name}")),
        ));
    }
    let product = tap(stages, &format!("{prefix}.mlp_gate_product"));
    linear_worst = linear_worst.max(compare_linear(
        &product,
        weights.layer(layer, LayerParameter::Down).unwrap(),
        &tap(stages, &format!("{prefix}.mlp_output")),
    ));
    let ple_raw = tap(stages, &format!("{prefix}.ple_projection_raw"));
    let ple = rms_norm(
        &ple_raw,
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

    // Exact split-half application with the captured stock factors.
    let cosine = tap(components, &format!("{prefix}.cos"));
    let sine = tap(components, &format!("{prefix}.sin"));
    let factor_width = cosine.len();
    assert_eq!(config.layers()[layer].head_dim, factor_width);
    let queries = tap(components, &format!("{prefix}.q_norm"));
    let rotated = apply_rotary(
        &queries,
        1,
        queries.len() / factor_width,
        factor_width,
        &cosine,
        &sine,
    )
    .unwrap();
    assert_eq!(
        rotated,
        tap(components, &format!("{prefix}.rotated_query")),
        "layer {layer} rotated query"
    );
    let keys = tap(components, &format!("{prefix}.k_norm"));
    let rotated_key = apply_rotary(
        &keys,
        1,
        keys.len() / factor_width,
        factor_width,
        &cosine,
        &sine,
    )
    .unwrap();
    assert_eq!(
        rotated_key,
        tap(components, &format!("{prefix}.rotated_key")),
        "layer {layer} rotated key"
    );
    (linear_worst, norm_worst)
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
    for layer in [0_usize, 5, 22, 23] {
        let (linear, norm) = compare_layer(&config, &weights, &stages, &components, layer);
        linear_worst = linear_worst.max(linear);
        norm_worst = norm_worst.max(norm);
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
    assert!(linear_worst <= 1.0, "linear bound usage {linear_worst}");
    assert!(norm_worst <= 1.0, "normalization bound usage {norm_worst}");
    println!("linear bound usage {linear_worst}, normalization bound usage {norm_worst}");
}
