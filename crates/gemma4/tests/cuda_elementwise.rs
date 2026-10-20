//! Remaining Gemma layer math against the archived CPU/stock-CUDA stage taps.
#![cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};
use engine_gemma4::Gemma4Config;
use engine_nvidia::{CudaBf16Elementwise, CudaBf16Matrix};
use ribn_foundation::{BytePool, ScalarType};
use ribn_safetensors::SafeTensorArtifact;
use serde_json::Value;
use std::{io::Read, path::PathBuf, sync::Arc};

const CONFIG: &str = include_str!("../../../benchmarks/gemma4-reference/config.json");

fn value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn ulp(bits: u16) -> f32 {
    let difference = (value(bits.wrapping_add(1)) - value(bits)).abs();
    if difference.is_finite() && difference > 0.0 {
        difference
    } else {
        value(bits).abs() * 2.0 * f32::EPSILON
    }
}

fn tap(artifact: &SafeTensorArtifact, name: &str) -> Vec<u16> {
    let tensor = artifact.tensor(name).unwrap();
    assert_eq!(*tensor.scalar_type(), ScalarType::Bf16, "{name}");
    let mut bytes = vec![0_u8; tensor.len()];
    tensor.reader().read_exact(&mut bytes).unwrap();
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

fn upload(values: &[u16], stream: &Arc<CudaStream>, pool: &Arc<BytePool>) -> CudaBf16Matrix {
    let mut matrix =
        CudaBf16Matrix::allocate(1, values.len(), Arc::clone(stream), pool, 5).unwrap();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    matrix.upload(&bytes).unwrap();
    matrix
}

fn read(matrix: &mut CudaBf16Matrix) -> Vec<u16> {
    let len = matrix.shape().0 * matrix.shape().1 * 2;
    let mut bytes = Vec::with_capacity(len);
    for offset in (0..len).step_by(5) {
        bytes.extend(
            matrix
                .read_chunk(offset as u64, 5.min(len - offset))
                .unwrap(),
        );
    }
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

fn run(
    ops: &mut CudaBf16Elementwise,
    stream: &Arc<CudaStream>,
    pool: &Arc<BytePool>,
    values: &[u16],
    operation: impl FnOnce(&mut CudaBf16Elementwise, &CudaBf16Matrix, &mut CudaBf16Matrix),
) -> Vec<u16> {
    let input = upload(values, stream, pool);
    let mut output =
        CudaBf16Matrix::allocate(1, values.len(), Arc::clone(stream), pool, 5).unwrap();
    operation(ops, &input, &mut output);
    let result = read(&mut output);
    let mut input = input;
    input.close().unwrap();
    output.close().unwrap();
    result
}

fn assert_exact(actual: &[u16], expected: &[u16], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual, expected, "{label} index {index}");
    }
}

fn assert_gelu(actual: &[u16], reference: &[u16]) {
    for (index, (&a, &b)) in actual.iter().zip(reference).enumerate() {
        let (a, b) = (value(a), value(b));
        assert!(a.is_finite() && b.is_finite(), "non-finite GELU {index}");
        let bound = 16.0 * f32::EPSILON * value(reference[index]).abs() + ulp(reference[index]);
        assert!(
            (a - b).abs() <= bound,
            "GELU {index}: device {a:e} reference {b:e} bound {bound:e}"
        );
    }
}

fn assert_softcap(actual: &[u16], reference: &[u16], cap: f32) {
    for (index, (&a, &b)) in actual.iter().zip(reference).enumerate() {
        let (a, b) = (value(a), value(b));
        assert!(
            a.is_finite() && b.is_finite(),
            "non-finite soft cap {index}"
        );
        let bound = 8.0 * f32::EPSILON * cap + 2.0 * ulp(reference[index]);
        assert!(
            (a - b).abs() <= bound,
            "soft cap {index}: device {a:e} reference {b:e} bound {bound:e}"
        );
    }
}

#[test]
#[ignore = "requires CUDA and the archived Gemma stage tensors; serialize device runs"]
#[expect(
    clippy::too_many_lines,
    reason = "one linear device sequence per archived layer"
)]
fn layer_math_matches_the_archived_stages() {
    let artifact = SafeTensorArtifact::open(PathBuf::from(
        std::env::var_os("RIBN_GEMMA_STAGES").expect("stage tensors"),
    ))
    .unwrap();
    let config = Gemma4Config::from_json(&serde_json::from_str::<Value>(CONFIG).unwrap()).unwrap();
    let width = config.hidden_size();
    assert_eq!(width, 2560);
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(64 * 1024 * 1024).shared();
    let mut ops = CudaBf16Elementwise::new(Arc::clone(&stream)).unwrap();
    for layer in [0_usize, 5, 22, 23] {
        let prefix = format!("chat.0.layer.{layer}");
        let input = tap(&artifact, &format!("{prefix}.input"));
        let attention_norm = tap(&artifact, &format!("{prefix}.post_attention_norm"));
        let attention_residual = tap(&artifact, &format!("{prefix}.post_attention_residual"));
        let ffn_norm = tap(&artifact, &format!("{prefix}.post_feedforward_norm"));
        let ffn_residual = tap(&artifact, &format!("{prefix}.post_feedforward_residual"));
        let ple_norm = tap(&artifact, &format!("{prefix}.ple_norm"));
        let per_layer = tap(&artifact, &format!("{prefix}.per_layer_input"));
        let output = tap(&artifact, &format!("{prefix}.output"));
        let gate = tap(&artifact, &format!("{prefix}.mlp_gate"));
        let activated = tap(&artifact, &format!("{prefix}.mlp_gate_activation"));
        let up = tap(&artifact, &format!("{prefix}.mlp_up"));
        let product = tap(&artifact, &format!("{prefix}.mlp_gate_product"));
        let scalar = value(tap(&artifact, &format!("{prefix}.layer_scalar"))[0]);
        for tensor in [
            &input,
            &attention_norm,
            &attention_residual,
            &ffn_norm,
            &ffn_residual,
            &ple_norm,
            &output,
        ] {
            assert_eq!(tensor.len(), width, "layer {layer} residual width");
        }
        let mlp_width = gate.len();
        assert_eq!(mlp_width, 10_240, "layer {layer} MLP width");
        for tensor in [&activated, &up, &product] {
            assert_eq!(tensor.len(), mlp_width, "layer {layer} MLP width");
        }

        // Exact: one IEEE operation on BF16 operands, then RNE to BF16.
        let actual = run(&mut ops, &stream, &pool, &input, |ops, input, output| {
            let norm = upload(&attention_norm, &stream, &pool);
            ops.add_into(input, &norm, output).unwrap();
            let mut norm = norm;
            norm.close().unwrap();
        });
        assert_exact(
            &actual,
            &attention_residual,
            &format!("{prefix} attention residual"),
        );

        let actual = run(
            &mut ops,
            &stream,
            &pool,
            &attention_residual,
            |ops, residual, output| {
                let norm = upload(&ffn_norm, &stream, &pool);
                ops.add_into(residual, &norm, output).unwrap();
                let mut norm = norm;
                norm.close().unwrap();
            },
        );
        assert_exact(
            &actual,
            &ffn_residual,
            &format!("{prefix} feed-forward residual"),
        );

        let projection = tap(&artifact, "chat.0.ple_projection_norm");
        let lookup = tap(&artifact, "chat.0.ple_lookup");
        let ple_width = per_layer.len();
        assert_eq!(ple_width, 256);
        let start = layer * ple_width;
        let actual = run(
            &mut ops,
            &stream,
            &pool,
            &projection[start..start + ple_width],
            |ops, projection, output| {
                let lookup = upload(&lookup[start..start + ple_width], &stream, &pool);
                ops.ple_combine_into(projection, &lookup, output).unwrap();
                let mut lookup = lookup;
                lookup.close().unwrap();
            },
        );
        assert_exact(&actual, &per_layer, &format!("{prefix} PLE combine"));

        let actual = run(
            &mut ops,
            &stream,
            &pool,
            &ffn_residual,
            |ops, residual, output| {
                let ple_norm = upload(&ple_norm, &stream, &pool);
                ops.add_into(residual, &ple_norm, output).unwrap();
                let mut ple_norm = ple_norm;
                ple_norm.close().unwrap();
            },
        );
        assert_eq!(actual.len(), width);
        let summed: Vec<u16> = actual;
        let mut residual = upload(&summed, &stream, &pool);
        let mut scaled = CudaBf16Matrix::allocate(1, width, Arc::clone(&stream), &pool, 5).unwrap();
        ops.scale_into(&residual, scalar, &mut scaled).unwrap();
        assert_exact(
            &read(&mut scaled),
            &output,
            &format!("{prefix} layer scalar"),
        );
        residual.close().unwrap();
        scaled.close().unwrap();

        let actual = run(
            &mut ops,
            &stream,
            &pool,
            &activated,
            |ops, activated, output| {
                let up = upload(&up, &stream, &pool);
                ops.multiply_into(activated, &up, output).unwrap();
                let mut up = up;
                up.close().unwrap();
            },
        );
        assert_exact(&actual, &product, &format!("{prefix} gated product"));

        let actual = run(&mut ops, &stream, &pool, &gate, |ops, gate, output| {
            ops.gelu_tanh_into(gate, output).unwrap();
        });
        assert_gelu(&actual, &activated);
    }

    let raw = tap(&artifact, "chat.0.raw_logits");
    let capped = tap(&artifact, "chat.0.logits");
    let actual = run(&mut ops, &stream, &pool, &raw, |ops, raw, output| {
        ops.softcap_into(raw, 30.0, output).unwrap();
    });
    assert_softcap(&actual, &capped, 30.0);

    ops.close().unwrap();
    assert_eq!(pool.granted(), 0);
}
