//! Native attention against the archived stock attention operands.
#![cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};
use engine_gemma4::{AttentionConfig, attention};
use engine_nvidia::{AttentionGeometry, AttentionSpan, CudaBf16Attention, CudaBf16Matrix};
use ribn_foundation::{BytePool, ScalarType};
use ribn_safetensors::SafeTensorArtifact;
use serde_json::Value;
use std::{io::Read, path::PathBuf, sync::Arc};

fn value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
fn tap(artifact: &SafeTensorArtifact, name: &str) -> Vec<u16> {
    let tensor = artifact.tensor(name).unwrap();
    assert_eq!(*tensor.scalar_type(), ScalarType::I32, "{name}");
    let mut bytes = vec![0_u8; tensor.len()];
    tensor.reader().read_exact(&mut bytes).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| {
            let raw = u32::from_le_bytes(*word);
            assert_eq!(raw & 0xffff_0000, 0, "{name} carries BF16 bits");
            u16::try_from(raw).unwrap()
        })
        .collect()
}
fn upload(
    values: &[u16],
    rows: usize,
    cols: usize,
    stream: &Arc<CudaStream>,
    pool: &Arc<BytePool>,
) -> CudaBf16Matrix {
    assert_eq!(values.len(), rows * cols);
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut matrix = CudaBf16Matrix::allocate(rows, cols, Arc::clone(stream), pool, 5).unwrap();
    matrix.upload(&bytes).unwrap();
    matrix
}
fn read(matrix: &mut CudaBf16Matrix) -> Vec<u16> {
    let len = matrix.shape().0 * matrix.shape().1 * 2;
    let mut bytes = Vec::new();
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
fn compare(actual: &[u16], expected: &[u16], label: &str) -> f32 {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let mut worst = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let (a, b) = (value(actual), value(expected));
        assert!(a.is_finite() && b.is_finite(), "{label} {index} non-finite");
        if b == 0.0 {
            assert_eq!(actual, expected, "{label} {index} signed zero");
            continue;
        }
        let expected_ulp = (value(expected.wrapping_add(1)) - b).abs();
        worst = worst.max((a - b).abs() / expected_ulp);
        // PREDECLARED: one adjacent BF16 encoding against the stock output.
        // Scores/weights round to BF16; only FP32 reduction order and the
        // exponential differ. A failure is investigated, not re-toleranced.
        assert_eq!(actual & 0x8000, expected & 0x8000, "{label} {index} sign");
        assert!(
            actual.abs_diff(expected) <= 1,
            "{label} {index}: device {a:e} reference {b:e}"
        );
    }
    worst
}

#[test]
#[ignore = "requires CUDA and the archived Gemma attention operands; serialize device runs"]
fn attention_matches_the_stock_operands_for_local_and_global_layers() {
    let root =
        PathBuf::from(std::env::var_os("RIBN_GEMMA_ATTENTION").expect("archived attention corpus"));
    let reference: Value =
        serde_json::from_slice(&std::fs::read(root.join("reference.json")).unwrap()).unwrap();
    assert_eq!(reference["schema_version"], 1);
    assert_eq!(reference["kind"], "cpu_bf16_attention_operands");
    assert_eq!(reference["complete"], true);
    assert_eq!(reference["precision"], "bf16");
    let artifact = SafeTensorArtifact::open(root.join("tensors.safetensors")).unwrap();
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(64 * 1024 * 1024).shared();
    let mut attention_ops =
        CudaBf16Attention::new(Arc::clone(&stream), &pool, 8 * 16 * 16).unwrap();
    let mut worst = 0.0_f32;
    for case in reference["cases"].as_array().unwrap() {
        let layer = usize::try_from(case["layer"].as_u64().unwrap()).unwrap();
        let heads = usize::try_from(case["heads"].as_u64().unwrap()).unwrap();
        let kv_heads = usize::try_from(case["kv_heads"].as_u64().unwrap()).unwrap();
        let width = usize::try_from(case["head_dim"].as_u64().unwrap()).unwrap();
        let sequence = usize::try_from(case["sequence"].as_u64().unwrap()).unwrap();
        let window = case["window"].as_u64().map(|w| usize::try_from(w).unwrap());
        let prefix = format!("layer.{layer}");
        let query = tap(&artifact, &format!("{prefix}.query"));
        let keys = tap(&artifact, &format!("{prefix}.key"));
        let values = tap(&artifact, &format!("{prefix}.value"));
        let expected = tap(&artifact, &format!("{prefix}.output"));
        assert_eq!(query.len(), sequence * heads * width);
        assert_eq!(keys.len(), sequence * kv_heads * width);

        // The host decoder is the correctness oracle; confirm it still matches
        // the independent stock capture before judging the device.
        let geometry = AttentionConfig {
            heads,
            kv_heads,
            width,
            window,
        };
        let host = attention(&query, &keys, &values, geometry, 0, 0).unwrap();
        assert_eq!(host, expected, "layer {layer} host parity");

        let mut q = upload(&query, sequence, heads * width, &stream, &pool);
        let mut k = upload(&keys, sequence, kv_heads * width, &stream, &pool);
        let mut v = upload(&values, sequence, kv_heads * width, &stream, &pool);
        let mut out =
            CudaBf16Matrix::allocate(sequence, heads * width, Arc::clone(&stream), &pool, 5)
                .unwrap();
        attention_ops
            .attend_into(
                &q,
                &k,
                &v,
                AttentionGeometry {
                    heads,
                    kv_heads,
                    width,
                    window,
                },
                AttentionSpan {
                    query_start: 0,
                    key_start: 0,
                },
                &mut out,
            )
            .unwrap();
        worst = worst.max(compare(
            &read(&mut out),
            &expected,
            &format!("layer {layer}"),
        ));
        q.close().unwrap();
        k.close().unwrap();
        v.close().unwrap();
        out.close().unwrap();
    }
    println!("attention worst ulp usage {worst}");
    attention_ops.close().unwrap();
    assert_eq!(pool.granted(), 0);
}
