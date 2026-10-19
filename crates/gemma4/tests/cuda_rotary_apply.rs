//! Native rotary execution against pinned stock CUDA operands and factors.
#![cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use engine_gemma4::{AttentionKind, Gemma4Config};
use engine_nvidia::{CudaBf16Matrix, CudaBf16RotaryApply, CudaBf16RotaryFactors};
use ribn_foundation::BytePool;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{path::PathBuf, sync::Arc};

const CONFIG: &str = include_str!("../../../benchmarks/gemma4-reference/config.json");
const HOST_FIXTURE: &str = include_str!("fixtures/rotary.json");
const GENERATOR: &[u8] = include_bytes!("../../../benchmarks/gemma4_rotary_reference.py");

fn reference() -> Value {
    let path = PathBuf::from(
        std::env::var_os("RIBN_ROTARY_REFERENCE").expect("generate stock CUDA rotary reference"),
    );
    assert!(path.metadata().unwrap().len() <= 8 * 1024 * 1024);
    let reference: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(reference["schema_version"], 1);
    assert_eq!(reference["kind"], "cuda_bf16_rotary_apply");
    assert_eq!(
        reference["generator_sha256"],
        format!("{:x}", Sha256::digest(GENERATOR))
    );
    assert_eq!(
        reference["config_sha256"],
        format!("{:x}", Sha256::digest(CONFIG.as_bytes()))
    );
    assert_eq!(
        reference["frequency_preparation"],
        "CPU FP32, transfer without dtype conversion"
    );
    assert_eq!(reference["packages"]["transformers"], "5.20.0");
    reference
}

fn config(case: &Value) -> Gemma4Config {
    let mut root: Value = serde_json::from_str(CONFIG).unwrap();
    for (key, value) in case["text_overrides"].as_object().unwrap() {
        root["text_config"][key] = value.clone();
    }
    Gemma4Config::from_json(&root).unwrap()
}

fn bits(value: &Value, count: usize) -> Vec<u16> {
    let values: Vec<_> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u16::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(values.len(), count);
    assert!(
        values
            .iter()
            .all(|&v| f32::from_bits(u32::from(v) << 16).is_finite())
    );
    values
}

fn upload(matrix: &mut CudaBf16Matrix, values: &[u16]) {
    matrix
        .upload(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
}

fn read(matrix: &mut CudaBf16Matrix) -> Vec<u16> {
    let len = matrix.shape().0 * matrix.shape().1 * 2;
    let mut bytes = Vec::new();
    for offset in (0..len).step_by(4096) {
        bytes.extend(
            matrix
                .read_chunk(offset as u64, 4096.min(len - offset))
                .unwrap(),
        );
    }
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| u16::from_le_bytes(*v))
        .collect()
}

#[test]
#[ignore = "requires CUDA and RIBN_ROTARY_REFERENCE from --apply-cuda; see benchmark guide"]
fn stock_cuda_eager_factors_preserve_all_rotation_bits() {
    let reference = reference();
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(16 * 1024 * 1024).shared();
    let mut execution = CudaBf16RotaryApply::new(Arc::clone(&stream)).unwrap();
    let cases = reference["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for case in cases {
        let config = config(case);
        for kind in ["sliding_attention", "full_attention"] {
            let expected_kind = if kind == "sliding_attention" {
                AttentionKind::Sliding
            } else {
                AttentionKind::Full
            };
            let width = config
                .layers()
                .iter()
                .find(|g| g.kind == expected_kind)
                .unwrap()
                .head_dim;
            assert_eq!(
                case["kinds"][kind]["head_dim"].as_u64().unwrap(),
                width as u64
            );
            let applications = case["kinds"][kind]["applications"].as_array().unwrap();
            assert_eq!(applications.len(), 2);
            for application in applications {
                let heads = usize::try_from(application["heads"].as_u64().unwrap()).unwrap();
                assert!(heads > 0 && heads <= 8);
                let max = config.max_position_embeddings();
                assert_eq!(
                    application["positions"],
                    serde_json::json!([4095, 0, 511, max - 1, 512, 513, max / 2 - 1, 4095, 1])
                );
                assert!(application["single_rounding_mismatches"].as_u64().unwrap() > 0);
                let mut x =
                    CudaBf16Matrix::allocate(9, heads * width, Arc::clone(&stream), &pool, 4096)
                        .unwrap();
                let mut y =
                    CudaBf16Matrix::allocate(9, heads * width, Arc::clone(&stream), &pool, 4096)
                        .unwrap();
                let mut c =
                    CudaBf16Matrix::allocate(9, width, Arc::clone(&stream), &pool, 4096).unwrap();
                let mut s =
                    CudaBf16Matrix::allocate(9, width, Arc::clone(&stream), &pool, 4096).unwrap();
                upload(&mut x, &bits(&application["input_bits"], 9 * heads * width));
                upload(&mut c, &bits(&application["cos_bits"], 9 * width));
                upload(&mut s, &bits(&application["sin_bits"], 9 * width));
                execution
                    .apply_into(&mut c, &mut s, &mut x, &mut y)
                    .unwrap();
                let actual = read(&mut y);
                let expected = bits(&application["output_bits"], 9 * heads * width);
                assert_eq!(actual.len(), expected.len());
                assert_eq!(
                    actual.iter().zip(&expected).position(|(a, b)| a != b),
                    None,
                    "{} {kind} heads={heads}: first differing element",
                    case["name"]
                );
                // Exercise reuse without treating a previous output as a reservation.
                execution
                    .apply_into(&mut c, &mut s, &mut x, &mut y)
                    .unwrap();
                assert!(read(&mut y).iter().zip(&expected).all(|(a, b)| a == b));
                x.close().unwrap();
                y.close().unwrap();
                c.close().unwrap();
                s.close().unwrap();
                assert_eq!(pool.granted(), 0);
            }
        }
    }
    execution.close().unwrap();
}

struct Expectation {
    positions: Vec<u32>,
    heads: usize,
    cos: Vec<u16>,
    sin: Vec<u16>,
    input: Vec<u16>,
    output: Vec<u16>,
}

fn expectations(case: &Value, kind: &str, width: usize) -> Vec<Expectation> {
    case["kinds"][kind]["applications"]
        .as_array()
        .unwrap()
        .iter()
        .map(|application| {
            let heads = usize::try_from(application["heads"].as_u64().unwrap()).unwrap();
            Expectation {
                positions: application["positions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                    .collect(),
                heads,
                cos: bits(&application["cos_bits"], 9 * width),
                sin: bits(&application["sin_bits"], 9 * width),
                input: bits(&application["input_bits"], 9 * heads * width),
                output: bits(&application["output_bits"], 9 * heads * width),
            }
        })
        .collect()
}

fn check_kind(
    case: &Value,
    kind: &str,
    frequencies: &[f32],
    expectations: &[Expectation],
    stream: &Arc<cudarc::driver::CudaStream>,
    pool: &Arc<BytePool>,
    apply: &mut CudaBf16RotaryApply,
) {
    let width = frequencies.len() * 2;
    let name = &case["name"];
    let mut factors = CudaBf16RotaryFactors::new(Arc::clone(stream), pool, frequencies, 9).unwrap();
    assert_eq!(factors.width(), width);
    let mut cosine = CudaBf16Matrix::allocate(9, width, Arc::clone(stream), pool, 4096).unwrap();
    let mut sine = CudaBf16Matrix::allocate(9, width, Arc::clone(stream), pool, 4096).unwrap();
    let charged = cosine.allocated_bytes()
        + sine.allocated_bytes()
        + u64::from(u32::try_from(width / 2).unwrap()) * 4
        + 9 * 4;
    for expectation in expectations {
        factors
            .prepare_into(&expectation.positions, &mut cosine, &mut sine)
            .unwrap();
        assert_eq!(&read(&mut cosine), &expectation.cos, "{name} {kind} cosine");
        assert_eq!(&read(&mut sine), &expectation.sin, "{name} {kind} sine");
        let heads = expectation.heads;
        let mut x =
            CudaBf16Matrix::allocate(9, heads * width, Arc::clone(stream), pool, 4096).unwrap();
        let mut y =
            CudaBf16Matrix::allocate(9, heads * width, Arc::clone(stream), pool, 4096).unwrap();
        upload(&mut x, &expectation.input);
        apply
            .apply_into(&mut cosine, &mut sine, &mut x, &mut y)
            .unwrap();
        assert_eq!(
            read(&mut y)
                .iter()
                .zip(&expectation.output)
                .position(|(a, b)| a != b),
            None,
            "{name} {kind} heads={heads}: first differing element"
        );
        x.close().unwrap();
        y.close().unwrap();
        assert_eq!(pool.granted(), charged);
    }
    cosine.close().unwrap();
    sine.close().unwrap();
    factors.close().unwrap();
    assert_eq!(pool.granted(), 0);
}

#[test]
#[ignore = "requires CUDA and RIBN_ROTARY_REFERENCE from --apply-cuda; see benchmark guide"]
fn native_factor_preparation_reproduces_stock_cosine_and_sine() {
    let reference = reference();
    let host: Value = serde_json::from_str(HOST_FIXTURE).unwrap();
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(16 * 1024 * 1024).shared();
    let mut apply = CudaBf16RotaryApply::new(Arc::clone(&stream)).unwrap();
    let cases = reference["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for (case, host_case) in cases.iter().zip(host["cases"].as_array().unwrap()) {
        assert_eq!(case["name"], host_case["name"]);
        let config = config(case);
        for kind in ["sliding_attention", "full_attention"] {
            // The device fixture uses exactly the host-qualified model vector.
            assert_eq!(
                case["kinds"][kind]["inverse_frequency_bits"],
                host_case["kinds"][kind]["inverse_frequency_bits"],
                "{} {kind} frequencies differ from the host gate",
                case["name"]
            );
            let frequencies: Vec<f32> = case["kinds"][kind]["inverse_frequency_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| f32::from_bits(u32::try_from(v.as_u64().unwrap()).unwrap()))
                .collect();
            let width = frequencies.len() * 2;
            let expected_kind = if kind == "sliding_attention" {
                AttentionKind::Sliding
            } else {
                AttentionKind::Full
            };
            let head_dim = config
                .layers()
                .iter()
                .find(|g| g.kind == expected_kind)
                .unwrap()
                .head_dim;
            assert_eq!(width, head_dim, "{} {kind} width", case["name"]);
            let expectations = expectations(case, kind, width);
            check_kind(
                case,
                kind,
                &frequencies,
                &expectations,
                &stream,
                &pool,
                &mut apply,
            );
        }
    }
    apply.close().unwrap();
}
