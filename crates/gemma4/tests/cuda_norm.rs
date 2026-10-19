//! Model-owned grouping, direct scales and owner-only unscaled V; not a decoder.
#![cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};
use engine_gemma4::{Gemma4CudaWeights, Gemma4Source, LayerParameter, ParameterId};
use engine_nvidia::{CudaBf16Matrix, CudaBf16RmsNorm};
use ribn_foundation::BytePool;
use std::{io::Read, sync::Arc};
#[path = "common/weights.rs"]
mod weights;
use weights::{Dir, inventory, package};

fn read(matrix: &mut CudaBf16Matrix) -> Vec<u16> {
    let len = matrix.shape().0 * matrix.shape().1 * 2;
    let mut output = Vec::new();
    for offset in (0..len).step_by(5) {
        output.extend(
            matrix
                .read_chunk(offset as u64, 5.min(len - offset))
                .unwrap(),
        );
    }
    output
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}
fn f64_value(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}
fn check(output: &[u16], input: &[f32], scale: &[f64]) {
    let width = scale.len();
    for (group, x) in input.chunks_exact(width).enumerate() {
        let energy: f64 = x.iter().map(|&v| f64::from(v).powi(2)).sum();
        let inverse = (energy / f64::from(u32::try_from(width).unwrap()) + f64::from(1e-6_f32))
            .sqrt()
            .recip();
        for (j, &x) in x.iter().enumerate() {
            let exact = f64::from(x) * inverse * scale[j];
            let actual = output[group * width + j];
            assert!(f64_value(actual).is_finite());
            if exact == 0.0 {
                assert_eq!(actual, if exact.is_sign_negative() { 0x8000 } else { 0 });
                continue;
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "seed only; nearest BF16 selected in FP64"
            )]
            let seed = u16::try_from((exact as f32).to_bits() >> 16).unwrap();
            let expected = (seed - 1..=seed + 1)
                .min_by(|&a, &b| {
                    (f64_value(a) - exact)
                        .abs()
                        .total_cmp(&(f64_value(b) - exact).abs())
                        .then_with(|| (a & 1).cmp(&(b & 1)))
                })
                .unwrap();
            // Same predeclared component policy as the independent backend gate:
            // one adjacent BF16 encoding vs FP64, finite bounded inputs only.
            assert_eq!(actual & 0x8000, expected & 0x8000);
            assert!(actual.abs_diff(expected) <= 1, "group={group} j={j}");
        }
    }
}
fn input(columns: usize) -> Vec<f32> {
    (0..3 * columns)
        .map(|j| f32::from(i16::try_from(j % 13).unwrap() - 6) / 2048.0)
        .collect()
}
fn upload(matrix: &mut CudaBf16Matrix, input: &[f32]) {
    let bytes: Vec<_> = input
        .iter()
        .flat_map(|v| {
            assert_eq!(v.to_bits() & 0xffff, 0);
            u16::try_from(v.to_bits() >> 16).unwrap().to_le_bytes()
        })
        .collect();
    matrix.upload(&bytes).unwrap();
}
fn check_values(
    model: &mut Gemma4CudaWeights,
    execution: &mut CudaBf16RmsNorm,
    stream: &Arc<CudaStream>,
    pool: &Arc<BytePool>,
) {
    for (layer, columns) in [(0, 4), (1, 8)] {
        let values = input(columns);
        let mut x = CudaBf16Matrix::allocate(3, columns, Arc::clone(stream), pool, 5).unwrap();
        let mut y = CudaBf16Matrix::allocate(3, columns, Arc::clone(stream), pool, 5).unwrap();
        upload(&mut x, &values);
        model
            .value_norm_into(layer, execution, &mut x, &mut y)
            .unwrap();
        let before = read(&mut y);
        check(&before, &values, &vec![1.0; columns]);
        let error = model
            .value_norm_into(layer + 2, execution, &mut x, &mut y)
            .unwrap_err();
        assert!(!error.retirement_pending());
        assert_eq!(
            read(&mut y),
            before,
            "shared tails must not renormalize cached V"
        );
        x.close().unwrap();
        y.close().unwrap();
    }
}

#[test]
#[ignore = "requires CUDA; serialize native BF16 model normalization qualification"]
fn model_norms_bind_direct_scales_grouping_and_unscaled_owner_values() {
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&package).unwrap();
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(16 * 1024 * 1024).shared();
    let mut model = Gemma4CudaWeights::load(&source, Arc::clone(&stream), &pool, 5).unwrap();
    let allocation = model.allocation_id();
    let mut execution = CudaBf16RmsNorm::new(Arc::clone(&stream)).unwrap();
    for (role, columns) in [
        (ParameterId::FinalNorm, 7),
        (ParameterId::PleNorm, 8),
        (
            ParameterId::Layer {
                layer: 0,
                parameter: LayerParameter::InputNorm,
            },
            7,
        ),
        (
            ParameterId::Layer {
                layer: 0,
                parameter: LayerParameter::QueryNorm,
            },
            8,
        ),
        (
            ParameterId::Layer {
                layer: 1,
                parameter: LayerParameter::QueryNorm,
            },
            16,
        ),
        (
            ParameterId::Layer {
                layer: 1,
                parameter: LayerParameter::KeyNorm,
            },
            8,
        ),
        (
            ParameterId::Layer {
                layer: 3,
                parameter: LayerParameter::PostPleNorm,
            },
            7,
        ),
    ] {
        let mut bytes = Vec::new();
        source.parameters()[&role]
            .reader()
            .read_to_end(&mut bytes)
            .unwrap();
        let scale: Vec<_> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f64_value(u16::from_le_bytes(*b)))
            .collect();
        let values = input(columns);
        let mut x = CudaBf16Matrix::allocate(3, columns, Arc::clone(&stream), &pool, 5).unwrap();
        let mut y = CudaBf16Matrix::allocate(3, columns, Arc::clone(&stream), &pool, 5).unwrap();
        upload(&mut x, &values);
        model
            .rms_norm_into(role, &mut execution, &mut x, &mut y)
            .unwrap();
        let before = read(&mut y);
        check(&before, &values, &scale);
        let bad = ParameterId::Layer {
            layer: 0,
            parameter: LayerParameter::LayerScalar,
        };
        let error = model
            .rms_norm_into(bad, &mut execution, &mut x, &mut y)
            .unwrap_err();
        assert!(!error.retirement_pending());
        assert_eq!(read(&mut y), before);
        if columns == 8 {
            // Inactive K, and active Q with a missing query head. Both fit a
            // generic width-8 RMS group, but neither is valid model execution.
            for parameter in [LayerParameter::KeyNorm, LayerParameter::QueryNorm] {
                let invalid = ParameterId::Layer {
                    layer: 3,
                    parameter,
                };
                let error = model
                    .rms_norm_into(invalid, &mut execution, &mut x, &mut y)
                    .unwrap_err();
                assert!(!error.retirement_pending());
                assert_eq!(read(&mut y), before);
            }
        }
        x.close().unwrap();
        y.close().unwrap();
    }
    check_values(&mut model, &mut execution, &stream, &pool);
    assert_eq!(model.allocation_id(), allocation);
    execution.close().unwrap();
    model.close().unwrap();
    assert_eq!(pool.granted(), 0);
}
