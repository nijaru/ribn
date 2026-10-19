//! Grouped RMS component acceptance, not a full-model numerical budget.
#![cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use engine_core::{DataType, WeightTensorSpec};
use engine_nvidia::{CudaBf16Error, CudaBf16Matrix, CudaBf16RmsNorm, CudaBf16Weights};
use ribn_foundation::BytePool;
use std::{io::Cursor, sync::Arc};

fn encode(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| {
            assert_eq!(
                v.to_bits() & 0xffff,
                0,
                "fixture must already be BF16 exact"
            );
            u16::try_from(v.to_bits() >> 16).unwrap().to_le_bytes()
        })
        .collect()
}
fn read(matrix: &mut CudaBf16Matrix) -> Vec<u16> {
    let len = matrix.shape().0 * matrix.shape().1 * 2;
    let mut out = Vec::new();
    for offset in (0..len).step_by(5) {
        out.extend(
            matrix
                .read_chunk(offset as u64, 5.min(len - offset))
                .unwrap(),
        );
    }
    out.as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}
fn value(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

// Choose nearest BF16 directly in FP64, including even-tie selection. Searching
// neighbors avoids relying on an FP64 -> FP32 -> BF16 double-rounded reference.
#[expect(
    clippy::cast_possible_truncation,
    reason = "FP32 only seeds a neighboring BF16 search"
)]
fn nearest(exact: f64) -> u16 {
    if exact == 0.0 {
        return if exact.is_sign_negative() { 0x8000 } else { 0 };
    }
    let seed = u16::try_from((exact as f32).to_bits() >> 16).unwrap();
    (seed - 1..=seed + 1)
        .min_by(|&a, &b| {
            (value(a) - exact)
                .abs()
                .total_cmp(&(value(b) - exact).abs())
                .then_with(|| (a & 1).cmp(&(b & 1)))
        })
        .unwrap()
}

fn check(got: &[u16], input: &[f32], gamma: &[f32], scaled: bool) {
    let width = gamma.len();
    for (group, values) in input.chunks_exact(width).enumerate() {
        let energy: f64 = values.iter().map(|&v| f64::from(v).powi(2)).sum();
        let count = f64::from(u32::try_from(width).unwrap());
        let inverse = (energy / count + f64::from(1e-6_f32)).sqrt().recip();
        for (j, &x) in values.iter().enumerate() {
            let exact = f64::from(x) * inverse * if scaled { f64::from(gamma[j]) } else { 1.0 };
            let expected = nearest(exact);
            let actual = got[group * width + j];
            assert!(value(actual).is_finite(), "nonfinite RMS output");
            if exact == 0.0 {
                assert_eq!(actual, expected, "signed zeros are exact");
            } else {
                // PREDECLARED: <=1 adjacent BF16 encoding vs the FP64
                // equation for these bounded finite vectors. Positive
                // FP32 energy reduction gamma_2560 ~1.53e-4 (inverse
                // sqrt halves relative perturbation), FP32 multiplies
                // and documented powf error are much finer than BF16
                // spacing. This permits a final rounding-boundary
                // difference, not a model/logit tolerance or an assumed
                // universal libdevice guarantee. No fast-math is used.
                assert_eq!(actual & 0x8000, expected & 0x8000);
                assert!(
                    actual.abs_diff(expected) <= 1,
                    "width={width} group={group} j={j} scaled={scaled} actual={} reference={exact}",
                    value(actual)
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA; serialize native BF16 normalization qualification"]
fn grouped_rms_matches_independent_fp64_with_direct_and_absent_scale() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(16 * 1024 * 1024).shared();
    let mut rms = CudaBf16RmsNorm::new(Arc::clone(&stream)).unwrap();
    for (rows, groups, width) in [
        (1, 2, 7),
        (3, 2, 63),
        (8, 2, 256),
        (2, 8, 512),
        (3, 2, 2560),
    ] {
        let gamma: Vec<_> = (0..width)
            .map(|j| f32::from(i16::try_from(j % 9).unwrap() - 4) / 4.0)
            .collect();
        let spec = WeightTensorSpec::new("scale", vec![width as u64], DataType::BF16).unwrap();
        let mut weights =
            CudaBf16Weights::allocate(vec![spec], Arc::clone(&stream), &pool, 3).unwrap();
        weights.upload(0, &mut Cursor::new(encode(&gamma))).unwrap();
        let mut x =
            CudaBf16Matrix::allocate(rows, groups * width, Arc::clone(&stream), &pool, 5).unwrap();
        let mut y =
            CudaBf16Matrix::allocate(rows, groups * width, Arc::clone(&stream), &pool, 5).unwrap();
        let error = rms
            .normalize_into(None, width, 1e-6, &mut x, &mut y)
            .unwrap_err();
        assert!(matches!(error.cause(), CudaBf16Error::NotReady));
        assert!(!error.retirement_pending());
        assert!(!y.ready());
        for sign in [1.0, -1.0] {
            let input: Vec<_> = (0..rows * groups)
                .flat_map(|group| {
                    (0..width).map(move |j| {
                        let n = if group == 0 {
                            0
                        } else {
                            i16::try_from((group * 5 + j * 7) % 33).unwrap() - 16
                        };
                        // Mixed exponents force non-exact FP32 energy reductions,
                        // while the FP64 reference still sums these dyadics exactly.
                        let magnitude = [1.0, 1.0 / 256.0, 16.0][j % 3];
                        sign * f32::from(n) / 8.0 * magnitude
                    })
                })
                .collect();
            x.upload(&encode(&input)).unwrap();
            for scaled in [true, false] {
                rms.normalize_into(
                    if scaled {
                        Some((&mut weights, 0))
                    } else {
                        None
                    },
                    width,
                    1e-6,
                    &mut x,
                    &mut y,
                )
                .unwrap();
                let got = read(&mut y);
                assert_eq!(got.len(), input.len());
                check(&got, &input, &gamma, scaled);
            }
        }
        let before = read(&mut y);
        for (bad_width, epsilon) in [
            (0, 1e-6),
            (width + 1, 1e-6),
            (width, 0.0),
            (width, f32::NAN),
        ] {
            let error = rms
                .normalize_into(None, bad_width, epsilon, &mut x, &mut y)
                .unwrap_err();
            assert!(!error.retirement_pending());
            assert_eq!(read(&mut y), before);
        }
        let error = rms
            .normalize_into(Some((&mut weights, 0)), 1, 1e-6, &mut x, &mut y)
            .unwrap_err();
        assert!(!error.retirement_pending());
        assert_eq!(read(&mut y), before, "bad scale width preserves output");
        let foreign_stream = context.new_stream().unwrap();
        let mut foreign =
            CudaBf16Matrix::allocate(rows, groups * width, foreign_stream, &pool, 5).unwrap();
        foreign.upload(&vec![0; rows * groups * width * 2]).unwrap();
        let error = rms
            .normalize_into(None, width, 1e-6, &mut foreign, &mut y)
            .unwrap_err();
        assert!(!error.retirement_pending());
        assert_eq!(read(&mut y), before);
        foreign.close().unwrap();
        x.close().unwrap();
        y.close().unwrap();
        weights.close().unwrap();
        assert_eq!(pool.granted(), 0);
    }
    rms.close().unwrap();
    rms.close().unwrap();
}
