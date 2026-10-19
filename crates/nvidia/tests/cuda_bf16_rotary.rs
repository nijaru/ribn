//! Exact eager BF16 arithmetic with supplied factors, not phase/model acceptance.
#![cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use engine_nvidia::{CudaBf16Error, CudaBf16Matrix, CudaBf16RotaryApply};
use ribn_foundation::BytePool;
use std::sync::Arc;

fn upload(matrix: &mut CudaBf16Matrix, values: &[f64]) {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|&v| nearest(v).to_le_bytes())
        .collect();
    matrix.upload(&bytes).unwrap();
}
fn read(matrix: &mut CudaBf16Matrix) -> Vec<u16> {
    let len = matrix.shape().0 * matrix.shape().1 * 2;
    let mut bytes = Vec::new();
    for offset in (0..len).step_by(257) {
        bytes.extend(
            matrix
                .read_chunk(offset as u64, 257.min(len - offset))
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
fn value(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}
// Independent FP64 nearest-neighbor selection for these bounded normal dyadics.
#[expect(
    clippy::cast_possible_truncation,
    reason = "FP32 only seeds the neighboring BF16 search"
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
fn reference(
    input: &[f64],
    cos: &[f64],
    sin: &[f64],
    heads: usize,
    width: usize,
) -> (Vec<u16>, usize) {
    let mut different = 0;
    let output = input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let channel = i % width;
            let row = i / (heads * width);
            let half = width / 2;
            let (partner, polarity) = if channel < half {
                (i + half, -1.0)
            } else {
                (i - half, 1.0)
            };
            let cosine = cos[row * width + channel];
            let sine = sin[row * width + channel];
            let left = x * cosine;
            let right = polarity * input[partner] * sine;
            let rounded = nearest(value(nearest(left)) + value(nearest(right)));
            different += usize::from(rounded != nearest(left + right));
            rounded
        })
        .collect();
    (output, different)
}

#[test]
#[ignore = "requires CUDA; serialize native BF16 rotary qualification"]
fn split_half_rounds_each_product_and_preserves_validation_output() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(16 * 1024 * 1024).shared();
    let mut execution = CudaBf16RotaryApply::new(Arc::clone(&stream)).unwrap();
    let mut counterexamples = 0;
    for (rows, heads, width) in [(1, 1, 2), (3, 3, 10), (1, 8, 126), (3, 2, 256), (3, 8, 512)] {
        let mut x =
            CudaBf16Matrix::allocate(rows, heads * width, Arc::clone(&stream), &pool, 257).unwrap();
        let mut y =
            CudaBf16Matrix::allocate(rows, heads * width, Arc::clone(&stream), &pool, 257).unwrap();
        let mut c = CudaBf16Matrix::allocate(rows, width, Arc::clone(&stream), &pool, 257).unwrap();
        let mut s = CudaBf16Matrix::allocate(rows, width, Arc::clone(&stream), &pool, 257).unwrap();
        let mut input: Vec<_> = (0..rows * heads * width)
            .map(|i| {
                f64::from(i16::try_from((i * 17) % 65).unwrap() - 32) / 16.0
                    * [1.0, 1.0 / 256.0, 16.0][i % 3]
            })
            .collect();
        input[0] = -0.0;
        input[1] = 0.0;
        let cos: Vec<_> = (0..rows * width)
            .map(|i| f64::from(i16::try_from((i * 7) % 129).unwrap() - 64) / 64.0)
            .collect();
        let sin: Vec<_> = (0..rows * width)
            .map(|i| f64::from(i16::try_from((i * 13) % 129).unwrap() - 64) / 64.0)
            .collect();
        for initialized in 0..3 {
            let error = execution
                .apply_into(&mut c, &mut s, &mut x, &mut y)
                .unwrap_err();
            assert!(matches!(error.cause(), CudaBf16Error::NotReady));
            assert!(!error.retirement_pending());
            assert!(!y.ready());
            match initialized {
                0 => upload(&mut x, &input),
                1 => upload(&mut c, &cos),
                _ => upload(&mut s, &sin),
            }
        }
        for sign in [1.0, -1.0] {
            let sin: Vec<_> = sin.iter().map(|v| v * sign).collect();
            upload(&mut s, &sin);
            execution
                .apply_into(&mut c, &mut s, &mut x, &mut y)
                .unwrap();
            let (expected, different) = reference(&input, &cos, &sin, heads, width);
            counterexamples += different;
            assert_eq!(
                read(&mut y),
                expected,
                "rows={rows} heads={heads} width={width}"
            );
        }
        let before = read(&mut y);
        let mut foreign =
            CudaBf16Matrix::allocate(rows, width, context.new_stream().unwrap(), &pool, 257)
                .unwrap();
        upload(&mut foreign, &cos);
        let error = execution
            .apply_into(&mut foreign, &mut s, &mut x, &mut y)
            .unwrap_err();
        assert!(!error.retirement_pending());
        assert_eq!(read(&mut y), before);
        foreign.close().unwrap();
        c.close().unwrap();
        s.close().unwrap();
        x.close().unwrap();
        y.close().unwrap();
        assert_eq!(pool.granted(), 0);
    }
    assert!(
        counterexamples > 0,
        "single final rounding must fail this fixture"
    );
    execution.close().unwrap();
    execution.close().unwrap();
}
