//! Native factor preparation contract; stock-CUDA trig accuracy is gated by the
//! Gemma rotary reference, which reuses this backend with pinned frequencies.
#![cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use engine_nvidia::{CudaBf16Error, CudaBf16Matrix, CudaBf16RotaryFactors};
use ribn_foundation::BytePool;
use std::sync::Arc;

const ONE: u16 = 0x3f80;
const POSITIVE_ZERO: u16 = 0x0000;

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
#[ignore = "requires CUDA; serializes native BF16 rotary-factor qualification"]
fn factors_duplicate_halves_and_preserve_analytic_identities() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(16 * 1024 * 1024).shared();
    let half = 4;
    let width = half * 2;
    let mut factors =
        CudaBf16RotaryFactors::new(Arc::clone(&stream), &pool, &[1.0, 0.0, -1.0, 0.25], 3).unwrap();
    assert_eq!(factors.width(), width);
    assert_eq!(factors.capacity_rows(), 3);
    let mut cosine = CudaBf16Matrix::allocate(3, width, Arc::clone(&stream), &pool, 64).unwrap();
    let mut sine = CudaBf16Matrix::allocate(3, width, Arc::clone(&stream), &pool, 64).unwrap();
    // Factors are outputs: no prior successful publication is required.
    factors
        .prepare_into(&[0, 7, 131_071], &mut cosine, &mut sine)
        .unwrap();
    let cos = read(&mut cosine);
    let sin = read(&mut sine);
    for row in 0..3 {
        for channel in 0..half {
            assert_eq!(
                cos[row * width + channel],
                cos[row * width + channel + half]
            );
            assert_eq!(
                sin[row * width + channel],
                sin[row * width + channel + half]
            );
        }
    }
    // Zero position gives exactly one; the sine keeps the product's zero sign,
    // so a negative frequency yields -0.0 even though the phase is zero.
    let zero_signs: Vec<u16> = [1.0f32, 0.0, -1.0, 0.25]
        .iter()
        .map(|frequency| u16::from(frequency.is_sign_negative()) << 15)
        .collect();
    for channel in 0..half {
        assert_eq!((cos[channel], sin[channel]), (ONE, zero_signs[channel]));
    }
    // Zero frequency is exactly one and positive zero at every position.
    for row in 0..3 {
        assert_eq!(
            (cos[row * width + 1], sin[row * width + 1]),
            (ONE, POSITIVE_ZERO)
        );
    }
    // Opposite frequencies share cosine and negate sine at nonzero positions. This
    // is an observed property of the qualified CUDA math implementation; the stock
    // reference gate owns exact trig accuracy against pinned Transformers.
    for row in 1..3 {
        assert_eq!(cos[row * width], cos[row * width + 2]);
        assert_eq!(
            sin[row * width + 2],
            sin[row * width] ^ 0x8000,
            "odd sine symmetry at row {row}"
        );
    }
    // Fewer rows reuse the capacity. Re-reading row 1's stale position (7) instead
    // of row 0 would change these bits, so this also checks row indexing.
    let mut single = CudaBf16Matrix::allocate(1, width, Arc::clone(&stream), &pool, 64).unwrap();
    let mut single_sin =
        CudaBf16Matrix::allocate(1, width, Arc::clone(&stream), &pool, 64).unwrap();
    factors
        .prepare_into(&[0], &mut single, &mut single_sin)
        .unwrap();
    assert_eq!(read(&mut single), cos[..width].to_vec());
    assert_eq!(read(&mut single_sin), sin[..width].to_vec());
    single.close().unwrap();
    single_sin.close().unwrap();
    cosine.close().unwrap();
    sine.close().unwrap();
    factors.close().unwrap();
    factors.close().unwrap();
    assert_eq!(pool.granted(), 0);
}

#[test]
#[ignore = "requires CUDA; serializes native BF16 rotary-factor qualification"]
fn factor_preflight_rejects_mismatched_rows_and_preserves_outputs() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(16 * 1024 * 1024).shared();
    let width = 8;
    let mut factors =
        CudaBf16RotaryFactors::new(Arc::clone(&stream), &pool, &[1.0, 0.0, -1.0, 0.25], 3).unwrap();
    let mut cosine = CudaBf16Matrix::allocate(3, width, Arc::clone(&stream), &pool, 64).unwrap();
    let mut sine = CudaBf16Matrix::allocate(3, width, Arc::clone(&stream), &pool, 64).unwrap();
    factors
        .prepare_into(&[1, 2, 3], &mut cosine, &mut sine)
        .unwrap();
    let before = read(&mut cosine);
    // Rows must equal positions in both directions; the wide case would otherwise
    // read positions past the uploaded capacity.
    for positions in [vec![], vec![0, 1, 2, 3], vec![1 << 24], vec![0, 1]] {
        let error = factors
            .prepare_into(&positions, &mut cosine, &mut sine)
            .unwrap_err();
        assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
        assert!(!error.retirement_pending());
        assert_eq!(read(&mut cosine), before);
    }
    let mut wide = CudaBf16Matrix::allocate(4, width, Arc::clone(&stream), &pool, 64).unwrap();
    let mut wide_sin = CudaBf16Matrix::allocate(4, width, Arc::clone(&stream), &pool, 64).unwrap();
    let error = factors
        .prepare_into(&[0, 1, 2], &mut wide, &mut wide_sin)
        .unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
    assert!(!error.retirement_pending());
    assert!(!wide.ready() && !wide_sin.ready());
    wide.close().unwrap();
    wide_sin.close().unwrap();
    // More positions than rows would make the kernel write past these matrices.
    let mut narrow = CudaBf16Matrix::allocate(2, width, Arc::clone(&stream), &pool, 64).unwrap();
    let mut narrow_sin =
        CudaBf16Matrix::allocate(2, width, Arc::clone(&stream), &pool, 64).unwrap();
    let error = factors
        .prepare_into(&[0, 1, 2], &mut narrow, &mut narrow_sin)
        .unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
    assert!(!error.retirement_pending());
    assert!(!narrow.ready() && !narrow_sin.ready());
    narrow.close().unwrap();
    narrow_sin.close().unwrap();
    // A foreign stream is rejected before either output is touched.
    let before_sine = read(&mut sine);
    let mut foreign =
        CudaBf16Matrix::allocate(3, width, context.new_stream().unwrap(), &pool, 64).unwrap();
    let error = factors
        .prepare_into(&[1, 2, 3], &mut foreign, &mut sine)
        .unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
    assert!(!error.retirement_pending());
    assert!(!foreign.ready());
    assert_eq!(read(&mut sine), before_sine);
    assert_eq!(read(&mut cosine), before);
    foreign.close().unwrap();
    cosine.close().unwrap();
    sine.close().unwrap();
    factors.close().unwrap();
    assert_eq!(pool.granted(), 0);
    // Constructor validation happens before any charged buffer is published.
    for frequencies in [vec![], vec![0.0, f32::NAN], vec![0.0, f32::INFINITY]] {
        let error = CudaBf16RotaryFactors::new(Arc::clone(&stream), &pool, &frequencies, 1)
            .err()
            .expect("reject invalid frequencies");
        assert!(!error.retirement_pending());
    }
    let error = CudaBf16RotaryFactors::new(Arc::clone(&stream), &pool, &[0.0], 0)
        .err()
        .expect("reject zero capacity");
    assert!(!error.retirement_pending());
    assert_eq!(pool.granted(), 0);
}
