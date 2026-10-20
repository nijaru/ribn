//! Elementwise BF16 component acceptance, not a full-model numerical budget.
#![cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use engine_nvidia::{CudaBf16Elementwise, CudaBf16Error, CudaBf16Matrix};
use ribn_foundation::BytePool;
use std::sync::Arc;

fn bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn number(value: f32) -> u16 {
    assert!(
        value.to_bits().trailing_zeros() >= 16 || value == 0.0,
        "fixture must be BF16 exact"
    );
    (value.to_bits() >> 16) as u16
}
fn value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
// Independent host RNE reference for the finite device fixtures.
fn round(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
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
fn matrix(
    values: &[u16],
    rows: usize,
    cols: usize,
    stream: Arc<cudarc::driver::CudaStream>,
    pool: &Arc<BytePool>,
) -> CudaBf16Matrix {
    assert_eq!(values.len(), rows * cols);
    let mut matrix = CudaBf16Matrix::allocate(rows, cols, stream, pool, 5).unwrap();
    matrix.upload(&bytes(values)).unwrap();
    matrix
}
fn assert_ulp(got: &[u16], expected: &[u16], label: &str) {
    assert_eq!(got.len(), expected.len(), "{label} length");
    for (index, (&actual, &wanted)) in got.iter().zip(expected).enumerate() {
        assert!(value(actual).is_finite(), "{label} {index} is not finite");
        if value(wanted) == 0.0 {
            assert_eq!(actual, wanted, "{label} {index} signed zero");
            continue;
        }
        // PREDECLARED: one adjacent BF16 encoding. Every operation rounds once;
        // soft cap's three stored roundings are reproduced here. FP32 tanh error
        // is far below BF16 spacing for these bounded inputs.
        assert_eq!(actual & 0x8000, wanted & 0x8000, "{label} {index} sign");
        assert!(
            actual.abs_diff(wanted) <= 1,
            "{label} {index}: device {} host {}",
            value(actual),
            value(wanted)
        );
    }
}

#[test]
#[ignore = "requires CUDA; serialize native BF16 elementwise qualification"]
fn elementwise_matches_exact_dyadics_and_host_gelu_softcap() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(16 * 1024 * 1024).shared();
    let mut ops = CudaBf16Elementwise::new(Arc::clone(&stream)).unwrap();

    for (rows, cols) in [(2_usize, 8_usize), (1, 17), (3, 5)] {
        let len = rows * cols;
        let left: Vec<u16> = (0..len)
            .map(|i| number([1.5, -2.0, 0.5, 0.25][i % 4]))
            .collect();
        let right: Vec<u16> = (0..len)
            .map(|i| number([2.0, 0.5, -4.0, 8.0][i % 4]))
            .collect();
        let mut left_matrix = matrix(&left, rows, cols, Arc::clone(&stream), &pool);
        let mut right_matrix = matrix(&right, rows, cols, Arc::clone(&stream), &pool);
        let mut out = CudaBf16Matrix::allocate(rows, cols, Arc::clone(&stream), &pool, 5).unwrap();

        ops.add_into(&left_matrix, &right_matrix, &mut out).unwrap();
        // 1.5+2=3.5, -2+0.5=-1.5, 0.5-4=-3.5, 0.25+8=8.25.
        let added: Vec<u16> = (0..len)
            .map(|i| number([3.5, -1.5, -3.5, 8.25][i % 4]))
            .collect();
        assert_eq!(read(&mut out), added);

        ops.multiply_into(&left_matrix, &right_matrix, &mut out)
            .unwrap();
        let multiplied: Vec<u16> = (0..len)
            .map(|i| number([3.0, -1.0, -2.0, 2.0][i % 4]))
            .collect();
        assert_eq!(read(&mut out), multiplied);

        ops.scale_into(&left_matrix, 0.5, &mut out).unwrap();
        let scaled: Vec<u16> = (0..len)
            .map(|i| number([0.75, -1.0, 0.25, 0.125][i % 4]))
            .collect();
        assert_eq!(read(&mut out), scaled);

        // Host policy: BF16 sum, then BF16 multiply by 1/sqrt(2).
        let context: Vec<u16> = (0..len)
            .map(|i| number([1.0, 3.0, -1.0, 0.0][i % 4]))
            .collect();
        let token: Vec<u16> = (0..len)
            .map(|i| number([1.0, 1.0, 3.0, 0.0][i % 4]))
            .collect();
        let mut context_matrix = matrix(&context, rows, cols, Arc::clone(&stream), &pool);
        let mut token_matrix = matrix(&token, rows, cols, Arc::clone(&stream), &pool);
        ops.ple_combine_into(&context_matrix, &token_matrix, &mut out)
            .unwrap();
        let combined: Vec<u16> = context
            .iter()
            .zip(&token)
            .map(|(&x, &y)| {
                let sum = round(value(x) + value(y));
                round(value(sum) * std::f32::consts::FRAC_1_SQRT_2)
            })
            .collect();
        assert_ulp(&read(&mut out), &combined, "ple combine");

        let gelu_input: Vec<u16> = (0..len)
            .map(|i| number([0.0, 1.0, -1.0, 3.0, -3.0, 10.0][i % 6]))
            .collect();
        let mut gelu_matrix = matrix(&gelu_input, rows, cols, Arc::clone(&stream), &pool);
        ops.gelu_tanh_into(&gelu_matrix, &mut out).unwrap();
        let gelu: Vec<u16> = gelu_input
            .iter()
            .map(|&bits| {
                let x = value(bits);
                let inner = 0.797_884_6_f32 * x * (1.0 + 0.044_715 * x * x);
                round(0.5 * x * (1.0 + inner.tanh()))
            })
            .collect();
        assert_ulp(&read(&mut out), &gelu, "gelu");

        let logits: Vec<u16> = (0..len)
            .map(|i| number([0.0, 1.0, -1.0, 30.0, -30.0, 60.0][i % 6]))
            .collect();
        let mut logit_matrix = matrix(&logits, rows, cols, Arc::clone(&stream), &pool);
        ops.softcap_into(&logit_matrix, 30.0, &mut out).unwrap();
        let capped: Vec<u16> = logits
            .iter()
            .map(|&bits| {
                let divided = round(value(bits) / 30.0);
                let hyperbolic = round(value(divided).tanh());
                round(value(hyperbolic) * 30.0)
            })
            .collect();
        assert_ulp(&read(&mut out), &capped, "softcap");

        left_matrix.close().unwrap();
        right_matrix.close().unwrap();
        context_matrix.close().unwrap();
        token_matrix.close().unwrap();
        gelu_matrix.close().unwrap();
        logit_matrix.close().unwrap();
        out.close().unwrap();
    }
    assert_eq!(pool.granted(), 0);
    ops.close().unwrap();
    ops.close().unwrap();
}

#[test]
#[ignore = "requires CUDA; serialize native BF16 elementwise qualification"]
fn elementwise_rejects_invalid_operands_before_enqueue() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(16 * 1024 * 1024).shared();
    let mut ops = CudaBf16Elementwise::new(Arc::clone(&stream)).unwrap();
    let mut input = matrix(&[number(1.0); 6], 2, 3, Arc::clone(&stream), &pool);
    let mut other = matrix(&[number(2.0); 6], 2, 3, Arc::clone(&stream), &pool);
    let mut wide = matrix(&[number(3.0); 8], 2, 4, Arc::clone(&stream), &pool);
    let mut output = CudaBf16Matrix::allocate(2, 3, Arc::clone(&stream), &pool, 5).unwrap();

    let error = ops.add_into(&input, &wide, &mut output).unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
    assert!(!error.retirement_pending());
    assert!(!output.ready());

    let error = ops.scale_into(&input, f32::NAN, &mut output).unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
    assert!(!error.retirement_pending());
    let error = ops.softcap_into(&input, 0.0, &mut output).unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
    assert!(!error.retirement_pending());

    let foreign_stream = context.new_stream().unwrap();
    let mut foreign = matrix(&[number(1.0); 6], 2, 3, foreign_stream, &pool);
    let error = ops.gelu_tanh_into(&foreign, &mut output).unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
    assert!(!error.retirement_pending());

    ops.add_into(&input, &other, &mut output).unwrap();
    let before = read(&mut output);
    let error = ops.add_into(&input, &wide, &mut output).unwrap_err();
    assert!(!error.retirement_pending());
    assert_eq!(read(&mut output), before, "invalid call preserves output");

    input.close().unwrap();
    other.close().unwrap();
    wide.close().unwrap();
    foreign.close().unwrap();
    output.close().unwrap();
    assert_eq!(pool.granted(), 0);
    ops.close().unwrap();
}
