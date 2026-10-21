//! Native BF16 component gates, not full-model numerical/performance acceptance.
#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use engine_core::{DataType, WeightTensorSpec};
use engine_nvidia::{
    CudaBf16Error, CudaBf16Gemm, CudaBf16Matrix, CudaBf16RmsNorm, CudaBf16RotaryApply,
    CudaBf16RotaryFactors, CudaBf16Weights,
};
use ribn_foundation::BytePool;
use std::{io::Cursor, sync::Arc};

#[path = "cuda_bf16_gemm/error_bound.rs"]
mod error_bound;

fn bf16(value: f32) -> u16 {
    assert!(value.is_finite());
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}
fn bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|v| bf16(v).to_le_bytes())
        .collect()
}
fn input_numerator(row: usize, k: usize) -> i16 {
    i16::try_from((row * 5 + k * 3) % 17).unwrap() - 8
}
fn weight_numerator(n: usize, k: usize) -> i16 {
    i16::try_from((n * 7 + k * 11) % 19).unwrap() - 9
}
fn read(matrix: &mut CudaBf16Matrix) -> Vec<u16> {
    let (rows, cols) = matrix.shape();
    let mut data = Vec::new();
    for offset in (0..rows * cols * 2).step_by(5) {
        data.extend(
            matrix
                .read_chunk(offset as u64, 5.min(rows * cols * 2 - offset))
                .unwrap(),
        );
    }
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

#[test]
#[ignore = "requires CUDA; serialize device qualification"]
fn native_bf16_linear_matches_exact_dot_products_and_reuses_output() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let pool = BytePool::new(128 * 1024 * 1024).shared();
    let cases = [
        (1, 7, 11),
        (3, 63, 9),
        (8, 256, 32),
        (2, 512, 17),
        (1, 2560, 16),
    ];
    let specs = cases
        .iter()
        .enumerate()
        .map(|(i, &(_, k, n))| {
            WeightTensorSpec::new(format!("w{i}"), vec![n as u64, k as u64], DataType::BF16)
                .unwrap()
        })
        .collect();
    let mut weights = CudaBf16Weights::allocate(specs, Arc::clone(&stream), &pool, 3).unwrap();
    for (slot, &(_, k, n)) in cases.iter().enumerate() {
        let data = bytes(
            (0..n).flat_map(|out| (0..k).map(move |j| f32::from(weight_numerator(out, j)) / 8.0)),
        );
        weights.upload(slot, &mut Cursor::new(data)).unwrap();
    }
    let weight_charge = weights.allocated_bytes();
    let workspace_bytes = 8 * 1024 * 1024;
    let mut gemm = CudaBf16Gemm::new(Arc::clone(&stream), &pool, workspace_bytes).unwrap();
    for (slot, &(m, k, n)) in cases.iter().enumerate() {
        let mut input = CudaBf16Matrix::allocate(m, k, Arc::clone(&stream), &pool, 5).unwrap();
        let mut output = CudaBf16Matrix::allocate(m, n, Arc::clone(&stream), &pool, 5).unwrap();
        assert!(!output.ready());
        let error = gemm
            .linear_into(&mut weights, slot, &mut input, &mut output)
            .unwrap_err();
        assert!(matches!(error.cause(), CudaBf16Error::NotReady));
        assert!(!error.retirement_pending());
        for repetition in [1.0, -1.0] {
            input
                .upload(&bytes((0..m).flat_map(|row| {
                    (0..k).map(move |j| repetition * f32::from(input_numerator(row, j)) / 8.0)
                })))
                .unwrap();
            gemm.linear_into(&mut weights, slot, &mut input, &mut output)
                .unwrap();
            let got = read(&mut output);
            assert_eq!(got.len(), m * n);
            for row in 0..m {
                for out in 0..n {
                    // Integer sum is an independent exact reference. Products and
                    // every partial sum are multiples of 1/64 fitting FP32 exactly
                    // for these bounded numerators/K, regardless of reduction order.
                    let sum: i32 = (0..k)
                        .map(|j| {
                            i32::from(input_numerator(row, j)) * i32::from(weight_numerator(out, j))
                        })
                        .sum();
                    assert!(sum.unsigned_abs() < (1 << 24));
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "asserted integer sum fits FP32 exactly"
                    )]
                    let exact = sum as f32;
                    let expected = bf16(repetition * (exact / 64.0));
                    assert_ne!(got[row * n + out] & 0x7f80, 0x7f80, "nonfinite output");
                    // Exact cancellation yields +0 irrespective of input sign.
                    let expected = if sum == 0 { 0 } else { expected };
                    assert_eq!(
                        got[row * n + out],
                        expected,
                        "slot={slot} row={row} column={out}"
                    );
                }
            }
        }
        let before = read(&mut output);
        let wrong_stream = context.new_stream().unwrap();
        let mut foreign = CudaBf16Matrix::allocate(m, k, wrong_stream, &pool, 5).unwrap();
        foreign.upload(&vec![0; m * k * 2]).unwrap();
        let error = gemm
            .linear_into(&mut weights, slot, &mut foreign, &mut output)
            .unwrap_err();
        assert!(!error.retirement_pending());
        assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
        assert_eq!(
            read(&mut output),
            before,
            "whole-call validation preserves previous output"
        );
        foreign.close().unwrap();
        input.close().unwrap();
        output.close().unwrap();
        assert_eq!(pool.granted(), weight_charge + workspace_bytes as u64);
    }
    gemm.close().unwrap();
    gemm.close().unwrap();
    weights.close().unwrap();
    assert_eq!(pool.granted(), 0);
}

#[test]
#[ignore = "requires CUDA; validates Send-owner stream affinity before allocation"]
fn per_thread_default_stream_is_rejected_even_after_cross_thread_move() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.per_thread_stream();
    let pool = BytePool::new(16 * 1024 * 1024).shared();
    for cross_thread in [false, true] {
        let stream = Arc::clone(&stream);
        let pool = Arc::clone(&pool);
        let check = move || {
            let spec = WeightTensorSpec::new("w", vec![1, 1], DataType::BF16).unwrap();
            let weights = CudaBf16Weights::allocate(vec![spec], Arc::clone(&stream), &pool, 3);
            assert!(matches!(weights, Err(CudaBf16Error::Invalid(_))));
            assert_eq!(pool.granted(), 0);
            let matrix = CudaBf16Matrix::allocate(1, 1, Arc::clone(&stream), &pool, 3);
            assert!(matches!(matrix, Err(CudaBf16Error::Invalid(_))));
            assert_eq!(pool.granted(), 0);
            let error = CudaBf16Gemm::new(Arc::clone(&stream), &pool, 8 * 1024 * 1024)
                .err()
                .expect("reject unstable stream before handle creation");
            assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
            assert!(!error.retirement_pending());
            assert_eq!(pool.granted(), 0);
            let error = CudaBf16RmsNorm::new(Arc::clone(&stream))
                .err()
                .expect("reject unstable stream before module loading");
            assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
            assert!(!error.retirement_pending());
            let error = CudaBf16RotaryApply::new(Arc::clone(&stream))
                .err()
                .expect("reject unstable stream before rotary module loading");
            assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
            assert!(!error.retirement_pending());
            let error = CudaBf16RotaryFactors::new(Arc::clone(&stream), &pool, &[1.0], 1)
                .err()
                .expect("reject unstable stream before factor buffers");
            assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
            assert!(!error.retirement_pending());
            assert_eq!(pool.granted(), 0);
        };
        if cross_thread {
            std::thread::spawn(check).join().unwrap();
        } else {
            check();
        }
    }
}
