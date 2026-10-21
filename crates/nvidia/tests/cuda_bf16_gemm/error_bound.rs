//! Reordered FP32 accumulation against exact encoded-operand integer dots.
use super::*;

// Every operand is a signed integer in units of 2^-11, with a BF16-sized
// significand and exponent in [-4, 4]. Products are multiples of 2^-22. Even
// sum(abs(products)) for K=10240 is below 2^46 in those integer units, so both
// integer references and their FP64 conversions are exact. Products, nonzero
// partial sums and BF16 outputs stay far away from underflow and overflow.
fn operand(index: usize) -> u16 {
    let mixed = index.wrapping_mul(131).wrapping_add(73);
    let mantissa = u16::try_from(mixed % 128).unwrap();
    let exponent = u16::try_from((mixed / 128) % 9).unwrap() + 123;
    let sign = u16::try_from((mixed / 1152) % 2).unwrap() << 15;
    sign | (exponent << 7) | mantissa
}

fn units(bits: u16) -> i32 {
    let magnitude = i32::from(128 + (bits & 127)) << (((bits >> 7) & 255) - 123);
    if bits & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn assert_forward_error(
    actual: &[u8],
    input_units: &[i32],
    weight_units: &[i32],
    shape: (usize, usize, usize),
) {
    let (m, n, k) = shape;
    let ku = f64::from(u32::try_from(k).unwrap()) * 2.0_f64.powi(-23);
    let gamma = ku / (1.0 - ku);
    // One full BF16 ULP is conservative (RNE unit roundoff is 2^-8).
    let bf16_roundoff = 2.0_f64.powi(-7);
    for row in 0..m {
        for column in 0..n {
            let (dot, absolute): (i64, i64) = input_units[row * k..(row + 1) * k]
                .iter()
                .zip(&weight_units[column * k..(column + 1) * k])
                .map(|(&x, &w)| i64::from(x) * i64::from(w))
                .fold((0, 0), |(dot, absolute), product| {
                    (dot + product, absolute + product.abs())
                });
            assert!(absolute < (1_i64 << 46));
            #[expect(
                clippy::cast_precision_loss,
                reason = "integer magnitudes below 2^46 are exact in FP64"
            )]
            let (reference, magnitude) = (
                dot as f64 * 2.0_f64.powi(-22),
                absolute as f64 * 2.0_f64.powi(-22),
            );
            let bits = u16::from_le_bytes([
                actual[(row * n + column) * 2],
                actual[(row * n + column) * 2 + 1],
            ]);
            let got = f64::from(f32::from_bits(u32::from(bits) << 16));
            assert!(got.is_finite());
            let accumulated = gamma * magnitude;
            let bound = (accumulated + bf16_roundoff * (reference.abs() + accumulated))
                * (1.0 + 16.0 * f64::EPSILON);
            assert!(
                (got - reference).abs() <= bound,
                "M={m} N={n} K={k} row={row} column={column}: got={got}, exact={reference}, bound={bound}"
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA; full-geometry FP32 error bound; serialize release device runs"]
fn model_projection_shapes_satisfy_an_independent_accumulation_bound() {
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let pool = BytePool::new(128 * 1024 * 1024).shared();
    // All changed E4B projection geometries, including attention outputs. The
    // vocabulary head stays one-row and is not a selected batched projection.
    for (k, n) in [
        (2560, 10752),
        (2560, 2048),
        (2560, 512),
        (2560, 4096),
        (2560, 1024),
        (2560, 2560),
        (2560, 10240),
        (2560, 256),
        (2048, 2560),
        (4096, 2560),
        (10240, 2560),
        (256, 2560),
    ] {
        let weight_bits: Vec<u16> = (0..n)
            .flat_map(|column| {
                (0..k).map(move |inner| {
                    let bits = operand(column * 577 + inner / 2);
                    if inner % 2 == 0 { bits } else { bits ^ 0x8000 }
                })
            })
            .collect();
        let weight_units: Vec<i32> = weight_bits.iter().copied().map(units).collect();
        let spec =
            WeightTensorSpec::new("mixed", vec![n as u64, k as u64], DataType::BF16).unwrap();
        let mut weights =
            CudaBf16Weights::allocate(vec![spec], Arc::clone(&stream), &pool, 1 << 20).unwrap();
        let encoded: Vec<u8> = weight_bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        weights.upload(0, &mut Cursor::new(encoded)).unwrap();
        let mut gemm = CudaBf16Gemm::new(Arc::clone(&stream), &pool, 8 << 20).unwrap();
        for m in [1, 8] {
            let input_bits: Vec<u16> = (0..m)
                .flat_map(|row| {
                    (0..k).map(move |inner| {
                        let index = row * 997 + if row % 2 == 0 { inner / 2 } else { inner };
                        let bits = operand(index);
                        // Even rows cancel most product pairs, but retain a small
                        // residual; odd rows vary every input element independently.
                        if row % 2 == 0 && inner == k - 1 {
                            bits ^ 1
                        } else {
                            bits
                        }
                    })
                })
                .collect();
            let input_units: Vec<i32> = input_bits.iter().copied().map(units).collect();
            let mut input =
                CudaBf16Matrix::allocate(m, k, Arc::clone(&stream), &pool, m * k * 2).unwrap();
            let mut output =
                CudaBf16Matrix::allocate(m, n, Arc::clone(&stream), &pool, m * n * 2).unwrap();
            input
                .upload(
                    &input_bits
                        .iter()
                        .flat_map(|b| b.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            gemm.linear_into(&mut weights, 0, &mut input, &mut output)
                .unwrap();
            let actual = output.read_chunk(0, m * n * 2).unwrap();
            assert_forward_error(&actual, &input_units, &weight_units, (m, n, k));
            input.close().unwrap();
            output.close().unwrap();
        }
        gemm.close().unwrap();
        weights.close().unwrap();
        assert_eq!(pool.granted(), 0);
    }
}
