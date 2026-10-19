//! Apple GPU compute checks against an independent host reference.
//!
//! These run on the local Metal device. Without one they skip, so the workspace
//! stays usable on hosts that have no Apple GPU.
#![cfg(target_os = "macos")]

use engine_metal::MetalBf16Executor;

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return 0x7fc0;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

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

/// Deterministic non-dyadic operands with mixed magnitudes and signs.
fn operands(count: usize, seed: u64) -> Vec<u16> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
            #[expect(
                clippy::cast_precision_loss,
                reason = "the generator masks to twelve bits before converting"
            )]
            let magnitude = ((state >> 33) % 4096) as f32 / 512.0 + 0.25;
            bf16(sign * magnitude)
        })
        .collect()
}

fn reference_linear(
    input: &[u16],
    weight: &[u16],
    rows: usize,
    columns: usize,
    inner: usize,
) -> Vec<u16> {
    let mut output = vec![0_u16; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let mut sum = 0.0_f32;
            for i in 0..inner {
                sum += value(input[row * inner + i]) * value(weight[column * inner + i]);
            }
            output[row * columns + column] = bf16(sum);
        }
    }
    output
}

fn reference_rms(
    input: &[u16],
    rows: usize,
    width: usize,
    epsilon: f32,
    scale: Option<&[u16]>,
) -> Vec<u16> {
    let mut output = vec![0_u16; input.len()];
    for row in 0..rows {
        let mut energy = 0.0_f32;
        for j in 0..width {
            let value = value(input[row * width + j]);
            energy += value * value;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "RMS widths are small model dimensions"
        )]
        let inverse = (energy / width as f32 + epsilon).powf(-0.5);
        for j in 0..width {
            let mut result = value(input[row * width + j]) * inverse;
            if let Some(scale) = scale {
                result *= value(scale[j]);
            }
            output[row * width + j] = bf16(result);
        }
    }
    output
}

fn assert_close(actual: &[u16], expected: &[u16], label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (value(actual) - value(expected)).abs();
        assert!(
            error <= ulp(expected),
            "{label}[{index}]: {actual:#06x} vs {expected:#06x} error {error:e}"
        );
    }
}

#[test]
fn upload_round_trips_native_encodings() {
    let Ok(executor) = MetalBf16Executor::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let bits = [0x0000_u16, 0x8000, 0x3f80, 0xbf80, 0x7fc0, 0x1234];
    let buffer = executor.upload(&bits).unwrap();
    assert_eq!(buffer.elements(), bits.len());
    assert_eq!(buffer.bits(), bits);
}

#[test]
fn linear_matches_a_host_reference_and_exact_dyadics() {
    let Ok(executor) = MetalBf16Executor::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let (rows, columns, inner) = (3, 7, 16);
    let input = operands(rows * inner, 11);
    let weight = operands(columns * inner, 29);
    let actual = executor
        .linear(
            &executor.upload(&input).unwrap(),
            &executor.upload(&weight).unwrap(),
            rows,
            columns,
            inner,
        )
        .unwrap();
    assert_close(
        &actual.bits(),
        &reference_linear(&input, &weight, rows, columns, inner),
        "linear",
    );

    // Dyadic operands whose products and partial sums are exactly representable.
    let input: Vec<u16> = [1.0_f32, -2.0, 0.5, 4.0, -1.0, 1.0, -0.5, 2.0]
        .into_iter()
        .map(bf16)
        .collect();
    let weight: Vec<u16> = [1.0_f32, 1.0, 2.0, -1.0].into_iter().map(bf16).collect();
    let exact = executor
        .linear(
            &executor.upload(&input).unwrap(),
            &executor.upload(&weight).unwrap(),
            2,
            1,
            4,
        )
        .unwrap();
    assert_eq!(exact.bits(), reference_linear(&input, &weight, 2, 1, 4));
}

#[test]
fn rms_matches_a_host_reference_with_and_without_scale() {
    let Ok(executor) = MetalBf16Executor::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    for width in [16_usize, 256, 2560] {
        let rows = 2;
        let input = operands(rows * width, 7);
        let scale = operands(width, 13);
        let buffer = executor.upload(&input).unwrap();
        let scale_buffer = executor.upload(&scale).unwrap();
        let scaled = executor
            .rms_norm(&buffer, rows, width, 1e-6, Some(&scale_buffer))
            .unwrap();
        assert_close(
            &scaled.bits(),
            &reference_rms(&input, rows, width, 1e-6, Some(&scale)),
            "scaled RMS",
        );
        let plain = executor.rms_norm(&buffer, rows, width, 1e-6, None).unwrap();
        assert_close(
            &plain.bits(),
            &reference_rms(&input, rows, width, 1e-6, None),
            "plain RMS",
        );
    }
}

#[test]
fn invalid_shapes_and_epsilon_reject() {
    let Ok(executor) = MetalBf16Executor::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let input = executor.upload(&operands(8, 1)).unwrap();
    let weight = executor.upload(&operands(8, 2)).unwrap();
    assert!(executor.linear(&input, &weight, 0, 2, 4).is_err());
    assert!(executor.linear(&input, &weight, 2, 3, 4).is_err());
    assert!(executor.rms_norm(&input, 2, 3, 1e-6, None).is_err());
    assert!(executor.rms_norm(&input, 2, 4, 0.0, None).is_err());
    assert!(
        executor
            .rms_norm(&input, 2, 4, 1e-6, Some(&weight))
            .is_err()
    );
}
