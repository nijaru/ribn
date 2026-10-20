//! Apple GPU compute checks against an independent host reference.
//!
//! These run on the local Metal device. Without one they skip, so the workspace
//! stays usable on hosts that have no Apple GPU.
#![cfg(target_os = "macos")]

use engine_metal::{MetalAttentionConfig, MetalBf16Error, MetalBf16Executor};

/// Missing hardware skips; any compilation or pipeline failure must fail the test.
fn executor() -> Option<MetalBf16Executor> {
    match MetalBf16Executor::new() {
        Ok(executor) => Some(executor),
        Err(MetalBf16Error::Device) => {
            eprintln!("no Metal device; skipping");
            None
        }
        Err(error) => panic!("Metal preparation failed: {error}"),
    }
}

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
    let Some(executor) = executor() else {
        return;
    };
    let bits = [0x0000_u16, 0x8000, 0x3f80, 0xbf80, 0x7fc0, 0x1234];
    let buffer = executor.upload(&bits).unwrap();
    assert_eq!(buffer.elements(), bits.len());
    assert_eq!(buffer.bits(), bits);
}

#[test]
fn linear_matches_a_host_reference_and_exact_dyadics() {
    let Some(executor) = executor() else {
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
    let Some(executor) = executor() else {
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
    let Some(executor) = executor() else {
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

struct AttentionReference {
    output: Vec<u16>,
    /// Per output element: the reference's `sum |weight × value|` magnitude.
    magnitude: Vec<f32>,
    /// Per query position and head: largest visible BF16 score encoding step.
    slack: Vec<f32>,
}

fn reference_attention(
    query: &[u16],
    keys: &[u16],
    values: &[u16],
    config: MetalAttentionConfig,
    query_start: usize,
    key_start: usize,
) -> AttentionReference {
    let MetalAttentionConfig {
        heads,
        kv_heads,
        width,
        window,
    } = config;
    let sequence = query.len() / (heads * width);
    let key_sequence = keys.len() / (kv_heads * width);
    let groups = heads / kv_heads;
    let mut output = vec![0_u16; query.len()];
    let mut magnitude = vec![0.0_f32; query.len()];
    let mut slack = vec![0.0_f32; sequence * heads];
    for position in 0..sequence {
        let absolute = query_start + position;
        let last = (absolute - key_start).min(key_sequence - 1);
        let first = match window {
            None => 0,
            Some(window) => optional_start(absolute, window)
                .saturating_sub(key_start)
                .min(last),
        };
        for head in 0..heads {
            let kv_head = head / groups;
            let query_base = position * heads * width + head * width;
            let mut scores = Vec::new();
            for key_position in first..=last {
                let key_base = key_position * kv_heads * width + kv_head * width;
                let mut sum = 0.0_f32;
                for column in 0..width {
                    sum += value(query[query_base + column]) * value(keys[key_base + column]);
                }
                let bits = bf16(sum);
                scores.push(value(bits));
                slack[position * heads + head] = slack[position * heads + head].max(ulp(bits));
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let denominator: f32 = weights.iter().sum();
            for column in 0..width {
                let mut accumulator = 0.0_f32;
                let mut total = 0.0_f32;
                for (offset, weight) in weights.iter().enumerate() {
                    let key_base = (first + offset) * kv_heads * width + kv_head * width;
                    let weight = value(bf16(weight / denominator));
                    let stored = value(values[key_base + column]);
                    accumulator += weight * stored;
                    total += (weight * stored).abs();
                }
                output[query_base + column] = bf16(accumulator);
                magnitude[query_base + column] = total;
            }
        }
    }
    AttentionReference {
        output,
        magnitude,
        slack,
    }
}

fn optional_start(absolute: usize, window: usize) -> usize {
    absolute.saturating_sub(window - 1)
}

#[test]
fn attention_matches_a_host_reference_for_causal_and_sliding_spans() {
    let Some(executor) = executor() else {
        return;
    };
    // Dyadic operands under causal and sliding masks: every dot, probability and
    // weighted sum is exactly representable, so the bits must agree.
    let query: Vec<u16> = [
        1.0_f32, 0.5, -1.0, 2.0, 0.5, -0.5, 1.0, -2.0, -1.0, 1.0, 2.0, 0.5,
    ]
    .into_iter()
    .map(bf16)
    .collect();
    let keys: Vec<u16> = [1.0_f32, -1.0, 2.0, 0.5, -0.5, 1.0]
        .into_iter()
        .map(bf16)
        .collect();
    let values: Vec<u16> = [1.0_f32, 2.0, -1.0, 0.5, 2.0, 1.0]
        .into_iter()
        .map(bf16)
        .collect();
    for window in [None, Some(2)] {
        let config = MetalAttentionConfig {
            heads: 2,
            kv_heads: 1,
            width: 2,
            window,
        };
        let reference = reference_attention(&query, &keys, &values, config, 0, 0);
        let actual = executor
            .attention(
                &executor.upload(&query).unwrap(),
                &executor.upload(&keys).unwrap(),
                &executor.upload(&values).unwrap(),
                config,
                0,
                0,
            )
            .unwrap();
        assert_eq!(actual.bits(), reference.output, "window {window:?}");
    }

    // General operands with a cached prefix and a sliding window.
    let (sequence, key_sequence, heads, kv_heads, width) = (4, 8, 2, 1, 8);
    let query = operands(sequence * heads * width, 41);
    let keys = operands(key_sequence * kv_heads * width, 43);
    let values = operands(key_sequence * kv_heads * width, 47);
    let config = MetalAttentionConfig {
        heads,
        kv_heads,
        width,
        window: Some(3),
    };
    let reference = reference_attention(&query, &keys, &values, config, 4, 0);
    let actual = executor
        .attention(
            &executor.upload(&query).unwrap(),
            &executor.upload(&keys).unwrap(),
            &executor.upload(&values).unwrap(),
            config,
            4,
            0,
        )
        .unwrap();
    let actual = actual.bits();
    for (index, &expected) in reference.output.iter().enumerate() {
        let position = index / (heads * width);
        let head = (index / width) % heads;
        let visible = (4 + position) - optional_start(4 + position, 3) + 1;
        #[expect(clippy::cast_precision_loss, reason = "visible key counts are small")]
        let gamma = visible as f32 * f32::EPSILON / (1.0 - visible as f32 * f32::EPSILON);
        let bound = (gamma + 2.0 * reference.slack[position * heads + head])
            * reference.magnitude[index]
            + ulp(expected);
        let error = (value(actual[index]) - value(expected)).abs();
        assert!(
            error <= bound,
            "attention[{index}] error {error:e} exceeds {bound:e}"
        );
    }
}

#[test]
fn attention_rejects_malformed_spans_and_geometry() {
    let Some(executor) = executor() else {
        return;
    };
    let query = executor.upload(&operands(4, 1)).unwrap();
    let keys = executor.upload(&operands(4, 2)).unwrap();
    let values = executor.upload(&operands(4, 3)).unwrap();
    let plain = MetalAttentionConfig {
        heads: 1,
        kv_heads: 1,
        width: 2,
        window: None,
    };
    assert!(
        executor
            .attention(&query, &keys, &values, plain, 1, 0)
            .is_err()
    );
    assert!(
        executor
            .attention(&query, &keys, &values, plain, 0, 1)
            .is_err()
    );
    let uneven = MetalAttentionConfig {
        heads: 3,
        kv_heads: 2,
        ..plain
    };
    assert!(
        executor
            .attention(&query, &keys, &values, uneven, 0, 0)
            .is_err()
    );
    let closed = MetalAttentionConfig {
        window: Some(0),
        ..plain
    };
    assert!(
        executor
            .attention(&query, &keys, &values, closed, 0, 0)
            .is_err()
    );
}

fn reference_rotary(
    input: &[u16],
    cosine: &[u16],
    sine: &[u16],
    rows: usize,
    heads: usize,
    width: usize,
) -> Vec<u16> {
    let half = width / 2;
    let mut output = vec![0_u16; input.len()];
    for row in 0..rows {
        for head in 0..heads {
            let base = (row * heads + head) * width;
            for j in 0..half {
                let factor = row * width + j;
                let product = |a: f32, b: f32| value(bf16(a * b));
                let first = product(value(input[base + j]), value(cosine[factor]))
                    + product(-value(input[base + j + half]), value(sine[factor]));
                let second = product(value(input[base + j + half]), value(cosine[factor + half]))
                    + product(value(input[base + j]), value(sine[factor + half]));
                output[base + j] = bf16(first);
                output[base + j + half] = bf16(second);
            }
        }
    }
    output
}

#[test]
fn rotary_matches_a_host_reference_with_both_factor_halves() {
    let Some(executor) = executor() else {
        return;
    };
    let (rows, heads, width) = (3, 2, 4);
    let input = operands(rows * heads * width, 61);
    let cosine = operands(rows * width, 67);
    let sine = operands(rows * width, 71);
    let actual = executor
        .rotary_apply(
            &executor.upload(&input).unwrap(),
            &executor.upload(&cosine).unwrap(),
            &executor.upload(&sine).unwrap(),
            rows,
            heads,
            width,
        )
        .unwrap();
    assert_eq!(
        actual.bits(),
        reference_rotary(&input, &cosine, &sine, rows, heads, width)
    );
    assert!(
        executor
            .rotary_apply(
                &executor.upload(&input).unwrap(),
                &executor.upload(&cosine).unwrap(),
                &executor.upload(&sine).unwrap(),
                rows,
                heads,
                3,
            )
            .is_err()
    );
}

fn reference_gelu(input: &[u16]) -> Vec<u16> {
    const ALPHA: f32 = 0.797_884_6;
    const CUBIC: f32 = 0.044_715;
    input
        .iter()
        .map(|&bits| {
            let x = value(bits);
            bf16(0.5 * x * (1.0 + (ALPHA * x * (1.0 + CUBIC * x * x)).tanh()))
        })
        .collect()
}

fn reference_softcap(input: &[u16], cap: f32) -> Vec<u16> {
    input
        .iter()
        .map(|&bits| {
            let divided = bf16(value(bits) / cap);
            let hyperbolic = bf16(value(divided).tanh());
            bf16(value(hyperbolic) * cap)
        })
        .collect()
}

#[test]
fn elementwise_operations_match_host_references() {
    let Some(executor) = executor() else {
        return;
    };
    let left = operands(64, 73);
    let right = operands(64, 79);
    let left_buffer = executor.upload(&left).unwrap();
    let right_buffer = executor.upload(&right).unwrap();

    let added = executor.add(&left_buffer, &right_buffer).unwrap();
    let expected: Vec<u16> = left
        .iter()
        .zip(&right)
        .map(|(&a, &b)| bf16(value(a) + value(b)))
        .collect();
    assert_eq!(added.bits(), expected);

    let multiplied = executor.multiply(&left_buffer, &right_buffer).unwrap();
    let expected: Vec<u16> = left
        .iter()
        .zip(&right)
        .map(|(&a, &b)| bf16(value(a) * value(b)))
        .collect();
    assert_eq!(multiplied.bits(), expected);

    let scaled = executor.scale(&left_buffer, 0.5).unwrap();
    let expected: Vec<u16> = left.iter().map(|&a| bf16(value(a) * 0.5)).collect();
    assert_eq!(scaled.bits(), expected);

    let gelu = executor.gelu(&left_buffer).unwrap();
    let expected = reference_gelu(&left);
    for (index, (&actual, &expected)) in gelu.bits().iter().zip(&expected).enumerate() {
        let bound = 16.0 * f32::EPSILON * value(left[index]).abs() + ulp(expected);
        let error = (value(actual) - value(expected)).abs();
        assert!(
            error <= bound,
            "gelu[{index}] error {error:e} exceeds {bound:e}"
        );
    }

    let capped = executor.softcap(&left_buffer, 30.0).unwrap();
    let expected = reference_softcap(&left, 30.0);
    for (index, (&actual, &expected)) in capped.bits().iter().zip(&expected).enumerate() {
        let bound = 8.0 * f32::EPSILON * 30.0 + 2.0 * ulp(expected);
        let error = (value(actual) - value(expected)).abs();
        assert!(
            error <= bound,
            "softcap[{index}] error {error:e} exceeds {bound:e}"
        );
    }
}

#[test]
fn elementwise_validation_rejects_bad_operands() {
    let Some(executor) = executor() else {
        return;
    };
    let left = executor.upload(&operands(8, 83)).unwrap();
    let shorter = executor.upload(&operands(4, 89)).unwrap();
    assert!(executor.add(&left, &shorter).is_err());
    assert!(executor.multiply(&left, &shorter).is_err());
    assert!(executor.scale(&left, f32::NAN).is_err());
    assert!(executor.softcap(&left, 0.0).is_err());
}
