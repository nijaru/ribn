//! Apple GPU compute checks against an independent host reference.
//!
//! These run on the local Metal device. Without one they skip, so the workspace
//! stays usable on hosts that have no Apple GPU.
#![cfg(target_os = "macos")]

use engine_metal::{MetalAttentionConfig, MetalBf16Error, MetalBf16Executor, MetalLinearShape};

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
    let mut source = bits.to_vec();
    let buffer = executor.upload(&source).unwrap();
    // Shared storage owns a copy, not a view into the loading allocation.
    source.fill(0xffff);
    drop(source);
    assert_eq!(buffer.elements(), bits.len());
    assert_eq!(buffer.as_slice(), bits);
    assert_eq!(buffer.bits(), bits);
}

#[test]
fn streamed_encodings_cross_scratch_boundaries_without_aliasing_source() {
    use std::io::{self, Read};

    struct ShortReads<'a> {
        bytes: &'a [u8],
        largest_request: usize,
    }
    impl Read for ShortReads<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.largest_request = self.largest_request.max(buffer.len());
            let count = buffer.len().min(self.bytes.len()).min(7);
            buffer[..count].copy_from_slice(&self.bytes[..count]);
            self.bytes = &self.bytes[count..];
            Ok(count)
        }
    }

    let Some(executor) = executor() else { return };
    let bits: Vec<u16> = (0..=u16::MAX).take(32_771).collect();
    let mut bytes: Vec<u8> = bits.iter().flat_map(|bits| bits.to_le_bytes()).collect();
    let mut reader = ShortReads {
        bytes: &bytes,
        largest_request: 0,
    };
    let buffer = executor.read_bf16(bits.len(), &mut reader).unwrap();
    assert!(reader.bytes.is_empty());
    assert!(reader.largest_request <= 64 * 1024);
    bytes.fill(0xff);
    drop(bytes);
    assert_eq!(buffer.as_slice(), bits);
}

#[test]
fn streamed_initialization_rejects_incomplete_sources_and_recovers() {
    use std::io::{self, Cursor, Read};

    struct Broken {
        remaining: usize,
        panic: bool,
    }
    impl Read for Broken {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                assert!(!self.panic, "injected source panic");
                return Err(io::Error::other("injected source error"));
            }
            let count = self.remaining.min(bytes.len());
            bytes[..count].fill(0x3f);
            self.remaining -= count;
            Ok(count)
        }
    }

    let Some(executor) = executor() else { return };
    for data in [&[0x80_u8][..], &[0x80, 0x3f, 0xff][..]] {
        assert!(executor.read_bf16(1, &mut Cursor::new(data)).is_err());
    }
    assert!(matches!(
        executor.read_bf16(
            32_769,
            &mut Broken {
                remaining: 65_536,
                panic: false
            }
        ),
        Err(MetalBf16Error::Read(_))
    ));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor.read_bf16(
                32_769,
                &mut Broken {
                    remaining: 65_536,
                    panic: true,
                },
            )
        }))
        .is_err()
    );
    for elements in [0, usize::MAX, usize::MAX / 4] {
        let mut reader = Cursor::new([0x80, 0x3f]);
        assert!(matches!(
            executor.read_bf16(elements, &mut reader),
            Err(MetalBf16Error::Invalid(_))
        ));
        assert_eq!(reader.position(), 0);
    }
    let buffer = executor
        .read_bf16(1, &mut Cursor::new([0x80, 0x3f]))
        .unwrap();
    assert_eq!(buffer.as_slice(), [0x3f80]);
}

#[test]
fn linear_matches_a_host_reference_and_exact_dyadics() {
    let Some(executor) = executor() else {
        return;
    };
    // Paired and unpaired rows, tiny grids, column tails and non-vector widths.
    // Every output preserves its own source-order accumulation, not a reduction
    // across rows; adding the second row must not change the first row's bits.
    for (rows, columns, inner) in [
        (3, 7, 16),
        (2, 1, 4),
        (8, 7, 16),
        (9, 7, 16),
        (18, 257, 256),
        (64, 8, 1024),
        (66, 8, 1024),
        (8, 3, 7),
    ] {
        let input = operands(rows * inner, 11);
        let weight = operands(columns * inner, 29);
        let actual = executor
            .linear(
                &executor.upload(&input).unwrap(),
                &executor.upload(&weight).unwrap(),
                MetalLinearShape {
                    rows,
                    columns,
                    inner,
                },
            )
            .unwrap();
        assert_eq!(
            actual.bits(),
            reference_linear(&input, &weight, rows, columns, inner),
            "linear {rows}x{columns}x{inner}",
        );
    }

    // Dyadic operands whose products and partial sums are exactly representable.
    let input: Vec<u16> = [1.0_f32, -2.0, 0.5, 4.0, -1.0, 1.0, -0.5, 2.0]
        .into_iter()
        .cycle()
        .take(8 * 4)
        .map(bf16)
        .collect();
    let weight: Vec<u16> = [1.0_f32, 1.0, 2.0, -1.0].into_iter().map(bf16).collect();
    for rows in [2, 8] {
        let input = &input[..rows * 4];
        let exact = executor
            .linear(
                &executor.upload(input).unwrap(),
                &executor.upload(&weight).unwrap(),
                MetalLinearShape {
                    rows,
                    columns: 1,
                    inner: 4,
                },
            )
            .unwrap();
        assert_eq!(exact.bits(), reference_linear(input, &weight, rows, 1, 4));
    }
}

#[test]
fn linear_rms_preserves_standalone_bits_and_grouping() {
    let Some(executor) = executor() else {
        return;
    };
    // Paired/unpaired linear selection, column/inner tails, and multiple RMS
    // groups per projected row. Existing standalone tests supply independent
    // arithmetic references; this gate protects the in-command dependency and
    // the BF16 materialization boundary between the two kernels.
    for (rows, columns, inner, width) in [
        (1, 512, 256, 256),
        (3, 7, 16, 7),
        (8, 768, 256, 256),
        (9, 514, 7, 257),
        (64, 512, 256, 512),
        (66, 256, 256, 256),
    ] {
        let input = executor.upload(&operands(rows * inner, 11)).unwrap();
        let weight = executor.upload(&operands(columns * inner, 29)).unwrap();
        let scale = executor.upload(&operands(width, 13)).unwrap();
        let shape = MetalLinearShape {
            rows,
            columns,
            inner,
        };
        let projected = executor.linear(&input, &weight, shape).unwrap();
        for scale in [None, Some(&scale)] {
            let expected = executor
                .rms_norm(&projected, rows * columns / width, width, 1e-6, scale)
                .unwrap();
            let actual = executor
                .linear_rms(&input, &weight, shape, width, 1e-6, scale)
                .unwrap();
            assert_eq!(actual.elements(), rows * columns);
            assert_eq!(
                actual.bits(),
                expected.bits(),
                "linear/RMS {shape:?}, width={width}"
            );
        }
    }
}

#[test]
fn linear_rms_rejects_invalid_whole_chains_and_recovers() {
    let Some(executor) = executor() else {
        return;
    };
    let input_bits = operands(8, 11);
    let weight_bits = operands(8, 29);
    let input = executor.upload(&input_bits).unwrap();
    let weight = executor.upload(&weight_bits).unwrap();
    let shape = MetalLinearShape {
        rows: 2,
        columns: 2,
        inner: 4,
    };
    for (width, epsilon, scale) in [
        (0, 1e-6, None),
        (3, 1e-6, None),
        (2, 0.0, None),
        (2, f32::NAN, None),
        (2, 1e-6, Some(&weight)),
    ] {
        assert!(matches!(
            executor.linear_rms(&input, &weight, shape, width, epsilon, scale),
            Err(MetalBf16Error::Shape(_))
        ));
    }
    let overflow = MetalLinearShape {
        rows: usize::MAX,
        columns: 2,
        inner: 4,
    };
    assert!(
        executor
            .linear_rms(&input, &weight, overflow, 2, 1e-6, None)
            .is_err()
    );
    assert_eq!(input.bits(), input_bits);
    assert_eq!(weight.bits(), weight_bits);
    let projected = executor.linear(&input, &weight, shape).unwrap();
    let expected = executor.rms_norm(&projected, 2, 2, 1e-6, None).unwrap();
    let recovered = executor
        .linear_rms(&input, &weight, shape, 2, 1e-6, None)
        .unwrap();
    assert_eq!(recovered.bits(), expected.bits());
}

#[test]
fn rms_linear_pair_preserves_both_standalone_projections() {
    let Some(executor) = executor() else {
        return;
    };
    for (rows, columns, inner) in [
        (1, 7, 16),
        (8, 257, 256),
        (9, 7, 17),
        (64, 8, 256),
        (66, 8, 256),
    ] {
        let input = executor.upload(&operands(rows * inner, 11)).unwrap();
        let left = executor.upload(&operands(columns * inner, 29)).unwrap();
        let right = executor.upload(&operands(columns * inner, 47)).unwrap();
        let scale = executor.upload(&operands(inner, 13)).unwrap();
        let shape = MetalLinearShape {
            rows,
            columns,
            inner,
        };
        for scale in [None, Some(&scale)] {
            let normalized = executor.rms_norm(&input, rows, inner, 1e-6, scale).unwrap();
            let expected_left = executor.linear(&normalized, &left, shape).unwrap();
            let expected_right = executor.linear(&normalized, &right, shape).unwrap();
            let (actual_left, actual_right) = executor
                .rms_linear_pair(&input, &left, &right, shape, 1e-6, scale)
                .unwrap();
            assert_eq!(actual_left.bits(), expected_left.bits(), "left {shape:?}");
            assert_eq!(
                actual_right.bits(),
                expected_right.bits(),
                "right {shape:?}"
            );
        }
    }
}

#[test]
fn rms_linear_pair_rejects_invalid_right_weight_and_recovers() {
    let Some(executor) = executor() else {
        return;
    };
    let input = executor.upload(&operands(8, 11)).unwrap();
    let weight = executor.upload(&operands(8, 29)).unwrap();
    let short = executor.upload(&operands(4, 47)).unwrap();
    let shape = MetalLinearShape {
        rows: 2,
        columns: 2,
        inner: 4,
    };
    assert!(matches!(
        executor.rms_linear_pair(&input, &weight, &short, shape, 1e-6, None),
        Err(MetalBf16Error::Shape(_))
    ));
    assert!(
        executor
            .rms_linear_pair(&input, &weight, &weight, shape, 0.0, None)
            .is_err()
    );
    assert!(
        executor
            .rms_linear_pair(&input, &weight, &weight, shape, 1e-6, Some(&weight))
            .is_err()
    );
    let normalized = executor.rms_norm(&input, 2, 4, 1e-6, None).unwrap();
    let expected = executor.linear(&normalized, &weight, shape).unwrap();
    let (left, right) = executor
        .rms_linear_pair(&input, &weight, &weight, shape, 1e-6, None)
        .unwrap();
    assert_eq!(left.bits(), expected.bits());
    assert_eq!(right.bits(), expected.bits());
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
    for shape in [
        MetalLinearShape {
            rows: 0,
            columns: 2,
            inner: 4,
        },
        MetalLinearShape {
            rows: 2,
            columns: 3,
            inner: 4,
        },
    ] {
        assert!(executor.linear(&input, &weight, shape).is_err());
    }
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
