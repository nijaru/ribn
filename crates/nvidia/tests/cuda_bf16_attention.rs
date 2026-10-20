//! Causal GQA attention component acceptance, not a full-model budget.
#![cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};
use engine_nvidia::{
    AttentionGeometry, AttentionSpan, CudaBf16Attention, CudaBf16Error, CudaBf16Matrix,
};
use ribn_foundation::BytePool;
use std::sync::Arc;

fn value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
fn round(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
fn encode(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn read(matrix: &mut CudaBf16Matrix) -> Vec<u16> {
    let len = matrix.shape().0 * matrix.shape().1 * 2;
    let mut bytes = Vec::new();
    for offset in (0..len).step_by(5) {
        bytes.extend(
            matrix
                .read_chunk(offset as u64, 5.min(len - offset))
                .unwrap(),
        );
    }
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}
fn upload(
    values: &[u16],
    rows: usize,
    cols: usize,
    stream: &Arc<CudaStream>,
    pool: &Arc<BytePool>,
) -> CudaBf16Matrix {
    assert_eq!(values.len(), rows * cols);
    let mut matrix = CudaBf16Matrix::allocate(rows, cols, Arc::clone(stream), pool, 5).unwrap();
    matrix.upload(&encode(values)).unwrap();
    matrix
}

// Deterministic finite BF16 operands with mixed exponents and signs.
fn operands(seed: u64, count: usize) -> Vec<u16> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let sign = if state >> 63 == 0 { 1.0 } else { -1.0 };
            let exponent = i32::try_from((state >> 52) % 7).unwrap() - 3;
            let mantissa = f32::from(u16::try_from((state >> 16) & 0x3ff).unwrap()) / 1024.0;
            round(sign * mantissa * 2_f32.powi(exponent))
        })
        .collect()
}

fn reference(
    query: &[u16],
    keys: &[u16],
    values: &[u16],
    geometry: AttentionGeometry,
    query_start: usize,
    key_start: usize,
) -> Vec<u16> {
    let AttentionGeometry {
        heads,
        kv_heads,
        width,
        window,
    } = geometry;
    let sequence = query.len() / (heads * width);
    let groups = heads / kv_heads;
    let mut output = vec![0_u16; query.len()];
    for position in 0..sequence {
        let absolute = query_start + position;
        let last = absolute - key_start;
        let first = match window {
            None => 0,
            Some(window) => absolute
                .saturating_sub(window - 1)
                .saturating_sub(key_start),
        };
        for head in 0..heads {
            let kv_head = head / groups;
            let query_base = position * heads * width + head * width;
            let mut scores = Vec::with_capacity(last - first + 1);
            for key in first..=last {
                let key_base = key * kv_heads * width + kv_head * width;
                let mut sum = 0.0_f32;
                for column in 0..width {
                    sum += value(query[query_base + column]) * value(keys[key_base + column]);
                }
                scores.push(value(round(sum)));
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exponentials: Vec<f32> =
                scores.iter().map(|score| (score - maximum).exp()).collect();
            let total: f32 = exponentials.iter().sum();
            let mut accumulator = vec![0.0_f32; width];
            for (offset, key) in (first..=last).enumerate() {
                let weight = round(exponentials[offset] / total);
                let value_base = key * kv_heads * width + kv_head * width;
                for (slot, &element) in accumulator
                    .iter_mut()
                    .zip(&values[value_base..value_base + width])
                {
                    *slot += value(weight) * value(element);
                }
            }
            let output_base = position * heads * width + head * width;
            for (slot, &accumulated) in output[output_base..output_base + width]
                .iter_mut()
                .zip(&accumulator)
            {
                *slot = round(accumulated);
            }
        }
    }
    output
}

fn compare(actual: &[u16], expected: &[u16], label: &str) -> f32 {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let mut worst = 0.0_f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let (a, b) = (value(actual), value(expected));
        assert!(a.is_finite() && b.is_finite(), "{label} {index} non-finite");
        if b == 0.0 {
            assert_eq!(actual, expected, "{label} {index} signed zero");
            continue;
        }
        let expected_ulp = (value(expected.wrapping_add(1)) - b).abs();
        let error = (a - b).abs();
        worst = worst.max(error / expected_ulp);
        // PREDECLARED: one adjacent BF16 encoding. Scores and weights round to
        // BF16; only FP32 dot/reduction order and expf differ, both far below
        // BF16 spacing. A failure here is investigated, not absorbed by a wider
        // tolerance.
        assert_eq!(actual & 0x8000, expected & 0x8000, "{label} {index} sign");
        assert!(
            actual.abs_diff(expected) <= 1,
            "{label} {index}: device {a:e} host {b:e}"
        );
    }
    worst
}

#[test]
#[ignore = "requires CUDA; serialize native BF16 attention qualification"]
fn attention_matches_the_host_contract_on_gqa_windows_and_history() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(16 * 1024 * 1024).shared();
    let mut attention = CudaBf16Attention::new(Arc::clone(&stream), &pool, 8 * 16 * 16).unwrap();
    assert_eq!(attention.scratch_entries(), 2048);
    let cases = [
        (
            AttentionGeometry {
                heads: 4,
                kv_heads: 2,
                width: 8,
                window: None,
            },
            5,
            5,
        ),
        (
            AttentionGeometry {
                heads: 4,
                kv_heads: 2,
                width: 8,
                window: None,
            },
            1,
            5,
        ),
        (
            AttentionGeometry {
                heads: 4,
                kv_heads: 2,
                width: 8,
                window: Some(2),
            },
            4,
            4,
        ),
        (
            AttentionGeometry {
                heads: 4,
                kv_heads: 2,
                width: 8,
                window: Some(3),
            },
            2,
            6,
        ),
        (
            AttentionGeometry {
                heads: 8,
                kv_heads: 2,
                width: 256,
                window: None,
            },
            3,
            3,
        ),
        (
            AttentionGeometry {
                heads: 8,
                kv_heads: 2,
                width: 512,
                window: Some(512),
            },
            2,
            2,
        ),
    ];
    let mut worst = 0.0_f32;
    for (index, (geometry, sequence, key_sequence)) in cases.into_iter().enumerate() {
        let query_start = key_sequence - sequence;
        let query = operands(
            11 + index as u64,
            sequence * geometry.heads * geometry.width,
        );
        let keys = operands(
            29 + index as u64,
            key_sequence * geometry.kv_heads * geometry.width,
        );
        let values = operands(
            47 + index as u64,
            key_sequence * geometry.kv_heads * geometry.width,
        );
        let mut q = upload(
            &query,
            sequence,
            geometry.heads * geometry.width,
            &stream,
            &pool,
        );
        let mut k = upload(
            &keys,
            key_sequence,
            geometry.kv_heads * geometry.width,
            &stream,
            &pool,
        );
        let mut v = upload(
            &values,
            key_sequence,
            geometry.kv_heads * geometry.width,
            &stream,
            &pool,
        );
        let mut out = CudaBf16Matrix::allocate(
            sequence,
            geometry.heads * geometry.width,
            Arc::clone(&stream),
            &pool,
            5,
        )
        .unwrap();
        attention
            .attend_into(
                &q,
                &k,
                &v,
                geometry,
                AttentionSpan {
                    query_start,
                    key_start: 0,
                },
                &mut out,
            )
            .unwrap();
        let expected = reference(&query, &keys, &values, geometry, query_start, 0);
        worst = worst.max(compare(
            &read(&mut out),
            &expected,
            &format!("case {index}"),
        ));
        q.close().unwrap();
        k.close().unwrap();
        v.close().unwrap();
        out.close().unwrap();
    }
    println!("attention worst ulp usage {worst}");
    attention.close().unwrap();
    attention.close().unwrap();
    assert_eq!(pool.granted(), 0);
}

#[test]
#[ignore = "requires CUDA; serialize native BF16 attention qualification"]
fn attention_rejects_invalid_geometry_and_spans_before_enqueue() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(16 * 1024 * 1024).shared();
    let mut attention = CudaBf16Attention::new(Arc::clone(&stream), &pool, 64).unwrap();
    let geometry = AttentionGeometry {
        heads: 4,
        kv_heads: 2,
        width: 8,
        window: None,
    };
    let query = operands(3, 2 * 4 * 8);
    let keys = operands(5, 2 * 2 * 8);
    let mut q = upload(&query, 2, 32, &stream, &pool);
    let mut k = upload(&keys, 2, 16, &stream, &pool);
    let mut v = upload(&keys, 2, 16, &stream, &pool);
    let mut out = CudaBf16Matrix::allocate(2, 32, Arc::clone(&stream), &pool, 5).unwrap();

    // Scratch is 64 and needs 2*4*2 = 16: this passes, establishing readiness.
    attention
        .attend_into(
            &q,
            &k,
            &v,
            geometry,
            AttentionSpan {
                query_start: 0,
                key_start: 0,
            },
            &mut out,
        )
        .unwrap();
    let before = read(&mut out);
    for (label, changed, query_start, key_start) in [
        (
            "window",
            AttentionGeometry {
                window: Some(0),
                ..geometry
            },
            0,
            0,
        ),
        (
            "heads",
            AttentionGeometry {
                heads: 3,
                ..geometry
            },
            0,
            0,
        ),
        ("span", geometry, 1, 0),
        ("precede", geometry, 0, 1),
    ] {
        let error = attention
            .attend_into(
                &q,
                &k,
                &v,
                changed,
                AttentionSpan {
                    query_start,
                    key_start,
                },
                &mut out,
            )
            .unwrap_err();
        assert!(
            matches!(error.cause(), CudaBf16Error::Invalid(_)),
            "{label}"
        );
        assert!(!error.retirement_pending(), "{label}");
        assert_eq!(read(&mut out), before, "{label} preserves output");
    }

    let foreign = context.new_stream().unwrap();
    let mut foreign_q = upload(&query, 2, 32, &foreign, &pool);
    let error = attention
        .attend_into(
            &foreign_q,
            &k,
            &v,
            geometry,
            AttentionSpan {
                query_start: 0,
                key_start: 0,
            },
            &mut out,
        )
        .unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));

    let mut small = CudaBf16Attention::new(Arc::clone(&stream), &pool, 1).unwrap();
    let error = small
        .attend_into(
            &q,
            &k,
            &v,
            geometry,
            AttentionSpan {
                query_start: 0,
                key_start: 0,
            },
            &mut out,
        )
        .unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
    small.close().unwrap();

    q.close().unwrap();
    k.close().unwrap();
    v.close().unwrap();
    out.close().unwrap();
    foreign_q.close().unwrap();
    attention.close().unwrap();
    assert_eq!(pool.granted(), 0);
}
