//! Exact preservation of the standalone attention preparation contract.
#![cfg(target_os = "macos")]

mod common;
use common::executor;
use engine_metal::{
    MetalBf16Buffer, MetalBf16Error, MetalBf16Executor, MetalLinearShape,
    MetalNormalizedProjection, MetalProjectionNorm, MetalRotaryFactors,
};

fn operand(executor: &MetalBf16Executor, elements: usize, seed: usize) -> MetalBf16Buffer {
    // Prime-period finite values distinguish rows, heads, weights and factor halves.
    let bits: Vec<_> = (0..elements)
        .map(|i| {
            let magnitude = 0x3e80 + u16::try_from((i * 17 + seed) % 127).unwrap();
            magnitude
                | if (i + seed).is_multiple_of(3) {
                    0x8000
                } else {
                    0
                }
        })
        .collect();
    executor.upload(&bits).unwrap()
}

fn normalized_projection<'a>(
    weight: &'a MetalBf16Buffer,
    width: usize,
    scale: Option<&'a MetalBf16Buffer>,
) -> MetalNormalizedProjection<'a> {
    MetalNormalizedProjection {
        weight,
        norm: MetalProjectionNorm {
            width,
            epsilon: 1e-6,
            scale,
            rotary: None,
        },
    }
}

fn rotated_projection<'a>(
    weight: &'a MetalBf16Buffer,
    heads: usize,
    width: usize,
    scale: Option<&'a MetalBf16Buffer>,
    cosine: &'a MetalBf16Buffer,
    sine: &'a MetalBf16Buffer,
) -> MetalNormalizedProjection<'a> {
    MetalNormalizedProjection {
        weight,
        norm: MetalProjectionNorm {
            width,
            epsilon: 1e-6,
            scale,
            rotary: Some(MetalRotaryFactors {
                heads,
                cosine,
                sine,
            }),
        },
    }
}

fn standalone(
    executor: &MetalBf16Executor,
    input: &MetalBf16Buffer,
    rows: usize,
    inner: usize,
    projection: MetalNormalizedProjection<'_>,
) -> MetalBf16Buffer {
    let columns = projection.weight.elements() / inner;
    let projected = executor
        .linear(
            input,
            projection.weight,
            MetalLinearShape {
                rows,
                columns,
                inner,
            },
        )
        .unwrap();
    let norm = projection.norm;
    let normalized = executor
        .rms_norm(
            &projected,
            projected.elements() / norm.width,
            norm.width,
            norm.epsilon,
            norm.scale,
        )
        .unwrap();
    if let Some(factors) = norm.rotary {
        executor
            .rotary_apply(
                &normalized,
                factors.cosine,
                factors.sine,
                rows,
                factors.heads,
                norm.width,
            )
            .unwrap()
    } else {
        normalized
    }
}

#[test]
fn attention_projections_preserve_standalone_bits_with_and_without_kv() {
    let Some(executor) = executor() else { return };
    for (rows, heads, kv_heads, width, inner) in [
        (1, 8, 2, 256, 256),
        (16, 8, 2, 256, 256),
        (64, 8, 2, 512, 256),
        (65, 3, 1, 6, 7),
    ] {
        let input = operand(&executor, rows * inner, 7);
        let input_scale = operand(&executor, inner, 11);
        let q = operand(&executor, heads * width * inner, 13);
        let k = operand(&executor, kv_heads * width * inner, 23);
        let v = operand(&executor, kv_heads * width * inner, 29);
        let q_scale = operand(&executor, width, 31);
        let k_scale = operand(&executor, width, 37);
        let cosine = operand(&executor, rows * width, 41);
        let sine = operand(&executor, rows * width, 53);
        let query = MetalNormalizedProjection {
            weight: &q,
            norm: MetalProjectionNorm {
                width,
                epsilon: 1e-6,
                scale: Some(&q_scale),
                rotary: Some(MetalRotaryFactors {
                    heads,
                    cosine: &cosine,
                    sine: &sine,
                }),
            },
        };
        let key = MetalNormalizedProjection {
            weight: &k,
            norm: MetalProjectionNorm {
                width,
                epsilon: 1e-6,
                scale: Some(&k_scale),
                rotary: Some(MetalRotaryFactors {
                    heads: kv_heads,
                    cosine: &cosine,
                    sine: &sine,
                }),
            },
        };
        let value = normalized_projection(&v, width, None);
        let shape = MetalLinearShape {
            rows,
            columns: heads * width,
            inner,
        };
        for scale in [None, Some(&input_scale)] {
            let normalized = executor.rms_norm(&input, rows, inner, 1e-6, scale).unwrap();
            let expected_q = standalone(&executor, &normalized, rows, inner, query);
            let expected_k = standalone(&executor, &normalized, rows, inner, key);
            let expected_v = standalone(&executor, &normalized, rows, inner, value);
            for kv in [None, Some((key, value))] {
                let actual = executor
                    .rms_attention_projections(&input, shape, 1e-6, scale, query, kv)
                    .unwrap();
                assert_eq!(actual.query.bits(), expected_q.bits(), "query rows={rows}");
                assert!(
                    actual
                        .query
                        .as_slice()
                        .iter()
                        .all(|&b| f32::from_bits(u32::from(b) << 16).is_finite())
                );
                match actual.key_value {
                    Some((k, v)) => {
                        assert!(kv.is_some());
                        assert!(
                            k.as_slice()
                                .iter()
                                .chain(v.as_slice())
                                .all(|&b| { f32::from_bits(u32::from(b) << 16).is_finite() })
                        );
                        assert_eq!(k.bits(), expected_k.bits(), "key rows={rows}");
                        assert_eq!(v.bits(), expected_v.bits(), "value rows={rows}");
                    }
                    None => assert!(kv.is_none()),
                }
            }
        }
    }
}

#[test]
fn attention_projections_reject_late_invalid_branches_and_recover() {
    let Some(executor) = executor() else { return };
    let input = operand(&executor, 8, 7);
    let q = operand(&executor, 16, 13);
    let kv = operand(&executor, 8, 23);
    let short = operand(&executor, 7, 29);
    let factors = operand(&executor, 4, 31);
    let short_factors = operand(&executor, 2, 37);
    let query = rotated_projection(&q, 2, 2, None, &factors, &factors);
    let key = rotated_projection(&kv, 1, 2, None, &factors, &factors);
    let value = normalized_projection(&kv, 2, None);
    let shape = MetalLinearShape {
        rows: 2,
        columns: 4,
        inner: 4,
    };
    let before = input.bits();
    for (key, value) in [
        (
            key,
            MetalNormalizedProjection {
                weight: &short,
                ..value
            },
        ),
        (
            key,
            MetalNormalizedProjection {
                weight: &q,
                ..value
            },
        ),
        (
            rotated_projection(&kv, 1, 2, Some(&factors), &factors, &factors),
            value,
        ),
        (
            rotated_projection(&kv, 1, 2, None, &short_factors, &factors),
            value,
        ),
        (
            rotated_projection(&kv, 2, 2, None, &factors, &factors),
            value,
        ),
    ] {
        assert!(matches!(
            executor.rms_attention_projections(
                &input,
                shape,
                1e-6,
                None,
                query,
                Some((key, value))
            ),
            Err(MetalBf16Error::Shape(_))
        ));
    }
    for epsilon in [0.0, -1.0, f32::NAN] {
        assert!(
            executor
                .rms_attention_projections(&input, shape, epsilon, None, query, None)
                .is_err()
        );
    }
    assert!(
        executor
            .rms_attention_projections(&input, shape, 1e-6, Some(&short_factors), query, None)
            .is_err()
    );
    assert!(
        executor
            .rms_attention_projections(
                &input,
                MetalLinearShape { inner: 5, ..shape },
                1e-6,
                None,
                query,
                None
            )
            .is_err()
    );
    assert_eq!(input.bits(), before);
    let normalized = executor.rms_norm(&input, 2, 4, 1e-6, None).unwrap();
    let actual = executor
        .rms_attention_projections(&input, shape, 1e-6, None, query, Some((key, value)))
        .unwrap();
    assert_eq!(
        actual.query.bits(),
        standalone(&executor, &normalized, 2, 4, query).bits()
    );
    let (actual_k, actual_v) = actual.key_value.unwrap();
    assert_eq!(
        actual_k.bits(),
        standalone(&executor, &normalized, 2, 4, key).bits()
    );
    assert_eq!(
        actual_v.bits(),
        standalone(&executor, &normalized, 2, 4, value).bits()
    );
}
