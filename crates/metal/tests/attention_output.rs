//! Fixed attention/output-projection composition must preserve the standalone bits.
#![cfg(target_os = "macos")]

use engine_metal::{
    MetalAttentionConfig, MetalBf16Error, MetalLinearShape, MetalNormalizedProjection,
    MetalProjectionNorm, MetalRotaryFactors,
};
mod common;
use common::executor;

fn operands(count: usize, seed: usize) -> Vec<u16> {
    (0..count)
        .map(|i| {
            let value = f32::from(u16::try_from((i * 17 + seed) % 37).unwrap()) / 32.0 - 0.5;
            u16::try_from(value.to_bits() >> 16).unwrap()
        })
        .collect()
}

#[test]
fn attention_output_matches_separate_commands_exactly() {
    let Some(executor) = executor() else { return };
    for (rows, keys, heads, kv_heads, width, columns, start, window, scaled, rotate) in [
        (1, 5, 2, 1, 6, 7, 3, None, false, false),
        (7, 11, 4, 2, 6, 8, 19, Some(3), true, true),
        (8, 9, 2, 1, 4, 7, 0, None, true, false),
        (16, 33, 4, 1, 8, 12, 7, Some(8), true, false),
        (32, 64, 2, 1, 4, 11, 0, Some(5), true, false),
        (64, 576, 2, 1, 4, 8, 0, Some(512), false, false),
        (65, 65, 2, 1, 4, 7, 0, None, false, false),
    ] {
        let inner = heads * width;
        let q = executor.upload(&operands(rows * inner, 1)).unwrap();
        let k = executor
            .upload(&operands(keys * kv_heads * width, 2))
            .unwrap();
        let v = executor
            .upload(&operands(keys * kv_heads * width, 3))
            .unwrap();
        let weight = executor.upload(&operands(columns * inner, 4)).unwrap();
        let norm_width = if rotate { 4 } else { columns };
        let scale = executor.upload(&operands(norm_width, 5)).unwrap();
        let cosine = executor.upload(&operands(rows * norm_width, 6)).unwrap();
        let sine = executor.upload(&operands(rows * norm_width, 7)).unwrap();
        let norm = MetalProjectionNorm {
            width: norm_width,
            epsilon: 1e-6,
            scale: scaled.then_some(&scale),
            rotary: rotate.then_some(MetalRotaryFactors {
                heads: columns / norm_width,
                cosine: &cosine,
                sine: &sine,
            }),
        };
        let config = MetalAttentionConfig {
            heads,
            kv_heads,
            width,
            window,
        };
        let position = start + keys - rows;
        let attended = executor
            .attention(&q, &k, &v, config, position, start)
            .unwrap();
        // Exercise the actual shared-storage host round trip being removed.
        let uploaded = executor.upload(&attended.bits()).unwrap();
        let expected = executor
            .project_normalized(
                &uploaded,
                &weight,
                MetalLinearShape {
                    rows,
                    columns,
                    inner,
                },
                norm,
            )
            .unwrap()
            .bits();
        let actual = executor
            .attention_project_normalized(
                &q,
                &k,
                &v,
                config,
                position,
                start,
                MetalNormalizedProjection {
                    weight: &weight,
                    norm,
                },
            )
            .unwrap()
            .bits();
        assert_eq!(actual.len(), rows * columns);
        assert!(
            actual
                .iter()
                .all(|&bits| f32::from_bits(u32::from(bits) << 16).is_finite())
        );
        assert_eq!(
            actual, expected,
            "rows={rows}, keys={keys}, window={window:?}"
        );
    }
}

#[test]
fn attention_output_rejects_late_invalid_projection_and_recovers() {
    let Some(executor) = executor() else { return };
    let q = executor.upload(&operands(8, 1)).unwrap();
    let k = executor.upload(&operands(8, 2)).unwrap();
    let v = executor.upload(&operands(8, 3)).unwrap();
    let weight = executor.upload(&operands(24, 4)).unwrap();
    let short = executor.upload(&operands(2, 5)).unwrap();
    let config = MetalAttentionConfig {
        heads: 2,
        kv_heads: 1,
        width: 4,
        window: None,
    };
    let norm = MetalProjectionNorm {
        width: 3,
        epsilon: 1e-6,
        scale: None,
        rotary: None,
    };
    let call = |weight: &engine_metal::MetalBf16Buffer,
                norm: MetalProjectionNorm<'_>,
                values: &engine_metal::MetalBf16Buffer| {
        executor.attention_project_normalized(
            &q,
            &k,
            values,
            config,
            1,
            0,
            MetalNormalizedProjection { weight, norm },
        )
    };
    for bad_norm in [
        MetalProjectionNorm { width: 0, ..norm },
        MetalProjectionNorm {
            epsilon: f32::NAN,
            ..norm
        },
        MetalProjectionNorm {
            scale: Some(&short),
            ..norm
        },
        MetalProjectionNorm {
            rotary: Some(MetalRotaryFactors {
                heads: 1,
                cosine: &short,
                sine: &short,
            }),
            ..norm
        },
    ] {
        assert!(matches!(
            call(&weight, bad_norm, &v),
            Err(MetalBf16Error::Shape(_))
        ));
    }
    let bad_weight = executor.upload(&operands(25, 6)).unwrap();
    assert!(matches!(
        call(&bad_weight, norm, &v),
        Err(MetalBf16Error::Shape(_))
    ));
    for values in [6, 10] {
        let wrong = executor.upload(&operands(values, 7)).unwrap();
        assert!(matches!(
            call(&weight, norm, &wrong),
            Err(MetalBf16Error::Shape(_))
        ));
    }
    let attended = executor.attention(&q, &k, &v, config, 1, 0).unwrap();
    let expected = executor
        .project_normalized(
            &attended,
            &weight,
            MetalLinearShape {
                rows: 1,
                columns: 3,
                inner: 8,
            },
            norm,
        )
        .unwrap()
        .bits();
    assert_eq!(call(&weight, norm, &v).unwrap().bits(), expected);
}
