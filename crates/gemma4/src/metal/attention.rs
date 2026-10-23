//! Attention preparation and cache use for the host-orchestrated Metal decoder.

use super::{Gemma4Config, Gemma4Error, Gemma4MetalWeights, MetalOps, OwnerCache, metal_error};
use crate::cpu::add;
use crate::{AttentionKind, LayerParameter};
use engine_metal::{
    MetalAttentionConfig, MetalAttentionProjections, MetalLinearShape, MetalNormalizedProjection,
    MetalProjectionNorm, MetalRotaryFactors,
};

// Each chain separately bounds its live native payload. The input fork counts
// normalized input, three Q buffers, three K buffers and two V buffers; output sizing
// is below. Inputs, immutable weights/factors, host results and driver overhead are separate.
const MAX_CHAIN_BYTES: usize = 4 * 1024 * 1024;
const MAX_CHAIN_TOKENS: usize = 64;
// Bound the history-dependent scratch to the independently exercised key span.
const MAX_OUTPUT_CHAIN_KEYS: usize = 576;

fn chain_bytes(rows: usize, inner: usize, query: usize, key_value: Option<usize>) -> Option<usize> {
    let kv = key_value.unwrap_or(0).checked_mul(5)?;
    inner
        .checked_add(query.checked_mul(3)?)?
        .checked_add(kv)?
        .checked_mul(rows)?
        .checked_mul(size_of::<u16>())
}

fn chain_eligible(rows: usize, inner: usize, query: usize, key_value: Option<usize>) -> bool {
    (1..=MAX_CHAIN_TOKENS).contains(&rows)
        && chain_bytes(rows, inner, query, key_value).is_some_and(|bytes| bytes <= MAX_CHAIN_BYTES)
}

// Gemma's output projection has columns == hidden, full-width RMS and no rotation.
// BF16 scores, two FP32 row summaries, attended values and two projected outputs.
fn output_chain_bytes(
    rows: usize,
    heads: usize,
    width: usize,
    keys: usize,
    hidden: usize,
) -> Option<usize> {
    let per_head = keys
        .checked_mul(2)?
        .checked_add(8)?
        .checked_add(width.checked_mul(2)?)?;
    heads
        .checked_mul(per_head)?
        .checked_add(hidden.checked_mul(4)?)?
        .checked_mul(rows)
}

fn output_chain_eligible(
    rows: usize,
    config: MetalAttentionConfig,
    keys: usize,
    hidden: usize,
) -> bool {
    (1..=MAX_CHAIN_TOKENS).contains(&rows)
        && (1..=MAX_OUTPUT_CHAIN_KEYS).contains(&keys)
        && output_chain_bytes(rows, config.heads, config.width, keys, hidden)
            .is_some_and(|bytes| bytes <= MAX_CHAIN_BYTES)
}

fn project_attention(
    ops: &MetalOps<'_>,
    queries: &[u16],
    cache: &OwnerCache,
    config: MetalAttentionConfig,
    position: usize,
    rows: usize,
    projection: MetalNormalizedProjection<'_>,
) -> Result<Vec<u16>, Gemma4Error> {
    let keys = cache.keys.len() / (config.kv_heads * config.width);
    if output_chain_eligible(rows, config, keys, projection.norm.width) {
        let query = ops
            .executor
            .upload(queries)
            .map_err(|error| metal_error(&error))?;
        let keys = ops
            .executor
            .upload(&cache.keys)
            .map_err(|error| metal_error(&error))?;
        let values = ops
            .executor
            .upload(&cache.values)
            .map_err(|error| metal_error(&error))?;
        let output = ops
            .executor
            .attention_project_normalized(&query, &keys, &values, config, position, 0, projection)
            .map_err(|error| metal_error(&error))?;
        Ok(output.bits())
    } else {
        let attended = ops.attention(queries, &cache.keys, &cache.values, config, position, 0)?;
        ops.project_normalized(&attended, rows, projection.weight, projection.norm)
    }
}

struct AttentionInputs {
    queries: Vec<u16>,
    key_value: Option<(Vec<u16>, Vec<u16>)>,
}

fn copy_native(native: MetalAttentionProjections) -> AttentionInputs {
    let queries = native.query.bits();
    drop(native.query);
    let key_value = native.key_value.map(|(key, value)| {
        let keys = key.bits();
        drop(key);
        (keys, value.bits())
    });
    AttentionInputs { queries, key_value }
}

#[expect(
    clippy::too_many_arguments,
    reason = "one layer's projection inputs, without cache mutation"
)]
fn prepare_projections(
    config: &Gemma4Config,
    metal: &Gemma4MetalWeights,
    ops: &MetalOps<'_>,
    geometry: &crate::LayerGeometry,
    layer: usize,
    sequence: usize,
    epsilon: f32,
    positions: &[u32],
    hidden: &[u16],
) -> Result<AttentionInputs, Gemma4Error> {
    let hidden_size = config.hidden_size();
    let heads = config.num_attention_heads();
    let kv_heads = config.num_key_value_heads();
    let width = geometry.head_dim;
    let (cosine, sine) = ops.rotary_factors(config, layer, positions)?;
    let query = MetalNormalizedProjection {
        weight: metal.layer(layer, LayerParameter::Query)?,
        norm: MetalProjectionNorm {
            width,
            epsilon,
            scale: Some(metal.layer(layer, LayerParameter::QueryNorm)?),
            rotary: Some(MetalRotaryFactors {
                heads,
                cosine: &cosine,
                sine: &sine,
            }),
        },
    };
    let key_value = if geometry.kv_owner == layer {
        Some((
            MetalNormalizedProjection {
                weight: metal.layer(layer, LayerParameter::Key)?,
                norm: MetalProjectionNorm {
                    width,
                    epsilon,
                    scale: Some(metal.layer(layer, LayerParameter::KeyNorm)?),
                    rotary: Some(MetalRotaryFactors {
                        heads: kv_heads,
                        cosine: &cosine,
                        sine: &sine,
                    }),
                },
            },
            MetalNormalizedProjection {
                weight: metal.layer(layer, LayerParameter::Value)?,
                norm: MetalProjectionNorm {
                    width,
                    epsilon,
                    scale: None,
                    rotary: None,
                },
            },
        ))
    } else {
        None
    };
    let columns = heads
        .checked_mul(width)
        .ok_or_else(|| Gemma4Error::new("Metal decoder", "query columns overflow"))?;
    let kv_columns = kv_heads
        .checked_mul(width)
        .ok_or_else(|| Gemma4Error::new("Metal decoder", "KV columns overflow"))?;
    let input_scale = Some(metal.layer(layer, LayerParameter::InputNorm)?);
    let use_chain = chain_eligible(
        sequence,
        hidden_size,
        columns,
        key_value.map(|_| kv_columns),
    );
    if use_chain {
        let input = ops
            .executor
            .upload(hidden)
            .map_err(|error| metal_error(&error))?;
        let native = ops
            .executor
            .rms_attention_projections(
                &input,
                MetalLinearShape {
                    rows: sequence,
                    columns,
                    inner: hidden_size,
                },
                epsilon,
                input_scale,
                query,
                key_value,
            )
            .map_err(|error| metal_error(&error))?;
        drop(input);
        Ok(copy_native(native))
    } else {
        let normed = ops.rms(hidden, sequence, hidden_size, epsilon, input_scale)?;
        let queries = ops.project_normalized(&normed, sequence, query.weight, query.norm)?;
        let key_value = key_value
            .map(|(key, value)| {
                Ok::<_, Gemma4Error>((
                    ops.project_normalized(&normed, sequence, key.weight, key.norm)?,
                    ops.project_normalized(&normed, sequence, value.weight, value.norm)?,
                ))
            })
            .transpose()?;
        Ok(AttentionInputs { queries, key_value })
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "one attention block's semantic inputs, kept explicit"
)]
pub(super) fn attention_forward(
    config: &Gemma4Config,
    metal: &Gemma4MetalWeights,
    caches: &mut [Option<OwnerCache>],
    ops: &MetalOps<'_>,
    geometry: &crate::LayerGeometry,
    layer: usize,
    position: usize,
    sequence: usize,
    epsilon: f32,
    positions: &[u32],
    hidden: &[u16],
) -> Result<Vec<u16>, Gemma4Error> {
    let hidden_size = config.hidden_size();
    let heads = config.num_attention_heads();
    let kv_heads = config.num_key_value_heads();
    let width = geometry.head_dim;
    let AttentionInputs { queries, key_value } = prepare_projections(
        config, metal, ops, geometry, layer, sequence, epsilon, positions, hidden,
    )?;
    if let Some((keys, values)) = key_value {
        let cache = caches[layer]
            .as_mut()
            .ok_or_else(|| Gemma4Error::new("Metal decoder", "owner cache missing"))?;
        cache.keys.extend_from_slice(&keys);
        cache.values.extend_from_slice(&values);
    }
    let cache = caches[geometry.kv_owner]
        .as_ref()
        .ok_or_else(|| Gemma4Error::new("Metal decoder", "owner cache missing"))?;
    let window = match geometry.kind {
        AttentionKind::Sliding => Some(config.sliding_window()),
        AttentionKind::Full => None,
    };
    let normed = project_attention(
        ops,
        &queries,
        cache,
        MetalAttentionConfig {
            heads,
            kv_heads,
            width,
            window,
        },
        position,
        sequence,
        MetalNormalizedProjection {
            weight: metal.layer(layer, LayerParameter::AttentionOutput)?,
            norm: MetalProjectionNorm {
                width: hidden_size,
                epsilon,
                scale: Some(metal.layer(layer, LayerParameter::PostAttentionNorm)?),
                rotary: None,
            },
        },
    )?;
    add(hidden, &normed)
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_CHAIN_BYTES, chain_bytes, chain_eligible, output_chain_bytes, output_chain_eligible,
    };
    use engine_metal::MetalAttentionConfig;

    #[test]
    fn attention_output_chain_bounds_live_scratch_and_history() {
        let config = MetalAttentionConfig {
            heads: 8,
            kv_heads: 2,
            width: 512,
            window: None,
        };
        assert_eq!(output_chain_bytes(64, 8, 512, 576, 2560), Some(1_773_568));
        assert!(output_chain_eligible(64, config, 576, 2560));
        assert!(!output_chain_eligible(65, config, 576, 2560));
        assert!(!output_chain_eligible(1, config, 577, 2560));
        assert!(!output_chain_eligible(0, config, 1, 2560));
        assert!(!output_chain_eligible(1, config, 0, 2560));
        let tiny = MetalAttentionConfig {
            heads: 1,
            kv_heads: 1,
            width: 1,
            window: None,
        };
        let hidden = (MAX_CHAIN_BYTES - 12) / 4;
        assert_eq!(
            output_chain_bytes(1, 1, 1, 1, hidden),
            Some(MAX_CHAIN_BYTES)
        );
        assert!(output_chain_eligible(1, tiny, 1, hidden));
        assert!(!output_chain_eligible(1, tiny, 1, hidden + 1));
        assert_eq!(output_chain_bytes(1, usize::MAX, 1, 1, 1), None);
        assert!(!output_chain_eligible(1, tiny, 1, usize::MAX));
    }

    #[test]
    fn attention_chain_has_a_byte_bound_and_a_large_shape_fallback() {
        for (rows, expected) in [(1, 39_936), (16, 638_976), (64, 2_555_904)] {
            assert_eq!(chain_bytes(rows, 2560, 4096, Some(1024)), Some(expected));
            assert!(chain_eligible(rows, 2560, 4096, Some(1024)));
        }
        assert_eq!(chain_bytes(16, 2560, 2048, Some(512)), Some(360_448));
        assert!(chain_eligible(64, 2560, 2048, None));
        assert!(!chain_eligible(65, 2560, 4096, Some(1024)));
        let inner = MAX_CHAIN_BYTES / 2 - 3;
        assert!(chain_eligible(1, inner, 1, None));
        assert!(!chain_eligible(1, inner + 1, 1, None));
        assert!(!chain_eligible(1, usize::MAX, 1, None));
        assert!(!chain_eligible(0, 2560, 4096, Some(1024)));
    }
}
