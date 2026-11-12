//! Exact-row preparation and the model-wide qualification configuration boundary.
use crate::{AttentionKind, Gemma4Config, Gemma4Error, LayerGeometry, Rope};
use engine_nvidia::CudaBf16RotaryFactors;

pub(super) const SERIAL_PREFILL_LANES: [usize; 2] = [8, 16];
pub(super) const MODEL_WIDE_PREFILL_LANES: [usize; 15] =
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

pub(super) struct Preparation {
    pub rows: usize,
    pub scratch_entries: usize,
}

pub(super) fn prepare(
    config: &Gemma4Config,
    max_positions: usize,
    scratch_ceiling: usize,
    device: &str,
    supported_rows: &[usize],
) -> Result<Preparation, Gemma4Error> {
    if max_positions > config.max_position_embeddings() {
        return Err(invalid("cache capacity exceeds the model context"));
    }
    // Promotion is scoped to the measured device and E4B projection geometries.
    // Other packages/devices retain the existing one-row execution path.
    let qualified = qualified_device_and_projections(config, device);
    plan(
        config.num_attention_heads(),
        max_positions,
        scratch_ceiling,
        qualified,
        supported_rows,
    )
}

fn qualified_device_and_projections(config: &Gemma4Config, device: &str) -> bool {
    device == "NVIDIA GeForce RTX 4090"
        && config.hidden_size() == 2560
        && config.intermediate_size() == 10240
        && config.ple_dim() == 256
        && config.layers().len() == 42
        && config.num_attention_heads() == 8
        && config.num_key_value_heads() == 2
        && config.sliding_window() == 512
        && config.layers().iter().all(|layer| {
            matches!(
                (layer.kind, layer.head_dim),
                (AttentionKind::Sliding, 256) | (AttentionKind::Full, 512)
            )
        })
}

pub(super) fn modelwide_scope(config: &Gemma4Config, device: &str, capacity: usize) -> bool {
    qualified_device_and_projections(config, device)
        && (1..=4096).contains(&capacity)
        && capacity <= config.max_position_embeddings()
        && config.vocab == 262_144
        && config.ple_vocab == 262_144
        && config.packed_ple == 10_752
        && config.bos == 2
        && config.pad == 0
        && config.media_token_ids() == &[258_880, 258_881, 258_884]
        && config.epsilon.to_bits() == 1e-6_f32.to_bits()
        && config.softcap.to_bits() == 30.0_f32.to_bits()
        && config.layers().iter().enumerate().all(|(index, layer)| {
            let full = index % 6 == 5;
            // The 18 shared-tail layers reuse the last local/global owners in
            // the first 24 layers. Match interpreted semantics, not JSON names.
            let kv_owner = if index < 24 {
                index
            } else if full {
                23
            } else {
                22
            };
            *layer
                == LayerGeometry {
                    kind: if full {
                        AttentionKind::Full
                    } else {
                        AttentionKind::Sliding
                    },
                    head_dim: if full { 512 } else { 256 },
                    kv_owner,
                    kv_last_consumer: match kv_owner {
                        22 => 40,
                        23 => 41,
                        _ => kv_owner,
                    },
                    rope: if full {
                        Rope::Proportional { theta: 1_000_000.0 }
                    } else {
                        Rope::Default { theta: 10_000.0 }
                    },
                }
        })
}

fn plan(
    heads: usize,
    max_positions: usize,
    scratch_ceiling: usize,
    qualified: bool,
    supported_rows: &[usize],
) -> Result<Preparation, Gemma4Error> {
    if heads == 0
        || max_positions == 0
        || max_positions > CudaBf16RotaryFactors::POSITION_LIMIT as usize
    {
        return Err(invalid("invalid cache capacity or rotary position reach"));
    }
    let scalar = heads
        .checked_mul(max_positions)
        .filter(|&n| i32::try_from(n).is_ok())
        .ok_or_else(|| invalid("attention scratch extent exceeds matrix limits"))?;
    if scratch_ceiling < scalar {
        return Err(invalid(
            "attention scratch cannot cover the declared cache reach",
        ));
    }
    let rows = supported_rows
        .iter()
        .copied()
        .rev()
        .find(|&rows| {
            qualified
                && (rows..=4096).contains(&max_positions)
                && scalar
                    .checked_mul(rows)
                    .filter(|&n| i32::try_from(n).is_ok())
                    .is_some_and(|n| n <= scratch_ceiling)
        })
        .unwrap_or(1);
    Ok(Preparation {
        rows,
        scratch_entries: scalar * rows,
    })
}

fn invalid(detail: &str) -> Gemma4Error {
    Gemma4Error::new("CUDA decoder preparation", detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinned_json() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../../../../benchmarks/gemma4-reference/config.json"
        ))
        .unwrap()
    }

    fn in_scope(config: &Gemma4Config, capacity: usize) -> bool {
        crate::Gemma4CudaModelWide::matches_qualification_scope(
            config,
            "NVIDIA GeForce RTX 4090",
            capacity,
        )
    }

    #[test]
    fn modelwide_scope_uses_prepared_reach_not_advertised_context_or_scratch_lane() {
        let config = Gemma4Config::from_json(&pinned_json()).unwrap();
        assert_eq!(config.max_position_embeddings(), 131_072);
        for capacity in [1, 18, 513, 4096] {
            assert!(in_scope(&config, capacity));
        }
        for capacity in [0, 4097, usize::MAX] {
            assert!(!in_scope(&config, capacity));
        }
        for device in ["NVIDIA H100 80GB HBM3", "RTX 4090", ""] {
            assert!(!crate::Gemma4CudaModelWide::matches_qualification_scope(
                &config, device, 4096
            ));
        }
        for (lanes, scratch, rows) in [
            (SERIAL_PREFILL_LANES.as_slice(), 32_768, 1),
            (SERIAL_PREFILL_LANES.as_slice(), 229_376, 1),
            (SERIAL_PREFILL_LANES.as_slice(), 262_144, 8),
            (SERIAL_PREFILL_LANES.as_slice(), 524_288, 16),
            (MODEL_WIDE_PREFILL_LANES.as_slice(), 32_768, 1),
            (MODEL_WIDE_PREFILL_LANES.as_slice(), 229_376, 7),
            (MODEL_WIDE_PREFILL_LANES.as_slice(), 262_144, 8),
            (MODEL_WIDE_PREFILL_LANES.as_slice(), 524_288, 16),
        ] {
            let prepared =
                prepare(&config, 4096, scratch, "NVIDIA GeForce RTX 4090", lanes).unwrap();
            assert_eq!(prepared.rows, rows);
            assert!(in_scope(&config, 4096));
        }
        let mut json = pinned_json();
        json["text_config"]["max_position_embeddings"] = 64.into();
        let small = Gemma4Config::from_json(&json).unwrap();
        assert!(in_scope(&small, 64));
        assert!(!in_scope(&small, 65));
    }

    #[test]
    fn modelwide_scope_rejects_semantic_near_misses_without_tightening_prefill() {
        for (pointer, value) in [
            ("/text_config/vocab_size", serde_json::json!(262_143)),
            (
                "/text_config/vocab_size_per_layer_input",
                serde_json::json!(262_145),
            ),
            ("/text_config/bos_token_id", serde_json::json!(3)),
            ("/text_config/pad_token_id", serde_json::json!(1)),
            ("/image_token_id", serde_json::json!(258_879)),
            ("/audio_token_id", serde_json::json!(258_878)),
            ("/video_token_id", serde_json::json!(258_877)),
            ("/text_config/rms_norm_eps", serde_json::json!(1e-5)),
            (
                "/text_config/final_logit_softcapping",
                serde_json::json!(20.0),
            ),
            ("/text_config/num_kv_shared_layers", serde_json::json!(16)),
            ("/text_config/num_kv_shared_layers", serde_json::json!(20)),
            (
                "/text_config/rope_parameters/sliding_attention/rope_theta",
                serde_json::json!(20_000.0),
            ),
            (
                "/text_config/rope_parameters/full_attention/rope_theta",
                serde_json::json!(2_000_000.0),
            ),
        ] {
            let mut json = pinned_json();
            *json.pointer_mut(pointer).unwrap() = value;
            let config = Gemma4Config::from_json(&json).unwrap();
            assert!(!in_scope(&config, 4096), "{pointer}");
            assert_eq!(
                prepare(
                    &config,
                    4096,
                    1 << 20,
                    "NVIDIA GeForce RTX 4090",
                    &SERIAL_PREFILL_LANES,
                )
                .unwrap()
                .rows,
                16,
                "the existing prefill qualification scope must not change: {pointer}"
            );
        }
        let mut json = pinned_json();
        json["text_config"]["layer_types"]
            .as_array_mut()
            .unwrap()
            .swap(4, 5);
        let config = Gemma4Config::from_json(&json).unwrap();
        assert!(
            !in_scope(&config, 4096),
            "same widths, different attention order"
        );
        assert_eq!(
            prepare(
                &config,
                4096,
                1 << 20,
                "NVIDIA GeForce RTX 4090",
                &SERIAL_PREFILL_LANES,
            )
            .unwrap()
            .rows,
            16
        );
    }

    #[test]
    fn modelwide_scope_rejects_projection_geometry_changes() {
        for (field, value) in [
            ("hidden_size", 2568),
            ("intermediate_size", 10248),
            ("hidden_size_per_layer_input", 128),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 4),
            ("sliding_window", 1024),
            ("head_dim", 128),
            ("global_head_dim", 256),
        ] {
            let mut json = pinned_json();
            json["text_config"][field] = value.into();
            let config = Gemma4Config::from_json(&json).unwrap();
            assert!(!in_scope(&config, 4096), "{field}");
        }
        let mut json = pinned_json();
        json["text_config"]["num_hidden_layers"] = 36.into();
        json["text_config"]["layer_types"]
            .as_array_mut()
            .unwrap()
            .truncate(36);
        assert!(!in_scope(&Gemma4Config::from_json(&json).unwrap(), 4096));
    }

    #[test]
    fn modelwide_exact_widths_preserve_serial_selection_at_scratch_and_reach_boundaries() {
        for rows in 2..=16 {
            let serial_rows = if rows == 16 {
                16
            } else if rows >= 8 {
                8
            } else {
                1
            };
            let serial_below = if rows > 8 { 8 } else { 1 };
            for positions in [rows, 513, 4096] {
                let scalar = 8 * positions;
                let ceiling = scalar * rows;
                for (lanes, expected, expected_below) in [
                    (MODEL_WIDE_PREFILL_LANES.as_slice(), rows, rows - 1),
                    (SERIAL_PREFILL_LANES.as_slice(), serial_rows, serial_below),
                ] {
                    let exact = plan(8, positions, ceiling, true, lanes).unwrap();
                    assert_eq!(exact.rows, expected);
                    assert_eq!(exact.scratch_entries, scalar * expected);
                    let below = plan(8, positions, ceiling - 1, true, lanes).unwrap();
                    assert_eq!(below.rows, expected_below);
                    assert_eq!(below.scratch_entries, scalar * expected_below);
                    assert_eq!(plan(8, positions, ceiling, false, lanes).unwrap().rows, 1);
                }
            }
            assert_eq!(
                plan(8, rows - 1, 1 << 20, true, &MODEL_WIDE_PREFILL_LANES)
                    .unwrap()
                    .rows,
                rows - 1
            );
            assert_eq!(
                plan(8, rows - 1, 1 << 20, true, &SERIAL_PREFILL_LANES)
                    .unwrap()
                    .rows,
                serial_below
            );
        }
    }

    #[test]
    fn scratch_covers_the_whole_reach_before_selecting_a_variant_prefill_width() {
        for (positions, ceiling, qualified, serial, modelwide) in [
            (32, 256, true, 1, 1),
            (32, 2047, true, 1, 7),
            (32, 2048, true, 8, 8),
            (32, 2048, false, 1, 1),
            (32, 4095, true, 8, 15),
            (32, 4096, true, 16, 16),
            (32, 4096, false, 1, 1),
            (15, 1920, true, 8, 15),
            (16, 2048, true, 16, 16),
            (7, 448, true, 1, 7),
            (8, 512, true, 8, 8),
            (4096, 262_144, true, 8, 8),
            (4096, 524_288, true, 16, 16),
            (4097, 262_208, true, 1, 1),
        ] {
            for (lanes, expected) in [
                (SERIAL_PREFILL_LANES.as_slice(), serial),
                (MODEL_WIDE_PREFILL_LANES.as_slice(), modelwide),
            ] {
                let p = plan(8, positions, ceiling, qualified, lanes).unwrap();
                assert_eq!(p.rows, expected);
                assert_eq!(p.scratch_entries, positions * 8 * expected);
                assert!(p.scratch_entries <= ceiling);
            }
        }
        for (heads, positions, ceiling) in [
            (8, 32, 255),
            (8, 0, 0),
            (0, 32, 256),
            (usize::MAX, 32, usize::MAX),
            (
                8,
                CudaBf16RotaryFactors::POSITION_LIMIT as usize + 1,
                usize::MAX,
            ),
        ] {
            for lanes in [
                SERIAL_PREFILL_LANES.as_slice(),
                MODEL_WIDE_PREFILL_LANES.as_slice(),
            ] {
                assert!(plan(heads, positions, ceiling, true, lanes).is_err());
            }
        }
    }
}
