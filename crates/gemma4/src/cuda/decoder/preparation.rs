//! Fixed-lane qualification and whole-reach attention workspace preparation.
use crate::{AttentionKind, Gemma4Config, Gemma4Error};
use engine_nvidia::CudaBf16RotaryFactors;

pub(super) const PREFILL_LANES: [usize; 2] = [8, 16];

pub(super) struct Preparation {
    pub rows: usize,
    pub scratch_entries: usize,
}

pub(super) fn prepare(
    config: &Gemma4Config,
    max_positions: usize,
    scratch_ceiling: usize,
    device: &str,
) -> Result<Preparation, Gemma4Error> {
    if max_positions > config.max_position_embeddings() {
        return Err(invalid("cache capacity exceeds the model context"));
    }
    // Promotion is scoped to the measured device and E4B projection geometries.
    // Other packages/devices retain the existing one-row execution path.
    let qualified = device == "NVIDIA GeForce RTX 4090"
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
        });
    plan(
        config.num_attention_heads(),
        max_positions,
        scratch_ceiling,
        qualified,
    )
}

fn plan(
    heads: usize,
    max_positions: usize,
    scratch_ceiling: usize,
    qualified: bool,
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
    let rows = PREFILL_LANES
        .into_iter()
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

    #[test]
    fn scratch_covers_the_whole_reach_before_selecting_a_fixed_lane() {
        for (positions, ceiling, qualified, expected) in [
            (32, 256, true, 1),
            (32, 2047, true, 1),
            (32, 2048, true, 8),
            (32, 2048, false, 1),
            (32, 4095, true, 8),
            (32, 4096, true, 16),
            (32, 4096, false, 1),
            (15, 1920, true, 8),
            (16, 2048, true, 16),
            (7, 448, true, 1),
            (8, 512, true, 8),
            (4096, 262_144, true, 8),
            (4096, 524_288, true, 16),
            (4097, 262_208, true, 1),
        ] {
            let p = plan(8, positions, ceiling, qualified).unwrap();
            assert_eq!(p.rows, expected);
            assert_eq!(p.scratch_entries, positions * 8 * expected);
            assert!(p.scratch_entries <= ceiling);
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
            assert!(plan(heads, positions, ceiling, true).is_err());
        }
    }
}
