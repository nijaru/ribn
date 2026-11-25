//! Model-neutral deterministic FP32/BF16 token sampling (`temperature`, `top_k`, `top_p`, `seed`).

use crate::{ExecutionError, RequestId, Sampling};

#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[inline]
fn next_unit_f64(state: &mut u64) -> f64 {
    // SplitMix64 mixing into the 53-bit IEEE-754 mantissa in [0, 1).
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    let mantissa = z >> 11;
    #[expect(
        clippy::cast_precision_loss,
        reason = "53-bit integer fits in f64 mantissa without loss"
    )]
    {
        (mantissa as f64) * (1.0 / ((1_u64 << 53) as f64))
    }
}

/// Initialize per-sequence deterministic PRNG state from an optional user `seed`
/// or the unique `request_id`.
#[must_use]
pub const fn init_rng_state(seed: Option<u64>, request_id: RequestId) -> u64 {
    match seed {
        Some(seed) => seed,
        None => request_id.get().wrapping_mul(0x517c_c1b7_2722_0a95),
    }
}

fn validate_sampling(sampling: Sampling) -> Result<(), ExecutionError> {
    if !sampling.temperature.is_finite() || sampling.temperature < 0.0 {
        return Err(ExecutionError::new("invalid sampling temperature"));
    }
    if !sampling.top_p.is_finite() || sampling.top_p <= 0.0 || sampling.top_p > 1.0 {
        return Err(ExecutionError::new("invalid sampling top_p"));
    }
    Ok(())
}

fn sample_candidates(
    mut candidates: Vec<(u32, f32)>,
    sampling: Sampling,
    rng_state: &mut u64,
) -> Result<u32, ExecutionError> {
    if candidates.is_empty() {
        return Err(ExecutionError::new("empty logits"));
    }
    if sampling.temperature == 0.0 {
        let mut best_id = candidates[0].0;
        let mut best_val = candidates[0].1;
        for &(id, val) in &candidates[1..] {
            if matches!(
                val.total_cmp(&best_val),
                std::cmp::Ordering::Greater
                    | std::cmp::Ordering::Equal if val > best_val || id < best_id
            ) {
                best_val = val;
                best_id = id;
            }
        }
        return Ok(best_id);
    }

    let cmp_desc = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));

    let top_k = sampling.top_k as usize;
    if top_k > 0 && top_k < candidates.len() {
        candidates.select_nth_unstable_by(top_k - 1, cmp_desc);
        candidates.truncate(top_k);
    }
    if sampling.top_p < 1.0 {
        candidates.sort_unstable_by(cmp_desc);
    }

    let inv_temp = 1.0 / f64::from(sampling.temperature);
    let max_val = candidates
        .iter()
        .map(|&(_, v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(u32, f64)> = Vec::with_capacity(candidates.len());
    let mut sum = 0.0_f64;
    for &(id, value) in &candidates {
        let p = ((f64::from(value) - f64::from(max_val)) * inv_temp).exp();
        sum += p;
        probs.push((id, p));
    }
    if !sum.is_finite() || sum <= 0.0 {
        let best = candidates
            .iter()
            .copied()
            .min_by(cmp_desc)
            .map_or(0, |(id, _)| id);
        return Ok(best);
    }

    if sampling.top_p < 1.0 {
        let threshold = f64::from(sampling.top_p) * sum;
        let mut cumulative = 0.0_f64;
        let mut keep = 0_usize;
        for &(_, p) in &probs {
            cumulative += p;
            keep += 1;
            if cumulative >= threshold {
                break;
            }
        }
        probs.truncate(keep.max(1));
        sum = cumulative;
    }

    let target = next_unit_f64(rng_state) * sum;
    let mut acc = 0.0_f64;
    for &(id, p) in &probs {
        acc += p;
        if acc >= target {
            return Ok(id);
        }
    }
    Ok(probs.last().map_or(0, |&(id, _)| id))
}

/// Sample one token ID from FP32 logits according to `sampling`.
///
/// Greedy (`temperature == 0.0`) returns exact deterministic `argmax` (lowest token ID on ties).
///
/// # Errors
/// Rejects empty logits, non-finite logits, vocab indices exceeding `u32`, or invalid sampling params.
pub fn sample_f32_logits(
    logits: &[f32],
    sampling: Sampling,
    rng_state: &mut u64,
) -> Result<u32, ExecutionError> {
    if logits.is_empty() {
        return Err(ExecutionError::new("empty logits"));
    }
    validate_sampling(sampling)?;
    if sampling.temperature == 0.0 {
        let mut best_idx = 0_usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &value) in logits.iter().enumerate() {
            if !value.is_finite() {
                return Err(ExecutionError::new("non-finite logit"));
            }
            if value > best_val {
                best_val = value;
                best_idx = idx;
            }
        }
        return u32::try_from(best_idx).map_err(|_| ExecutionError::new("vocab index exceeds u32"));
    }

    let mut candidates = Vec::with_capacity(logits.len());
    for (idx, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(ExecutionError::new("non-finite logit"));
        }
        let id = u32::try_from(idx).map_err(|_| ExecutionError::new("vocab index exceeds u32"))?;
        candidates.push((id, value));
    }
    sample_candidates(candidates, sampling, rng_state)
}

/// Sample one token ID from BF16 logits (IEEE-754 top-16-bit `u16` encodings) according to `sampling`.
///
/// Greedy (`temperature == 0.0`) returns exact deterministic `argmax` (lowest token ID on ties).
///
/// # Errors
/// Rejects empty logits, non-finite logits, vocab indices exceeding `u32`, or invalid sampling params.
pub fn sample_bf16_logits(
    logits: &[u16],
    sampling: Sampling,
    rng_state: &mut u64,
) -> Result<u32, ExecutionError> {
    if logits.is_empty() {
        return Err(ExecutionError::new("empty logits"));
    }
    validate_sampling(sampling)?;
    if sampling.temperature == 0.0 {
        let mut best_idx = 0_usize;
        let mut best_val = f32::NEG_INFINITY;
        for (idx, &bits) in logits.iter().enumerate() {
            let value = bf16_to_f32(bits);
            if !value.is_finite() {
                return Err(ExecutionError::new("non-finite logit"));
            }
            if value > best_val {
                best_val = value;
                best_idx = idx;
            }
        }
        return u32::try_from(best_idx).map_err(|_| ExecutionError::new("vocab index exceeds u32"));
    }

    let mut candidates = Vec::with_capacity(logits.len());
    for (idx, &bits) in logits.iter().enumerate() {
        let value = bf16_to_f32(bits);
        if !value.is_finite() {
            return Err(ExecutionError::new("non-finite logit"));
        }
        let id = u32::try_from(idx).map_err(|_| ExecutionError::new("vocab index exceeds u32"))?;
        candidates.push((id, value));
    }
    sample_candidates(candidates, sampling, rng_state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_to_bf16_trunc(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    #[test]
    fn greedy_argmax_ties_and_rejections_are_exact() {
        let mut rng = 0_u64;
        let greedy = Sampling::default();
        assert_eq!(
            sample_f32_logits(&[1.0, 3.0, 3.0, 2.0], greedy, &mut rng).unwrap(),
            1
        );
        assert!(sample_f32_logits(&[], greedy, &mut rng).is_err());
        assert!(sample_f32_logits(&[1.0, f32::NAN], greedy, &mut rng).is_err());
        assert!(sample_f32_logits(&[1.0, f32::INFINITY], greedy, &mut rng).is_err());
        assert!(
            sample_f32_logits(
                &[1.0, 2.0],
                Sampling {
                    temperature: -0.1,
                    ..greedy
                },
                &mut rng
            )
            .is_err()
        );
        assert!(
            sample_f32_logits(
                &[1.0, 2.0],
                Sampling {
                    temperature: 0.7,
                    top_p: 0.0,
                    ..greedy
                },
                &mut rng
            )
            .is_err()
        );
    }

    #[test]
    fn stochastic_sampling_is_deterministic_respects_top_k_top_p_and_matches_bf16() {
        let logits_bf16: Vec<u16> = [-2.0_f32, 0.5, 4.0, 3.5, 1.0, -5.0, 3.0, 2.0]
            .into_iter()
            .map(f32_to_bf16_trunc)
            .collect();
        let logits_f32: Vec<f32> = logits_bf16.iter().copied().map(bf16_to_f32).collect();

        let cfg = Sampling {
            temperature: 0.9,
            top_k: 3,
            top_p: 0.85,
            seed: Some(12345),
        };
        let mut rng_a = init_rng_state(cfg.seed, RequestId(1));
        let mut rng_b = init_rng_state(cfg.seed, RequestId(99));
        let mut rng_bf16 = init_rng_state(cfg.seed, RequestId(1));

        let mut draws_a = Vec::new();
        let mut draws_b = Vec::new();
        let mut draws_bf16 = Vec::new();
        for _ in 0..64 {
            let a = sample_f32_logits(&logits_f32, cfg, &mut rng_a).unwrap();
            let b = sample_f32_logits(&logits_f32, cfg, &mut rng_b).unwrap();
            let c = sample_bf16_logits(&logits_bf16, cfg, &mut rng_bf16).unwrap();
            // Top-3 indices are 2 (4.0), 3 (3.5), 6 (3.0)
            assert!(matches!(a, 2 | 3 | 6), "outside top_k=3: {a}");
            draws_a.push(a);
            draws_b.push(b);
            draws_bf16.push(c);
        }
        assert_eq!(draws_a, draws_b, "explicit seed must be deterministic");
        assert_eq!(draws_a, draws_bf16, "BF16 and decoded F32 must match");
        // Verify non-degenerate distribution (more than 1 unique token drawn)
        assert!(draws_a.iter().any(|&t| t != draws_a[0]));
    }

    #[test]
    fn extreme_logits_and_tiny_temperature_remain_finite() {
        let mut rng = 7_u64;
        let cfg = Sampling {
            temperature: 1e-4,
            top_k: 0,
            top_p: 1.0,
            seed: Some(7),
        };
        assert_eq!(
            sample_f32_logits(&[-1000.0, 1000.0, 999.0], cfg, &mut rng).unwrap(),
            1
        );
    }
}
