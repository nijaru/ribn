//! Deterministic FP32 token sampling over BF16 soft-capped logits.

use crate::{Gemma4Error, argmax};
use ribn::Sampling;

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

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

/// Sample one token ID from BF16 soft-capped logits according to `sampling`.
///
/// Greedy (`temperature == 0.0`) returns exact deterministic `argmax`.
/// # Errors
/// Rejects empty logits, non-finite values, or invalid sampling parameters.
pub(crate) fn sample_bf16_logits(
    logits: &[u16],
    sampling: Sampling,
    rng_state: &mut u64,
) -> Result<u32, Gemma4Error> {
    if logits.is_empty() {
        return Err(Gemma4Error::new("sampling", "empty logits"));
    }
    if sampling.temperature == 0.0 {
        return argmax(logits).ok_or_else(|| Gemma4Error::new("sampling", "empty logits"));
    }
    if !sampling.temperature.is_finite() || sampling.temperature < 0.0 {
        return Err(Gemma4Error::new("sampling", "invalid temperature"));
    }
    if !sampling.top_p.is_finite() || sampling.top_p <= 0.0 || sampling.top_p > 1.0 {
        return Err(Gemma4Error::new("sampling", "invalid top_p"));
    }
    let inv_temp = 1.0 / f64::from(sampling.temperature);
    let vocab = logits.len();
    let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(vocab);
    for (idx, &bits) in logits.iter().enumerate() {
        let value = bf16_to_f32(bits);
        if !value.is_finite() {
            return Err(Gemma4Error::new("sampling", "non-finite logit"));
        }
        let id = u32::try_from(idx)
            .map_err(|_| Gemma4Error::new("sampling", "vocab index exceeds u32"))?;
        candidates.push((id, value));
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
        return argmax(logits).ok_or_else(|| Gemma4Error::new("sampling", "degenerate softmax"));
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
