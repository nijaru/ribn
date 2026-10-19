//! Causal grouped-query attention with BF16 scores/weights and FP32 softmax.

use crate::Gemma4Error;
use crate::cpu::linear::bf16;
use crate::cpu::try_vector;

/// Attention geometry and mask policy for one layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionConfig {
    pub heads: usize,
    pub kv_heads: usize,
    pub width: usize,
    /// Sliding window length in positions; `None` is full attention.
    pub window: Option<usize>,
}

fn load(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// One attention step over `query` positions against `keys`/`values`.
///
/// `query` is `[seq, heads × width]`, keys/values `[kv_seq, kv_heads × width]`.
/// Absolute starts make cached/trimmed key spans explicit. A `window` enables
/// sliding attention; `None` is full attention. Both are causal. Containment is
/// checked, not coverage: the caller must pass every key its cache policy
/// requires, so an over-trimmed span is not detected here.
///
/// The pinned policy is reproduced: BF16 dot-product scores, FP32 softmax cast to
/// BF16 weights, then BF16 `weights × values` with one FP32 accumulation.
/// # Errors
/// Rejects empty/overflowing shapes, non-whole heads, future queries and mismatches.
pub fn attention(
    query: &[u16],
    keys: &[u16],
    values: &[u16],
    config: AttentionConfig,
    query_start: usize,
    key_start: usize,
) -> Result<Vec<u16>, Gemma4Error> {
    attention_with_weights(query, keys, values, config, query_start, key_start)
        .map(|(output, _)| output)
}

/// Attention output plus the BF16 softmax weights `[heads, seq, kv_seq]`.
///
/// Audit/diagnostic entry point; masked positions are exactly positive zero.
/// # Errors
/// As [`attention`].
pub fn attention_with_weights(
    query: &[u16],
    keys: &[u16],
    values: &[u16],
    config: AttentionConfig,
    query_start: usize,
    key_start: usize,
) -> Result<(Vec<u16>, Vec<u16>), Gemma4Error> {
    let view = View::new(query, keys, values, config, query_start, key_start)?;
    let mut weights = try_vector(
        view.heads
            .checked_mul(view.sequence)
            .and_then(|n| n.checked_mul(view.key_sequence))
            .ok_or_else(|| Gemma4Error::new("CPU attention", "weight count overflow"))?,
        0_u16,
    )?;
    let mut output = try_vector(query.len(), 0_u16)?;
    let mut scores = try_vector(view.key_sequence, 0.0_f32)?;
    let mut accumulator = try_vector(view.width, 0.0_f32)?;
    for position in 0..view.sequence {
        view.position(
            position,
            &mut weights,
            &mut output,
            &mut scores,
            &mut accumulator,
        );
    }
    Ok((output, weights))
}

struct View<'a> {
    query: &'a [u16],
    keys: &'a [u16],
    values: &'a [u16],
    heads: usize,
    kv_heads: usize,
    width: usize,
    window: Option<usize>,
    query_start: usize,
    key_start: usize,
    sequence: usize,
    key_sequence: usize,
    head_step: usize,
    kv_step: usize,
}

impl<'a> View<'a> {
    fn new(
        query: &'a [u16],
        keys: &'a [u16],
        values: &'a [u16],
        config: AttentionConfig,
        query_start: usize,
        key_start: usize,
    ) -> Result<Self, Gemma4Error> {
        let AttentionConfig {
            heads,
            kv_heads,
            width,
            window,
        } = config;
        if heads == 0 || kv_heads == 0 || width == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(Gemma4Error::new(
                "CPU attention",
                "heads, key-value heads and width must be positive whole groups",
            ));
        }
        if window == Some(0) {
            return Err(Gemma4Error::new(
                "CPU attention",
                "sliding window must be positive",
            ));
        }
        let head_step = heads
            .checked_mul(width)
            .ok_or_else(|| Gemma4Error::new("CPU attention", "head stride overflow"))?;
        let kv_step = kv_heads
            .checked_mul(width)
            .ok_or_else(|| Gemma4Error::new("CPU attention", "key stride overflow"))?;
        if query.is_empty()
            || !query.len().is_multiple_of(head_step)
            || keys.len() != values.len()
            || keys.is_empty()
            || !keys.len().is_multiple_of(kv_step)
        {
            return Err(Gemma4Error::new(
                "CPU attention",
                "query/key shapes are not whole positions",
            ));
        }
        let sequence = query.len() / head_step;
        let key_sequence = keys.len() / kv_step;
        let last_query = query_start
            .checked_add(sequence)
            .ok_or_else(|| Gemma4Error::new("CPU attention", "query positions overflow"))?;
        let last_key = key_start
            .checked_add(key_sequence)
            .ok_or_else(|| Gemma4Error::new("CPU attention", "key positions overflow"))?;
        if last_query > last_key || query_start < key_start {
            return Err(Gemma4Error::new(
                "CPU attention",
                "queries must not precede or outrun the key span",
            ));
        }
        Ok(Self {
            query,
            keys,
            values,
            heads,
            kv_heads,
            width,
            window,
            query_start,
            key_start,
            sequence,
            key_sequence,
            head_step,
            kv_step,
        })
    }

    fn visible(&self, position: usize) -> (usize, usize) {
        let absolute = self.query_start + position;
        let last = (absolute - self.key_start).min(self.key_sequence - 1);
        let first = match self.window {
            None => 0,
            Some(window) => (absolute.saturating_sub(window - 1))
                .saturating_sub(self.key_start)
                .min(last),
        };
        (first, last)
    }

    /// Weights and output for one query position, all heads.
    fn position(
        &self,
        position: usize,
        weights: &mut [u16],
        output: &mut [u16],
        scores: &mut [f32],
        accumulator: &mut [f32],
    ) {
        let (first, last) = self.visible(position);
        let groups = self.heads / self.kv_heads;
        for head in 0..self.heads {
            let kv_head = head / groups;
            let query_base = position * self.head_step + head * self.width;
            let mut maximum = f32::NEG_INFINITY;
            for (offset, score) in scores[first..=last].iter_mut().enumerate() {
                let key_base = (first + offset) * self.kv_step + kv_head * self.width;
                let mut sum = 0.0_f32;
                for column in 0..self.width {
                    sum +=
                        load(self.query[query_base + column]) * load(self.keys[key_base + column]);
                }
                // The pinned path multiplies BF16 scores by scaling 1.0.
                *score = load(bf16(sum));
                maximum = maximum.max(*score);
            }
            let mut total = 0.0_f32;
            for score in &mut scores[first..=last] {
                *score = (*score - maximum).exp();
                total += *score;
            }
            let weight_base =
                head * self.sequence * self.key_sequence + position * self.key_sequence;
            accumulator.fill(0.0);
            for (offset, &score) in scores[first..=last].iter().enumerate() {
                let weight = bf16(score / total);
                weights[weight_base + first + offset] = weight;
                let value_base = (first + offset) * self.kv_step + kv_head * self.width;
                for (slot, &value) in accumulator
                    .iter_mut()
                    .zip(&self.values[value_base..value_base + self.width])
                {
                    *slot += load(weight) * load(value);
                }
            }
            let output_base = position * self.head_step + head * self.width;
            for (slot, &value) in output[output_base..output_base + self.width]
                .iter_mut()
                .zip(&*accumulator)
            {
                *slot = bf16(value);
            }
        }
    }
}
