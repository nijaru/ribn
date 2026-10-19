//! Split-half rotary preparation and application with eager BF16 rounding.

use crate::cpu::linear::bf16;
use crate::cpu::try_vector;
use crate::{Gemma4Config, Gemma4Error};

/// Positions must convert to FP32 without rounding, as the pinned path does.
const MAX_EXACT_POSITION: u32 = 1 << 24;

/// Padded half-head FP32 inverse frequencies for one decoder layer.
///
/// Proportional padding is positive zero, supplied by the model configuration.
/// # Errors
/// Rejects an unknown layer or non-representable frequencies.
pub fn inverse_frequencies(config: &Gemma4Config, layer: usize) -> Result<Vec<f32>, Gemma4Error> {
    let geometry = config
        .layers()
        .get(layer)
        .ok_or_else(|| Gemma4Error::new("RoPE layer", "index outside decoder"))?;
    let half = geometry.head_dim / 2;
    (0..half)
        .map(|pair| config.rotary_inverse_frequency(layer, pair))
        .collect()
}

/// BF16 cosine/sine factors `[rows, head_dim]` for supplied integer positions.
///
/// Phase is FP32 `position × frequency`, cosine/sine are FP32, scaling is exactly
/// 1.0 in the pinned path and both halves are duplicated. This does not select
/// positions or model layer binding.
/// # Errors
/// Rejects empty/inexact positions, unknown layers or non-finite values.
pub fn factors(
    config: &Gemma4Config,
    layer: usize,
    positions: &[u32],
) -> Result<(Vec<u16>, Vec<u16>), Gemma4Error> {
    if positions.is_empty() {
        return Err(Gemma4Error::new(
            "RoPE positions",
            "at least one position is required",
        ));
    }
    if positions
        .iter()
        .any(|&position| position >= MAX_EXACT_POSITION)
    {
        return Err(Gemma4Error::new(
            "RoPE positions",
            "position does not convert to FP32 exactly",
        ));
    }
    let frequencies = inverse_frequencies(config, layer)?;
    let half = frequencies.len();
    let width = half
        .checked_mul(2)
        .ok_or_else(|| Gemma4Error::new("RoPE factors", "head width overflow"))?;
    let length = positions
        .len()
        .checked_mul(width)
        .ok_or_else(|| Gemma4Error::new("RoPE factors", "factor length overflow"))?;
    let mut cosine = try_vector(length, 0_u16)?;
    let mut sine = try_vector(length, 0_u16)?;
    for (row, &position) in positions.iter().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "validated positions below 2^24 convert to FP32 exactly"
        )]
        let position = position as f32;
        for (pair, &frequency) in frequencies.iter().enumerate() {
            let phase = position * frequency;
            let c = bf16(phase.cos());
            let s = bf16(phase.sin());
            let base = row * width + pair;
            cosine[base] = c;
            cosine[base + half] = c;
            sine[base] = s;
            sine[base + half] = s;
        }
    }
    Ok((cosine, sine))
}

/// Apply BF16 factors `[rows, width]` to every head of `[rows, heads × width]`.
///
/// Channel `j` pairs with `j + width/2`; each product rounds to BF16 before the
/// FP32 addition, then the sum rounds to BF16. Both factor halves are read.
/// # Errors
/// Rejects zero/odd widths, non-whole heads or disagreeing shapes.
pub fn apply_rotary(
    input: &[u16],
    rows: usize,
    heads: usize,
    width: usize,
    cosine: &[u16],
    sine: &[u16],
) -> Result<Vec<u16>, Gemma4Error> {
    let input_length = rows.checked_mul(heads).and_then(|n| n.checked_mul(width));
    let factor_length = rows.checked_mul(width);
    if width == 0
        || !width.is_multiple_of(2)
        || rows == 0
        || heads == 0
        || Some(input.len()) != input_length
        || Some(cosine.len()) != factor_length
        || Some(sine.len()) != factor_length
    {
        return Err(Gemma4Error::new(
            "CPU rotary",
            "expect [rows, heads*width] input and [rows, width] factors",
        ));
    }
    let half = width / 2;
    let mut output = try_vector(input.len(), 0_u16)?;
    for row in 0..rows {
        for head in 0..heads {
            let base = (row * heads + head) * width;
            for j in 0..half {
                let factor = row * width + j;
                let a = f32::from_bits(u32::from(input[base + j]) << 16);
                let b = f32::from_bits(u32::from(input[base + j + half]) << 16);
                let c1 = f32::from_bits(u32::from(cosine[factor]) << 16);
                let s1 = f32::from_bits(u32::from(sine[factor]) << 16);
                let c2 = f32::from_bits(u32::from(cosine[factor + half]) << 16);
                let s2 = f32::from_bits(u32::from(sine[factor + half]) << 16);
                let first = multiply(a, c1) + multiply(-b, s1);
                let second = multiply(b, c2) + multiply(a, s2);
                output[base + j] = bf16(first);
                output[base + j + half] = bf16(second);
            }
        }
    }
    Ok(output)
}

/// One BF16-rounded product, loaded back as FP32 for the addition.
fn multiply(a: f32, b: f32) -> f32 {
    let rounded = bf16(a * b);
    f32::from_bits(u32::from(rounded) << 16)
}
