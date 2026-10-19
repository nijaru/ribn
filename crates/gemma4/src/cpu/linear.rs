//! BF16 matrix products with FP32 accumulation and one final BF16 rounding.

use crate::cpu::try_vector;
use crate::{CpuParameter, Gemma4Error};

/// Round an FP32 value to the nearest BF16 encoding, ties to even.
///
/// NaN keeps a canonical encoding rather than rounding its payload to infinity.
#[must_use]
pub fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return 0x7fc0;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Row-major `output[m,n] = input[m,k] · weight[n,k]` with native BF16 operands.
///
/// Accumulation is FP32 in source order and the single output rounding is BF16
/// round-to-nearest-even, matching the eager CPU reference policy. This is a
/// straightforward host kernel, not a performance claim.
/// # Errors
/// Rejects non-matrix weights or shapes that disagree with the input.
pub fn linear(input: &[u16], rows: usize, weight: &CpuParameter) -> Result<Vec<u16>, Gemma4Error> {
    let (sums, _) = linear_f32(input, rows, weight)?;
    let mut output = try_vector(sums.len(), 0_u16)?;
    for (slot, value) in output.iter_mut().zip(sums) {
        *slot = bf16(value);
    }
    Ok(output)
}

/// FP32 dot products and their accumulated absolute magnitudes, without rounding.
///
/// The magnitude vector bounds how far a different FP32 summation order can move
/// each result, which is what an independent-reference audit needs.
/// # Errors
/// Rejects non-matrix weights or shapes that disagree with the input.
pub fn linear_f32(
    input: &[u16],
    rows: usize,
    weight: &CpuParameter,
) -> Result<(Vec<f32>, Vec<f32>), Gemma4Error> {
    let shape = weight.shape();
    if shape.len() != 2 || rows == 0 {
        return Err(Gemma4Error::new(
            "CPU linear",
            "weight must be [n,k] and rows positive",
        ));
    }
    let (columns, inner) = (shape[0], shape[1]);
    let input_length = rows
        .checked_mul(inner)
        .ok_or_else(|| Gemma4Error::new("CPU linear", "input dimensions overflow"))?;
    let weight_length = columns
        .checked_mul(inner)
        .ok_or_else(|| Gemma4Error::new("CPU linear", "weight dimensions overflow"))?;
    let output_length = rows
        .checked_mul(columns)
        .ok_or_else(|| Gemma4Error::new("CPU linear", "output dimensions overflow"))?;
    if columns == 0
        || inner == 0
        || input.len() != input_length
        || weight.values().len() != weight_length
    {
        return Err(Gemma4Error::new(
            "CPU linear",
            "input, weight and output dimensions disagree",
        ));
    }
    let mut sums = try_vector(output_length, 0.0_f32)?;
    let mut magnitudes = try_vector(output_length, 0.0_f32)?;
    for row in 0..rows {
        let row_values = &input[row * inner..(row + 1) * inner];
        for column in 0..columns {
            let slice = &weight.values()[column * inner..(column + 1) * inner];
            let mut sum = 0.0_f32;
            let mut magnitude = 0.0_f32;
            for (&left, &right) in row_values.iter().zip(slice) {
                let left = f32::from_bits(u32::from(left) << 16);
                let right = f32::from_bits(u32::from(right) << 16);
                let product = left * right;
                sum += product;
                magnitude += product.abs();
            }
            sums[row * columns + column] = sum;
            magnitudes[row * columns + column] = magnitude;
        }
    }
    Ok((sums, magnitudes))
}
