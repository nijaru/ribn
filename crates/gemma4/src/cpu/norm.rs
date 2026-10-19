//! BF16 RMS normalization with FP32 statistics and direct learned scale.

use crate::cpu::linear::bf16;
use crate::cpu::try_vector;
use crate::{CpuParameter, Gemma4Error};

/// Normalize each row with FP32 mean square, inverse square root and scale.
///
/// The implementation matches the pinned RMS policy: FP32 variance and
/// normalization, direct BF16 scale, and one final BF16 round-to-nearest-even.
/// # Errors
/// Rejects zero width, non-positive/non-finite epsilon or a scale of wrong width.
pub fn rms_norm(
    input: &[u16],
    rows: usize,
    width: usize,
    epsilon: f32,
    scale: &CpuParameter,
) -> Result<Vec<u16>, Gemma4Error> {
    let expected = rows.checked_mul(width);
    if width == 0
        || rows == 0
        || !input.len().is_multiple_of(width)
        || input.len() / width != rows
        || expected != Some(input.len())
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || scale.shape() != [width]
    {
        return Err(Gemma4Error::new(
            "CPU RMS",
            "input rows, width, epsilon or scale width is invalid",
        ));
    }
    let mut output = try_vector(input.len(), 0_u16)?;
    for row in 0..rows {
        let x = &input[row * width..(row + 1) * width];
        let mut energy = 0.0_f32;
        for &value in x {
            let value = f32::from_bits(u32::from(value) << 16);
            energy += value * value;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "validated width is a small positive model dimension"
        )]
        let mean = energy / width as f32;
        let inverse = (mean + epsilon).powf(-0.5);
        for (column, &value) in x.iter().enumerate() {
            let value = f32::from_bits(u32::from(value) << 16) * inverse;
            let gamma = f32::from_bits(u32::from(scale.values()[column]) << 16);
            output[row * width + column] = bf16(value * gamma);
        }
    }
    Ok(output)
}
