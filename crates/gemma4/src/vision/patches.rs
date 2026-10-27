//! RGB patch packing only; resize, decoded-media admission and feature handoff are separate.
use super::{Gemma4VisionConfig, VisionImage};
use crate::{Gemma4Error, cpu::try_vector};

/// Owned canonical processor output for one already-resized RGB8 image.
///
/// This is a host component, not an application media request or reservation.
/// It neither resizes nor decodes images, applies EXIF/color conversion, or
/// normalizes pixels to the encoder's internal [-1,1] range.
pub struct Gemma4ImagePatches {
    pixels: Vec<f32>,
    positions: Vec<[i64; 2]>,
    soft_tokens: usize,
}

impl Gemma4ImagePatches {
    /// Pack tightly interleaved RGB8 pixels in row-major HWC order. Dimensions
    /// must already be aligned to `patch_size * pooling_kernel_size`.
    ///
    /// Pixel values are rescaled to FP32 [0,1], with RGB innermost within each
    /// row-major patch. Both arrays include the selected budget's trailing padding.
    /// The borrowed RGB input is not retained. Allocations are fallible and bounded
    /// by this image's budget and validated model geometry, not a shared byte pool.
    /// # Errors
    /// Rejects unsupported soft-token budgets, invalid dimensions/byte counts,
    /// grids exceeding the budget or learned position table, and allocation failure.
    pub fn from_resized_rgb(
        config: &Gemma4VisionConfig,
        width: usize,
        height: usize,
        rgb: &[u8],
        max_soft_tokens: usize,
    ) -> Result<Self, Gemma4Error> {
        if !matches!(max_soft_tokens, 70 | 140 | 280 | 560 | 1120) {
            return Err(invalid(
                "soft-token budget must be 70, 140, 280, 560 or 1120",
            ));
        }
        let side = config.patch.checked_mul(config.pool).ok_or_else(overflow)?;
        if width == 0 || height == 0 || !width.is_multiple_of(side) || !height.is_multiple_of(side)
        {
            return Err(invalid(
                "resized RGB dimensions must be positive and pooling/patch aligned",
            ));
        }
        let bytes = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(overflow)?;
        if rgb.len() != bytes {
            return Err(invalid("expected exactly width * height * 3 RGB bytes"));
        }
        let columns = width / config.patch;
        let rows = height / config.patch;
        let valid = columns.checked_mul(rows).ok_or_else(overflow)?;
        let area = config.pool.checked_mul(config.pool).ok_or_else(overflow)?;
        let physical = max_soft_tokens.checked_mul(area).ok_or_else(overflow)?;
        if valid > physical {
            return Err(invalid("resized image exceeds selected patch budget"));
        }
        if columns > config.positions || rows > config.positions {
            return Err(invalid("resized image exceeds learned position table"));
        }
        let patch_values = config
            .patch
            .checked_mul(config.patch)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(overflow)?;
        let length = physical.checked_mul(patch_values).ok_or_else(overflow)?;
        // Also validate conversion before allocation; coordinates may never be
        // silently truncated on a target with a wider usize.
        i64::try_from(columns - 1)
            .and_then(|_| i64::try_from(rows - 1))
            .map_err(|_| overflow())?;
        let mut pixels = try_vector(length, 0.0_f32)?;
        let mut positions = try_vector(physical, [-1_i64, -1])?;
        for (index, position) in positions[..valid].iter_mut().enumerate() {
            let x = index % columns;
            let y = index / columns;
            *position = [
                i64::try_from(x).map_err(|_| overflow())?,
                i64::try_from(y).map_err(|_| overflow())?,
            ];
            for dy in 0..config.patch {
                let source = ((y * config.patch + dy) * width + x * config.patch) * 3;
                let target = index * patch_values + dy * config.patch * 3;
                for (out, &byte) in pixels[target..target + config.patch * 3]
                    .iter_mut()
                    .zip(&rgb[source..source + config.patch * 3])
                {
                    // Stock's uint8 -> float32 followed by scalar multiplication;
                    // division by 255 in the loop is not the same FP32 policy.
                    *out = f32::from(byte) * (1.0_f32 / 255.0);
                }
            }
        }
        Ok(Self {
            pixels,
            positions,
            soft_tokens: valid / area,
        })
    }

    /// Borrow the exact existing encoder input; no patch or position copy.
    #[must_use]
    pub fn as_image(&self) -> VisionImage<'_> {
        VisionImage {
            pixels: &self.pixels,
            positions: &self.positions,
        }
    }

    /// Number of valid pooled features, excluding physical padding.
    #[must_use]
    pub const fn soft_tokens(&self) -> usize {
        self.soft_tokens
    }
}

fn invalid(message: &str) -> Gemma4Error {
    Gemma4Error::new("vision RGB patches", message)
}
fn overflow() -> Gemma4Error {
    invalid("RGB patch geometry overflows")
}
