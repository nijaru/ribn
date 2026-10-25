//! Canonical patch geometry, axial rotation and spatial pooling.
use super::{Gemma4VisionConfig, VisionImage};
use crate::Gemma4Error;
use crate::cpu::{bf16, try_vector};

pub(super) fn load(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}
pub(super) struct Grid {
    pub rows: usize,
    pub valid: usize,
    pub width: usize,
}
impl Grid {
    pub fn positions(positions: &[[i64; 2]], c: &Gemma4VisionConfig) -> Result<Self, Gemma4Error> {
        let rows = positions.len();
        if rows == 0 || !rows.is_multiple_of(c.pool * c.pool) {
            return Err(invalid(
                "physical patches must be a nonempty multiple of pooling area",
            ));
        }
        let valid = positions
            .iter()
            .position(|&p| p == [-1, -1])
            .unwrap_or(rows);
        let mut width = 0;
        for &[x, y] in &positions[..valid] {
            let x = usize::try_from(x).map_err(|_| invalid("negative x position"))?;
            let y = usize::try_from(y).map_err(|_| invalid("negative y position"))?;
            if x >= c.positions || y >= c.positions {
                return Err(invalid("position exceeds learned table"));
            }
            width = width.max(x + 1);
        }
        if valid == 0
            || !valid.is_multiple_of(width)
            || !width.is_multiple_of(c.pool)
            || !(valid / width).is_multiple_of(c.pool)
        {
            return Err(invalid(
                "valid grid must be a complete pool-aligned rectangle",
            ));
        }
        for (i, &p) in positions.iter().enumerate() {
            let expected = if i < valid {
                [
                    i64::try_from(i % width).map_err(|_| invalid("x overflow"))?,
                    i64::try_from(i / width).map_err(|_| invalid("y overflow"))?,
                ]
            } else {
                [-1, -1]
            };
            if p != expected {
                return Err(invalid(
                    "positions must be row-major xy with trailing (-1,-1) padding",
                ));
            }
        }
        rows.checked_mul(c.hidden)
            .and_then(|_| rows.checked_mul(c.heads * c.head_dim))
            .ok_or_else(|| invalid("activation geometry overflows"))?;
        Ok(Self { rows, valid, width })
    }
    pub fn image(image: VisionImage<'_>, c: &Gemma4VisionConfig) -> Result<Self, Gemma4Error> {
        let grid = Self::positions(image.positions, c)?;
        let patch = 3 * c.patch * c.patch;
        if grid.rows.checked_mul(patch) != Some(image.pixels.len()) {
            return Err(invalid(
                "pixel matrix must match [physical patches, 3*patch_size^2]",
            ));
        }
        if image
            .pixels
            .iter()
            .any(|&x| !x.is_finite() || !(0.0..=1.0).contains(&x))
            || image.pixels[grid.valid * patch..].iter().any(|&x| x != 0.0)
        {
            return Err(invalid(
                "expected finite rescaled RGB patches and zero pixel padding",
            ));
        }
        Ok(grid)
    }
}
fn invalid(reason: &str) -> Gemma4Error {
    Gemma4Error::new("vision input", reason)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "validated model head dimensions and learned-table coordinates"
)]
pub(super) fn axial(
    input: &[u16],
    positions: &[[i64; 2]],
    heads: usize,
    c: &Gemma4VisionConfig,
) -> Result<Vec<u16>, Gemma4Error> {
    let mut output = try_vector(input.len(), 0_u16)?;
    let quarter = c.head_dim / 4;
    for (row, position) in positions.iter().enumerate() {
        for (axis, &coordinate) in position.iter().enumerate() {
            for i in 0..quarter {
                // HF from_pretrained keeps these derived frequencies FP32 even
                // with BF16 weights; casting a whole module to BF16 does not.
                let inverse = 1.0 / c.theta.powf((2 * i) as f32 / (2 * quarter) as f32);
                let angle = coordinate as f32 * inverse;
                let (cos, sin) = (load(bf16(angle.cos())), load(bf16(angle.sin())));
                for head in 0..heads {
                    let first =
                        row * heads * c.head_dim + head * c.head_dim + axis * 2 * quarter + i;
                    let second = first + quarter;
                    let (x, y) = (load(input[first]), load(input[second]));
                    output[first] = bf16(load(bf16(x * cos)) + load(bf16(-y * sin)));
                    output[second] = bf16(load(bf16(y * cos)) + load(bf16(x * sin)));
                }
            }
        }
    }
    Ok(output)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "positive validated model dimensions"
)]
pub(super) fn pool(
    hidden: &[u16],
    grid: &Grid,
    c: &Gemma4VisionConfig,
) -> Result<Vec<u16>, Gemma4Error> {
    let area = c.pool * c.pool;
    let mut sums = try_vector(grid.valid / area * c.hidden, 0.0_f32)?;
    let factor = 1.0 / area as f32;
    for row in 0..grid.valid {
        let cell =
            (row % grid.width) / c.pool + (grid.width / c.pool) * ((row / grid.width) / c.pool);
        for column in 0..c.hidden {
            sums[cell * c.hidden + column] += factor * load(hidden[row * c.hidden + column]);
        }
    }
    let scale = (c.hidden as f32).sqrt();
    let mut output = try_vector(sums.len(), 0_u16)?;
    for (slot, sum) in output.iter_mut().zip(sums) {
        *slot = bf16(load(bf16(sum)) * scale);
    }
    Ok(output)
}
