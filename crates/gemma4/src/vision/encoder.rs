//! Synchronous CPU vision component; application media admission remains separate.
#[cfg(test)]
#[path = "encoder/qualification.rs"]
mod qualification;

use super::ops::{Grid, axial, load, pool};
use super::{
    Gemma4VisionConfig, Gemma4VisionCpuWeights, VisionLinear, VisionLinearPart, VisionNorm,
    VisionParameterId,
};
use crate::Gemma4Error;
use crate::cpu::{
    AttentionConfig, add, bf16, gelu_tanh, multiply, project, rms_norm, rms_norm_scale_free,
    try_vector, vision_attention,
};

/// Processor output for one image, including physical trailing padding. Pixels
/// are rescaled FP32 HWC-within-patch values, not encoded media or normalized RGB.
#[derive(Clone, Copy)]
pub struct VisionImage<'a> {
    pub pixels: &'a [f32],
    pub positions: &'a [[i64; 2]],
}

/// Streaming diagnostics avoid retaining all encoder intermediates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionStage {
    PatchEmbedding,
    Layer(usize),
    Pooled,
    ProjectionNorm,
    Projected,
}

/// Model-local BF16 CPU encoder/projector. This is not a VLM request interface,
/// feature cache, decoder continuation or automatically selected device path.
pub struct Gemma4CpuVisionEncoder {
    weights: Gemma4VisionCpuWeights,
}
impl Gemma4CpuVisionEncoder {
    #[must_use]
    pub const fn new(weights: Gemma4VisionCpuWeights) -> Self {
        Self { weights }
    }
    #[must_use]
    pub fn config(&self) -> &Gemma4VisionConfig {
        self.weights.config()
    }
    /// Encode one canonical processed image. Returns row-major valid projected
    /// features `[valid_patches / pooling_area, text_hidden_size]`.
    /// # Errors
    /// Rejects invalid image geometry/pixels, allocation failure or nonfinite output.
    pub fn encode(&self, image: VisionImage<'_>) -> Result<Vec<u16>, Gemma4Error> {
        self.encode_observed(image, |_, _| Ok(()))
    }
    /// As [`Self::encode`], exposing borrowed stage results for reference capture.
    /// Patch/layer stages include physical padding; pooled/projected stages do not.
    /// # Errors
    /// As [`Self::encode`], or an observer error. No mutable execution state remains.
    pub fn encode_observed(
        &self,
        image: VisionImage<'_>,
        mut observe: impl FnMut(VisionStage, &[u16]) -> Result<(), Gemma4Error>,
    ) -> Result<Vec<u16>, Gemma4Error> {
        let c = self.config();
        let grid = Grid::image(image, c)?;
        let mut pixels = try_vector(image.pixels.len(), 0_u16)?;
        for (slot, &pixel) in pixels.iter_mut().zip(image.pixels) {
            *slot = bf16(2.0 * (pixel - 0.5));
        }
        let mut hidden = project(
            &pixels,
            grid.rows,
            self.weights.get(VisionParameterId::PatchProjection)?,
        )?;
        let table = self
            .weights
            .get(VisionParameterId::PositionEmbedding)?
            .values();
        for row in 0..grid.valid {
            let [x, y] = image.positions[row];
            let x = usize::try_from(x)
                .map_err(|_| Gemma4Error::new("vision position", "negative x"))?;
            let y = usize::try_from(y)
                .map_err(|_| Gemma4Error::new("vision position", "negative y"))?;
            for column in 0..c.hidden {
                let position = bf16(
                    load(table[x * c.hidden + column])
                        + load(table[(c.positions + y) * c.hidden + column]),
                );
                hidden[row * c.hidden + column] =
                    bf16(load(hidden[row * c.hidden + column]) + load(position));
            }
        }
        publish(&mut observe, VisionStage::PatchEmbedding, &hidden)?;
        for layer in 0..c.layers {
            hidden = self.layer(&hidden, image.positions, &grid, layer)?;
            publish(&mut observe, VisionStage::Layer(layer), &hidden)?;
        }
        let pooled = pool(&hidden, &grid, c)?;
        publish(&mut observe, VisionStage::Pooled, &pooled)?;
        let normalized =
            rms_norm_scale_free(&pooled, grid.valid / (c.pool * c.pool), c.hidden, c.epsilon)?;
        publish(&mut observe, VisionStage::ProjectionNorm, &normalized)?;
        let projected = project(
            &normalized,
            grid.valid / (c.pool * c.pool),
            self.weights.get(VisionParameterId::Projection)?,
        )?;
        publish(&mut observe, VisionStage::Projected, &projected)?;
        Ok(projected)
    }
    fn norm(
        &self,
        input: &[u16],
        rows: usize,
        layer: usize,
        norm: VisionNorm,
    ) -> Result<Vec<u16>, Gemma4Error> {
        let weight = self.weights.get(VisionParameterId::Norm { layer, norm })?;
        rms_norm(
            input,
            rows,
            weight.shape()[0],
            self.config().epsilon,
            weight,
        )
    }
    fn linear(
        &self,
        input: &[u16],
        rows: usize,
        layer: usize,
        linear: VisionLinear,
    ) -> Result<Vec<u16>, Gemma4Error> {
        let weight = self.weights.get(VisionParameterId::Linear {
            layer,
            linear,
            part: VisionLinearPart::Weight,
        })?;
        if !self.config().clipped {
            return project(input, rows, weight);
        }
        let [low, high, out_low, out_high] = self.weights.bounds(layer, linear)?;
        let mut clipped = try_vector(input.len(), 0_u16)?;
        for (slot, &value) in clipped.iter_mut().zip(input) {
            *slot = bf16(load(value).clamp(low, high));
        }
        let mut output = project(&clipped, rows, weight)?;
        for value in &mut output {
            *value = bf16(load(*value).clamp(out_low, out_high));
        }
        Ok(output)
    }
    fn layer(
        &self,
        hidden: &[u16],
        positions: &[[i64; 2]],
        grid: &Grid,
        layer: usize,
    ) -> Result<Vec<u16>, Gemma4Error> {
        let c = self.config();
        let normalized = self.norm(hidden, grid.rows, layer, VisionNorm::Input)?;
        let query = self.linear(&normalized, grid.rows, layer, VisionLinear::Query)?;
        let key = self.linear(&normalized, grid.rows, layer, VisionLinear::Key)?;
        let value = self.linear(&normalized, grid.rows, layer, VisionLinear::Value)?;
        let query = self.norm(&query, grid.rows * c.heads, layer, VisionNorm::Query)?;
        let key = self.norm(&key, grid.rows * c.kv_heads, layer, VisionNorm::Key)?;
        let value = rms_norm_scale_free(&value, grid.rows * c.kv_heads, c.head_dim, c.epsilon)?;
        let query = axial(&query, positions, c.heads, c)?;
        let key = axial(&key, positions, c.kv_heads, c)?;
        let attention = vision_attention(
            &query,
            &key,
            &value,
            AttentionConfig {
                heads: c.heads,
                kv_heads: c.kv_heads,
                width: c.head_dim,
                window: None,
            },
            grid.valid,
        )?;
        let attention = self.linear(&attention, grid.rows, layer, VisionLinear::AttentionOutput)?;
        let attention = self.norm(&attention, grid.rows, layer, VisionNorm::PostAttention)?;
        let hidden = add(hidden, &attention)?;
        let input = self.norm(&hidden, grid.rows, layer, VisionNorm::PreFeedforward)?;
        let gate = gelu_tanh(&self.linear(&input, grid.rows, layer, VisionLinear::Gate)?)?;
        let up = self.linear(&input, grid.rows, layer, VisionLinear::Up)?;
        let down = self.linear(&multiply(&gate, &up)?, grid.rows, layer, VisionLinear::Down)?;
        let down = self.norm(&down, grid.rows, layer, VisionNorm::PostFeedforward)?;
        add(&hidden, &down)
    }
}
fn publish(
    observe: &mut impl FnMut(VisionStage, &[u16]) -> Result<(), Gemma4Error>,
    stage: VisionStage,
    values: &[u16],
) -> Result<(), Gemma4Error> {
    if values.iter().any(|&v| !load(v).is_finite()) {
        return Err(Gemma4Error::new(
            "vision output",
            format!("nonfinite {stage:?}"),
        ));
    }
    observe(stage, values)
}
