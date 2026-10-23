//! One input RMS feeding the fixed query and optional key/value projection chains.

use super::{MetalLinearShape, MetalProjectionNorm, ProjectionGeometry};
use crate::{MetalBf16Buffer, MetalBf16Error, MetalBf16Executor};

/// One BF16 projection followed by grouped normalization and optional rotation.
#[derive(Clone, Copy)]
pub struct MetalNormalizedProjection<'a> {
    pub weight: &'a MetalBf16Buffer,
    pub norm: MetalProjectionNorm<'a>,
}

impl<'a> MetalNormalizedProjection<'a> {
    fn inferred_geometry(
        self,
        shape: MetalLinearShape,
        input: usize,
        block_rows: bool,
    ) -> Result<ProjectionGeometry<'a>, MetalBf16Error> {
        // Input RMS preflight establishes positive inner; linear checks exact divisibility.
        let columns = self.weight.elements / shape.inner;
        ProjectionGeometry::new(
            MetalLinearShape { columns, ..shape },
            input,
            self.weight.elements,
            self.norm,
            block_rows,
        )
    }
}

/// Completed attention inputs; shared-KV consumers only produce a query.
pub struct MetalAttentionProjections {
    pub query: MetalBf16Buffer,
    pub key_value: Option<(MetalBf16Buffer, MetalBf16Buffer)>,
}

impl MetalBf16Executor {
    /// Normalize whole input rows once, then prepare query and optional key/value.
    ///
    /// The input width is explicit in `shape.inner`; key/value column counts come
    /// from their weights. Every branch is validated before any allocation, and
    /// all BF16 intermediates stay private through the single terminal wait.
    /// # Errors
    /// Rejects invalid geometry/scales/factors, allocation or GPU execution failure.
    pub fn rms_attention_projections(
        &self,
        input: &MetalBf16Buffer,
        shape: MetalLinearShape,
        epsilon: f32,
        input_scale: Option<&MetalBf16Buffer>,
        query: MetalNormalizedProjection<'_>,
        key_value: Option<(MetalNormalizedProjection<'_>, MetalNormalizedProjection<'_>)>,
    ) -> Result<MetalAttentionProjections, MetalBf16Error> {
        let rms = super::RmsGeometry::new(
            input.elements,
            shape.rows,
            shape.inner,
            epsilon,
            input_scale,
        )?;
        let query_geometry = ProjectionGeometry::new(
            shape,
            input.elements,
            query.weight.elements,
            query.norm,
            self.block_linear_rows,
        )?;
        let key_value_geometry = key_value
            .map(|(key, value)| {
                let key_geometry =
                    key.inferred_geometry(shape, input.elements, self.block_linear_rows)?;
                let value_geometry =
                    value.inferred_geometry(shape, input.elements, self.block_linear_rows)?;
                if key_geometry.linear.columns != value_geometry.linear.columns {
                    return Err(MetalBf16Error::Shape(
                        "attention key/value projection columns disagree".into(),
                    ));
                }
                Ok::<_, MetalBf16Error>((key_geometry, value_geometry, key.weight, value.weight))
            })
            .transpose()?;
        let normalized = self.allocate_shared(input.elements * 2)?;
        let query_buffers = query_geometry.allocate(self)?;
        let key_value_buffers = key_value_geometry
            .map(|(key, value, key_weight, value_weight)| {
                Ok::<_, MetalBf16Error>((
                    key.allocate(self)?,
                    value.allocate(self)?,
                    key_weight,
                    value_weight,
                ))
            })
            .transpose()?;
        let command = self.command()?;
        let encoder = Self::compute_encoder(&command)?;
        self.encode_rms(&encoder, &input.buffer, &normalized, input_scale, &rms);
        query_buffers.encode(self, &encoder, &normalized, &query.weight.buffer);
        if let Some((key, value, key_weight, value_weight)) = &key_value_buffers {
            key.encode(self, &encoder, &normalized, &key_weight.buffer);
            value.encode(self, &encoder, &normalized, &value_weight.buffer);
        }
        encoder.end_encoding();
        Self::submit(&command, "attention input projections")?;
        Ok(MetalAttentionProjections {
            query: query_buffers.into_output(),
            key_value: key_value_buffers
                .map(|(key, value, _, _)| (key.into_output(), value.into_output())),
        })
    }
}
