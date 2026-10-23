//! Fixed projection chains using the existing linear and RMS kernels.

use super::rotary::RotaryGeometry;
use super::{MetalBf16Buffer, MetalBf16Error, MetalBf16Executor, MetalRotaryFactors, THREADS};
use metal::{Buffer, BufferRef, ComputeCommandEncoderRef, MTLSize};

mod attention;
pub use attention::{MetalAttentionProjections, MetalNormalizedProjection};

/// Row-major `input[rows, inner] × weight[columns, inner]ᵀ` geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalLinearShape {
    pub rows: usize,
    pub columns: usize,
    pub inner: usize,
}

/// Normalization and optional per-head rotation after a BF16 projection.
#[derive(Clone, Copy)]
pub struct MetalProjectionNorm<'a> {
    pub width: usize,
    pub epsilon: f32,
    pub scale: Option<&'a MetalBf16Buffer>,
    /// When present, each normalization group is one rotary head.
    pub rotary: Option<MetalRotaryFactors<'a>>,
}

struct LinearGeometry {
    columns: u32,
    inner: u32,
    total: u32,
    row_block: u32,
}

impl LinearGeometry {
    fn new(
        shape: MetalLinearShape,
        input: usize,
        weight: usize,
        block_rows: bool,
    ) -> Result<Self, MetalBf16Error> {
        let MetalLinearShape {
            rows,
            columns,
            inner,
        } = shape;
        let input_length = rows
            .checked_mul(inner)
            .ok_or_else(|| MetalBf16Error::Shape("input dimensions overflow".into()))?;
        let weight_length = columns
            .checked_mul(inner)
            .ok_or_else(|| MetalBf16Error::Shape("weight dimensions overflow".into()))?;
        let output_length = rows
            .checked_mul(columns)
            .ok_or_else(|| MetalBf16Error::Shape("output dimensions overflow".into()))?;
        if rows == 0
            || columns == 0
            || inner == 0
            || input != input_length
            || weight != weight_length
        {
            return Err(MetalBf16Error::Shape(
                "linear buffers disagree with the requested shape".into(),
            ));
        }
        if u32::try_from(input_length).is_err() || u32::try_from(weight_length).is_err() {
            return Err(MetalBf16Error::Shape(
                "linear input or weight exceeds u32 elements".into(),
            ));
        }
        Ok(Self {
            columns: u32::try_from(columns)
                .map_err(|_| MetalBf16Error::Shape("columns exceed u32".into()))?,
            inner: u32::try_from(inner)
                .map_err(|_| MetalBf16Error::Shape("inner exceeds u32".into()))?,
            total: u32::try_from(output_length)
                .map_err(|_| MetalBf16Error::Shape("output exceeds u32 elements".into()))?,
            // Block only the independently qualified M3 Max chunk scope.
            row_block: if block_rows
                && (8..=64).contains(&rows)
                && rows.is_multiple_of(2)
                && inner.is_multiple_of(4)
            {
                if rows >= 32 && rows.is_multiple_of(4) {
                    4
                } else {
                    2
                }
            } else {
                1
            },
        })
    }

    fn elements(&self) -> usize {
        self.total as usize
    }
}

struct RmsGeometry {
    rows: u64,
    width: u32,
    epsilon: f32,
    has_scale: u32,
}

impl RmsGeometry {
    fn new(
        elements: usize,
        rows: usize,
        width: usize,
        epsilon: f32,
        scale: Option<&MetalBf16Buffer>,
    ) -> Result<Self, MetalBf16Error> {
        if rows == 0
            || width == 0
            || rows.checked_mul(width) != Some(elements)
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || scale.is_some_and(|scale| scale.elements != width)
        {
            return Err(MetalBf16Error::Shape(
                "RMS buffers disagree with the requested shape".into(),
            ));
        }
        // Both kernels calculate element offsets with uint, not just dimensions.
        if u32::try_from(elements).is_err() {
            return Err(MetalBf16Error::Shape(
                "RMS input exceeds u32 elements".into(),
            ));
        }
        Ok(Self {
            rows: rows as u64,
            width: u32::try_from(width)
                .map_err(|_| MetalBf16Error::Shape("width exceeds u32".into()))?,
            epsilon,
            has_scale: u32::from(scale.is_some()),
        })
    }
}

/// Geometry-only preflight, shared by standalone and fixed attention chains.
pub(super) struct ProjectionGeometry<'a> {
    linear: LinearGeometry,
    rms: RmsGeometry,
    scale: Option<&'a MetalBf16Buffer>,
    rotation: Option<(MetalRotaryFactors<'a>, RotaryGeometry)>,
}

impl<'a> ProjectionGeometry<'a> {
    pub(super) fn new(
        shape: MetalLinearShape,
        input: usize,
        weight: usize,
        norm: MetalProjectionNorm<'a>,
        block_rows: bool,
    ) -> Result<Self, MetalBf16Error> {
        let linear = LinearGeometry::new(shape, input, weight, block_rows)?;
        let rows = linear
            .elements()
            .checked_div(norm.width)
            .ok_or_else(|| MetalBf16Error::Shape("RMS width must be positive".into()))?;
        let rms = RmsGeometry::new(
            linear.elements(),
            rows,
            norm.width,
            norm.epsilon,
            norm.scale,
        )?;
        let rotation = norm
            .rotary
            .map(|factors| {
                RotaryGeometry::new(linear.elements(), shape.rows, norm.width, factors)
                    .map(|geometry| (factors, geometry))
            })
            .transpose()?;
        Ok(Self {
            linear,
            rms,
            scale: norm.scale,
            rotation,
        })
    }

    pub(super) fn allocate(
        self,
        executor: &MetalBf16Executor,
    ) -> Result<ProjectionBuffers<'a>, MetalBf16Error> {
        let projected = executor.allocate_shared(self.linear.elements() * 2)?;
        let normalized = executor.allocate_shared(self.linear.elements() * 2)?;
        let rotation = self
            .rotation
            .map(|(factors, geometry)| {
                executor
                    .allocate_shared(self.linear.elements() * 2)
                    .map(|output| (factors, geometry, output))
            })
            .transpose()?;
        Ok(ProjectionBuffers {
            linear: self.linear,
            rms: self.rms,
            scale: self.scale,
            projected,
            normalized,
            rotation,
        })
    }
}

/// Every intermediate remains owned until the enclosing command's terminal wait.
pub(super) struct ProjectionBuffers<'a> {
    linear: LinearGeometry,
    rms: RmsGeometry,
    scale: Option<&'a MetalBf16Buffer>,
    projected: Buffer,
    normalized: Buffer,
    rotation: Option<(MetalRotaryFactors<'a>, RotaryGeometry, Buffer)>,
}

impl ProjectionBuffers<'_> {
    pub(super) fn encode(
        &self,
        executor: &MetalBf16Executor,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        weight: &BufferRef,
    ) {
        executor.encode_linear(encoder, input, weight, &self.projected, &self.linear);
        executor.encode_rms(
            encoder,
            &self.projected,
            &self.normalized,
            self.scale,
            &self.rms,
        );
        if let Some((factors, geometry, output)) = &self.rotation {
            executor.encode_rotary(encoder, &self.normalized, output, *factors, geometry);
        }
    }

    pub(super) fn into_output(self) -> MetalBf16Buffer {
        MetalBf16Buffer {
            buffer: self
                .rotation
                .map_or(self.normalized, |(_, _, output)| output),
            elements: self.linear.elements(),
        }
    }
}

impl MetalBf16Executor {
    /// Row-major BF16 projection, with source-order FP32 accumulation per output.
    /// # Errors
    /// Rejects invalid shapes, allocation failure or unsuccessful GPU execution.
    pub fn linear(
        &self,
        input: &MetalBf16Buffer,
        weight: &MetalBf16Buffer,
        shape: MetalLinearShape,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        let geometry = LinearGeometry::new(
            shape,
            input.elements,
            weight.elements,
            self.block_linear_rows,
        )?;
        let output = self.allocate_shared(geometry.elements() * 2)?;
        let command = self.command()?;
        let encoder = Self::compute_encoder(&command)?;
        self.encode_linear(&encoder, &input.buffer, &weight.buffer, &output, &geometry);
        encoder.end_encoding();
        Self::submit(&command, "linear")?;
        Ok(MetalBf16Buffer {
            buffer: output,
            elements: geometry.elements(),
        })
    }

    /// Grouped RMS over rows of `width`, with an optional direct BF16 scale.
    /// # Errors
    /// Rejects invalid shapes/epsilon, allocation failure or unsuccessful execution.
    pub fn rms_norm(
        &self,
        input: &MetalBf16Buffer,
        rows: usize,
        width: usize,
        epsilon: f32,
        scale: Option<&MetalBf16Buffer>,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        let geometry = RmsGeometry::new(input.elements, rows, width, epsilon, scale)?;
        let output = self.allocate_shared(input.elements * 2)?;
        let command = self.command()?;
        let encoder = Self::compute_encoder(&command)?;
        self.encode_rms(&encoder, &input.buffer, &output, scale, &geometry);
        encoder.end_encoding();
        Self::submit(&command, "RMS")?;
        Ok(MetalBf16Buffer {
            buffer: output,
            elements: input.elements,
        })
    }

    /// Project, normalize contiguous groups and optionally rotate each head.
    ///
    /// Existing kernels share a serial command; their BF16 boundaries remain.
    /// All private intermediates and inputs stay owned through terminal completion.
    /// # Errors
    /// Validates the complete chain before allocation/submission. Rejects invalid
    /// shapes/epsilon/scale/factors, allocation failure or unsuccessful execution.
    pub fn project_normalized(
        &self,
        input: &MetalBf16Buffer,
        weight: &MetalBf16Buffer,
        shape: MetalLinearShape,
        norm: MetalProjectionNorm<'_>,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        let buffers = ProjectionGeometry::new(
            shape,
            input.elements,
            weight.elements,
            norm,
            self.block_linear_rows,
        )?
        .allocate(self)?;
        let command = self.command()?;
        let encoder = Self::compute_encoder(&command)?;
        buffers.encode(self, &encoder, &input.buffer, &weight.buffer);
        encoder.end_encoding();
        Self::submit(&command, "normalized projection")?;
        Ok(buffers.into_output())
    }

    /// Normalize each input row, then project through two same-shaped weights.
    ///
    /// The normalized BF16 input remains private. Both projections use the same
    /// kernels and rounding boundaries as three standalone operations; neither
    /// output is published until the entire serial command completes successfully.
    /// # Errors
    /// Validates both weights and RMS before allocation/submission. Rejects invalid
    /// shapes/epsilon/scale, allocation failure or unsuccessful GPU execution.
    pub fn rms_linear_pair(
        &self,
        input: &MetalBf16Buffer,
        left: &MetalBf16Buffer,
        right: &MetalBf16Buffer,
        shape: MetalLinearShape,
        epsilon: f32,
        scale: Option<&MetalBf16Buffer>,
    ) -> Result<(MetalBf16Buffer, MetalBf16Buffer), MetalBf16Error> {
        let left_geometry =
            LinearGeometry::new(shape, input.elements, left.elements, self.block_linear_rows)?;
        let right_geometry = LinearGeometry::new(
            shape,
            input.elements,
            right.elements,
            self.block_linear_rows,
        )?;
        let rms = RmsGeometry::new(input.elements, shape.rows, shape.inner, epsilon, scale)?;
        let normalized = self.allocate_shared(input.elements * 2)?;
        let left_output = self.allocate_shared(left_geometry.elements() * 2)?;
        let right_output = self.allocate_shared(right_geometry.elements() * 2)?;
        let command = self.command()?;
        let encoder = Self::compute_encoder(&command)?;
        self.encode_rms(&encoder, &input.buffer, &normalized, scale, &rms);
        self.encode_linear(
            &encoder,
            &normalized,
            &left.buffer,
            &left_output,
            &left_geometry,
        );
        self.encode_linear(
            &encoder,
            &normalized,
            &right.buffer,
            &right_output,
            &right_geometry,
        );
        encoder.end_encoding();
        Self::submit(&command, "RMS/linear pair")?;
        Ok((
            MetalBf16Buffer {
                buffer: left_output,
                elements: left_geometry.elements(),
            },
            MetalBf16Buffer {
                buffer: right_output,
                elements: right_geometry.elements(),
            },
        ))
    }

    fn encode_linear(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        weight: &BufferRef,
        output: &BufferRef,
        geometry: &LinearGeometry,
    ) {
        encoder.set_compute_pipeline_state(if geometry.row_block == 4 {
            &self.linear_rows4
        } else {
            &self.linear
        });
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, 4, (&raw const geometry.columns).cast());
        encoder.set_bytes(4, 4, (&raw const geometry.inner).cast());
        encoder.set_bytes(5, 4, (&raw const geometry.total).cast());
        let paired = u32::from(geometry.row_block == 2);
        encoder.set_bytes(6, 4, (&raw const paired).cast());
        let grid = u64::from(geometry.total / geometry.row_block);
        encoder.dispatch_threads(
            MTLSize::new(grid, 1, 1),
            MTLSize::new(grid.min(THREADS), 1, 1),
        );
    }

    fn encode_rms(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        output: &BufferRef,
        scale: Option<&MetalBf16Buffer>,
        geometry: &RmsGeometry,
    ) {
        encoder.set_compute_pipeline_state(&self.rms);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(scale.map_or(input, |scale| &scale.buffer)), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, 4, (&raw const geometry.width).cast());
        encoder.set_bytes(4, 4, (&raw const geometry.epsilon).cast());
        encoder.set_bytes(5, 4, (&raw const geometry.has_scale).cast());
        encoder.dispatch_thread_groups(
            MTLSize::new(geometry.rows, 1, 1),
            MTLSize::new(THREADS, 1, 1),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{LinearGeometry, MetalLinearShape};

    #[test]
    fn row_blocking_stays_inside_the_qualified_device_and_shape_scope() {
        for (rows, inner, eligible_device, block) in [
            (6, 4, true, 1),
            (8, 4, true, 2),
            (30, 4, true, 2),
            (31, 4, true, 1),
            (32, 4, true, 4),
            (33, 4, true, 1),
            (34, 4, true, 2),
            (64, 4, true, 4),
            (65, 4, true, 1),
            (66, 4, true, 1),
            (32, 7, true, 1),
            (32, 4, false, 1),
            (64, 4, false, 1),
        ] {
            let shape = MetalLinearShape {
                rows,
                columns: 3,
                inner,
            };
            let geometry =
                LinearGeometry::new(shape, rows * inner, 3 * inner, eligible_device).unwrap();
            assert_eq!(
                geometry.row_block, block,
                "{shape:?}, eligible={eligible_device}"
            );
        }
    }
}
