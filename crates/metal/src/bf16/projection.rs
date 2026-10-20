//! Fixed projection chains using the existing linear and RMS kernels.

use super::{MetalBf16Buffer, MetalBf16Error, MetalBf16Executor, THREADS};
use metal::{BufferRef, ComputeCommandEncoderRef, MTLSize};

/// Row-major `input[rows, inner] × weight[columns, inner]ᵀ` geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalLinearShape {
    pub rows: usize,
    pub columns: usize,
    pub inner: usize,
}

struct LinearGeometry {
    columns: u32,
    inner: u32,
    total: u32,
    paired: u32,
}

impl LinearGeometry {
    fn new(
        shape: MetalLinearShape,
        input: usize,
        weight: usize,
        pair_rows: bool,
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
            // Keep the independently qualified M3 Max selection unchanged.
            paired: u32::from(
                pair_rows
                    && (8..=64).contains(&rows)
                    && rows.is_multiple_of(2)
                    && inner.is_multiple_of(4),
            ),
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
            self.pair_linear_rows,
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

    /// Project, then normalize contiguous groups of `width` in one serial command.
    ///
    /// The private BF16 projection preserves the standalone rounding boundary.
    /// Both output-sized buffers stay owned until terminal GPU completion; only
    /// a successfully initialized normalized output is published.
    /// # Errors
    /// Validates both operations before allocation/submission. Rejects invalid
    /// shapes/epsilon/scale, allocation failure or unsuccessful GPU execution.
    pub fn linear_rms(
        &self,
        input: &MetalBf16Buffer,
        weight: &MetalBf16Buffer,
        shape: MetalLinearShape,
        width: usize,
        epsilon: f32,
        scale: Option<&MetalBf16Buffer>,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        let linear = LinearGeometry::new(
            shape,
            input.elements,
            weight.elements,
            self.pair_linear_rows,
        )?;
        let rows = linear
            .elements()
            .checked_div(width)
            .ok_or_else(|| MetalBf16Error::Shape("RMS width must be positive".into()))?;
        let rms = RmsGeometry::new(linear.elements(), rows, width, epsilon, scale)?;
        let projected = self.allocate_shared(linear.elements() * 2)?;
        let output = self.allocate_shared(linear.elements() * 2)?;
        let command = self.command()?;
        // The default encoder is serial: the RMS dispatch observes all writes
        // from linear, without a CPU readback or a concurrent-encoder barrier.
        let encoder = Self::compute_encoder(&command)?;
        self.encode_linear(&encoder, &input.buffer, &weight.buffer, &projected, &linear);
        self.encode_rms(&encoder, &projected, &output, scale, &rms);
        encoder.end_encoding();
        Self::submit(&command, "linear/RMS")?;
        Ok(MetalBf16Buffer {
            buffer: output,
            elements: linear.elements(),
        })
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
            LinearGeometry::new(shape, input.elements, left.elements, self.pair_linear_rows)?;
        let right_geometry =
            LinearGeometry::new(shape, input.elements, right.elements, self.pair_linear_rows)?;
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
        encoder.set_compute_pipeline_state(&self.linear);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, 4, (&raw const geometry.columns).cast());
        encoder.set_bytes(4, 4, (&raw const geometry.inner).cast());
        encoder.set_bytes(5, 4, (&raw const geometry.total).cast());
        encoder.set_bytes(6, 4, (&raw const geometry.paired).cast());
        let grid = u64::from(if geometry.paired != 0 {
            geometry.total / 2
        } else {
            geometry.total
        });
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
