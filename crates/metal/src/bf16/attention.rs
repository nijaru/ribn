//! Fixed attention and attention-output chains with private synchronous scratch.

use super::projection::ProjectionGeometry;
use super::{
    ATTENTION_SCRATCH_BYTES, MetalBf16Buffer, MetalBf16Error, MetalBf16Executor, MetalLinearShape,
    MetalNormalizedProjection, THREADS,
};
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

/// Attention geometry and mask policy for one layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalAttentionConfig {
    pub heads: usize,
    pub kv_heads: usize,
    pub width: usize,
    /// Sliding window length in positions; `None` is full attention.
    pub window: Option<usize>,
}

/// Validated attention geometry and scalar kernel arguments.
struct AttentionGeometry {
    rows: u32,
    scores: u32,
    values: u32,
    heads: u32,
    kv_heads: u32,
    width: u32,
    key_sequence: u32,
    query_start: u32,
    key_start: u32,
    window: u32,
}

impl AttentionGeometry {
    fn new(
        config: MetalAttentionConfig,
        query_elements: usize,
        key_elements: usize,
        value_elements: usize,
        query_start: usize,
        key_start: usize,
    ) -> Result<Self, MetalBf16Error> {
        let MetalAttentionConfig {
            heads,
            kv_heads,
            width,
            window,
        } = config;
        if heads == 0 || kv_heads == 0 || width == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(MetalBf16Error::Shape(
                "attention heads or width are invalid".into(),
            ));
        }
        if window == Some(0) {
            return Err(MetalBf16Error::Shape(
                "attention window must be positive".into(),
            ));
        }
        let query_step = heads
            .checked_mul(width)
            .ok_or_else(|| MetalBf16Error::Shape("attention query stride overflow".into()))?;
        let key_step = kv_heads
            .checked_mul(width)
            .ok_or_else(|| MetalBf16Error::Shape("attention key stride overflow".into()))?;
        if query_elements == 0
            || !query_elements.is_multiple_of(query_step)
            || key_elements == 0
            || value_elements != key_elements
            || !key_elements.is_multiple_of(key_step)
        {
            return Err(MetalBf16Error::Shape(
                "attention buffers must have matching whole key/value positions".into(),
            ));
        }
        let sequence = query_elements / query_step;
        let key_sequence = key_elements / key_step;
        let last_query = query_start
            .checked_add(sequence)
            .ok_or_else(|| MetalBf16Error::Shape("attention query span overflow".into()))?;
        let last_key = key_start
            .checked_add(key_sequence)
            .ok_or_else(|| MetalBf16Error::Shape("attention key span overflow".into()))?;
        if last_query > last_key || query_start < key_start {
            return Err(MetalBf16Error::Shape(
                "attention queries precede or outrun the key span".into(),
            ));
        }
        let rows = sequence
            .checked_mul(heads)
            .ok_or_else(|| MetalBf16Error::Shape("attention rows overflow".into()))?;
        let scores = rows
            .checked_mul(key_sequence)
            .ok_or_else(|| MetalBf16Error::Shape("attention score count overflow".into()))?;
        if scores
            .checked_mul(2)
            .is_none_or(|bytes| bytes > ATTENTION_SCRATCH_BYTES)
        {
            return Err(MetalBf16Error::Shape(
                "attention score scratch exceeds its 512 MiB bound".into(),
            ));
        }
        let convert = |value: usize, what: &str| {
            u32::try_from(value).map_err(|_| MetalBf16Error::Shape(format!("{what} exceeds u32")))
        };
        Ok(Self {
            rows: convert(rows, "rows")?,
            scores: convert(scores, "scores")?,
            values: convert(query_elements, "attention output")?,
            heads: convert(heads, "heads")?,
            kv_heads: convert(kv_heads, "key-value heads")?,
            width: convert(width, "width")?,
            key_sequence: convert(key_sequence, "keys")?,
            query_start: convert(query_start, "query start")?,
            key_start: convert(key_start, "key start")?,
            window: match window {
                None => 0,
                Some(window) => convert(window, "window")?,
            },
        })
    }

    fn set_bytes(&self, encoder: &ComputeCommandEncoderRef, base: u64, reduction: bool) {
        encoder.set_bytes(base, 4, (&raw const self.heads).cast());
        if reduction {
            encoder.set_bytes(base + 1, 4, (&raw const self.key_sequence).cast());
            encoder.set_bytes(base + 2, 4, (&raw const self.query_start).cast());
            encoder.set_bytes(base + 3, 4, (&raw const self.key_start).cast());
            encoder.set_bytes(base + 4, 4, (&raw const self.window).cast());
            encoder.set_bytes(base + 5, 4, (&raw const self.rows).cast());
        } else {
            encoder.set_bytes(base + 1, 4, (&raw const self.kv_heads).cast());
            encoder.set_bytes(base + 2, 4, (&raw const self.width).cast());
            encoder.set_bytes(base + 3, 4, (&raw const self.key_sequence).cast());
            encoder.set_bytes(base + 4, 4, (&raw const self.query_start).cast());
            encoder.set_bytes(base + 5, 4, (&raw const self.key_start).cast());
            encoder.set_bytes(base + 6, 4, (&raw const self.window).cast());
        }
    }
}

/// Scores, row summaries and attended values remain live through the terminal wait.
struct AttentionBuffers {
    geometry: AttentionGeometry,
    scores: Buffer,
    summary: Buffer,
    output: Buffer,
}

impl AttentionBuffers {
    fn allocate(
        executor: &MetalBf16Executor,
        geometry: AttentionGeometry,
    ) -> Result<Self, MetalBf16Error> {
        let output = executor.allocate_shared(geometry.values as usize * 2)?;
        let scores = executor.allocate_shared(geometry.scores as usize * 2)?;
        let summary = executor.allocate_shared(geometry.rows as usize * 8)?;
        Ok(Self {
            geometry,
            scores,
            summary,
            output,
        })
    }

    fn encode(
        &self,
        executor: &MetalBf16Executor,
        encoder: &ComputeCommandEncoderRef,
        query: &MetalBf16Buffer,
        keys: &MetalBf16Buffer,
        values: &MetalBf16Buffer,
    ) {
        let geometry = &self.geometry;
        encoder.set_compute_pipeline_state(&executor.attention_scores);
        encoder.set_buffer(0, Some(&query.buffer), 0);
        encoder.set_buffer(1, Some(&keys.buffer), 0);
        encoder.set_buffer(2, Some(&self.scores), 0);
        geometry.set_bytes(encoder, 3, false);
        encoder.set_bytes(10, 4, (&raw const geometry.scores).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(geometry.scores), 1, 1),
            MTLSize::new(u64::from(geometry.scores).min(THREADS), 1, 1),
        );
        encoder.set_compute_pipeline_state(&executor.attention_reduce);
        encoder.set_buffer(0, Some(&self.scores), 0);
        encoder.set_buffer(1, Some(&self.summary), 0);
        geometry.set_bytes(encoder, 2, true);
        encoder.set_bytes(7, 4, (&raw const geometry.rows).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(geometry.rows), 1, 1),
            MTLSize::new(u64::from(geometry.rows).min(THREADS), 1, 1),
        );
        encoder.set_compute_pipeline_state(&executor.attention_values);
        encoder.set_buffer(0, Some(&values.buffer), 0);
        encoder.set_buffer(1, Some(&self.scores), 0);
        encoder.set_buffer(2, Some(&self.summary), 0);
        encoder.set_buffer(3, Some(&self.output), 0);
        geometry.set_bytes(encoder, 4, false);
        encoder.set_bytes(11, 4, (&raw const geometry.values).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(geometry.values), 1, 1),
            MTLSize::new(u64::from(geometry.values).min(THREADS), 1, 1),
        );
    }

    fn into_output(self) -> MetalBf16Buffer {
        MetalBf16Buffer {
            buffer: self.output,
            elements: self.geometry.values as usize,
        }
    }
}

impl MetalBf16Executor {
    /// Causal grouped-query attention over explicit absolute spans.
    ///
    /// Scores and weights round to BF16; reduction accumulates in FP32.
    /// # Errors
    /// Rejects invalid geometry, lengths, spans, scratch or GPU execution failure.
    pub fn attention(
        &self,
        query: &MetalBf16Buffer,
        keys: &MetalBf16Buffer,
        values: &MetalBf16Buffer,
        config: MetalAttentionConfig,
        query_start: usize,
        key_start: usize,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        let geometry = AttentionGeometry::new(
            config,
            query.elements,
            keys.elements,
            values.elements,
            query_start,
            key_start,
        )?;
        let buffers = AttentionBuffers::allocate(self, geometry)?;
        let mut command = self.command()?;
        let encoder = Self::compute_encoder(&command)?;
        buffers.encode(self, &encoder, query, keys, values);
        encoder.end_encoding();
        command.submit("attention")?;
        Ok(buffers.into_output())
    }

    /// Attend, project and normalize in one serial command using existing kernels.
    ///
    /// The attended BF16 boundary is private. All attention and projection geometry,
    /// scales and optional factors validate before any output/scratch allocation.
    /// # Errors
    /// Rejects malformed geometry, allocation or terminal GPU failure.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit attention inputs/spans and output projection"
    )]
    pub fn attention_project_normalized(
        &self,
        query: &MetalBf16Buffer,
        keys: &MetalBf16Buffer,
        values: &MetalBf16Buffer,
        config: MetalAttentionConfig,
        query_start: usize,
        key_start: usize,
        projection: MetalNormalizedProjection<'_>,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        let geometry = AttentionGeometry::new(
            config,
            query.elements,
            keys.elements,
            values.elements,
            query_start,
            key_start,
        )?;
        // Attention preflight establishes positive heads/width and whole query rows.
        let inner = config.heads * config.width;
        let projection_geometry = ProjectionGeometry::new(
            MetalLinearShape {
                rows: query.elements / inner,
                columns: projection.weight.elements / inner,
                inner,
            },
            query.elements,
            projection.weight.elements,
            projection.norm,
            self.block_linear_rows,
        )?;
        let attention = AttentionBuffers::allocate(self, geometry)?;
        let output = projection_geometry.allocate(self)?;
        let mut command = self.command()?;
        let encoder = Self::compute_encoder(&command)?;
        attention.encode(self, &encoder, query, keys, values);
        output.encode(self, &encoder, &attention.output, &projection.weight.buffer);
        encoder.end_encoding();
        command.submit("attention output projection")?;
        Ok(output.into_output())
    }
}
