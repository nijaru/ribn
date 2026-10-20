//! Chunked BF16 linear and RMS execution on one Metal command queue.

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};
use std::error::Error;
use std::fmt;

/// Metal preparation or execution failure.
#[derive(Debug)]
pub enum MetalBf16Error {
    Device,
    Library(String),
    Function(String),
    Pipeline(String),
    Invalid(&'static str),
    Shape(String),
}

impl fmt::Display for MetalBf16Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device => formatter.write_str("no Metal device is available"),
            Self::Library(message) => write!(formatter, "Metal library: {message}"),
            Self::Function(message) => write!(formatter, "Metal function: {message}"),
            Self::Pipeline(message) => write!(formatter, "Metal pipeline: {message}"),
            Self::Invalid(message) => formatter.write_str(message),
            Self::Shape(message) => write!(formatter, "Metal shape: {message}"),
        }
    }
}

impl Error for MetalBf16Error {}

const THREADS: u64 = 256;
const ATTENTION_SCRATCH_BYTES: usize = 512 * 1024 * 1024;

/// One immutable shared-storage BF16 buffer.
///
/// Initialization/copy or GPU writes finish before publication. No API modifies a
/// published buffer, so CPU borrows and GPU input reads may share its storage.
pub struct MetalBf16Buffer {
    buffer: Buffer,
    elements: usize,
}

impl MetalBf16Buffer {
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    /// Borrow the native BF16 encodings without copying shared storage.
    #[must_use]
    pub fn as_slice(&self) -> &[u16] {
        // SAFETY: shared Metal storage is aligned for u16 and allocated for this
        // element count. Upload copies or GPU writes finish before publication;
        // all subsequent GPU uses are read-only. The borrow keeps the owned Metal
        // buffer alive; no mutable view or raw Metal handle escapes this wrapper.
        unsafe { std::slice::from_raw_parts(self.buffer.contents().cast::<u16>(), self.elements) }
    }

    /// Copy the native BF16 encodings out of shared storage.
    #[must_use]
    pub fn bits(&self) -> Vec<u16> {
        self.as_slice().to_vec()
    }
}

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
    #[expect(
        clippy::too_many_arguments,
        reason = "the validated scalar extent of one attention call"
    )]
    fn new(
        heads: usize,
        kv_heads: usize,
        width: usize,
        window: Option<usize>,
        query_elements: usize,
        key_elements: usize,
        query_start: usize,
        key_start: usize,
    ) -> Result<Self, MetalBf16Error> {
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
            || !key_elements.is_multiple_of(key_step)
        {
            return Err(MetalBf16Error::Shape(
                "attention buffers are not whole positions".into(),
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

/// BF16 linear, RMS and attention pipelines on one Metal device and queue.
pub struct MetalBf16Executor {
    device: Device,
    queue: CommandQueue,
    linear: ComputePipelineState,
    pair_linear_rows: bool,
    rms: ComputePipelineState,
    attention_scores: ComputePipelineState,
    attention_reduce: ComputePipelineState,
    attention_values: ComputePipelineState,
    rotary: ComputePipelineState,
    elementwise: ComputePipelineState,
}

impl MetalBf16Executor {
    /// Compile the kernels at runtime on the default Apple GPU.
    /// # Errors
    /// Rejects a missing device, MSL compilation or pipeline creation failure.
    pub fn new() -> Result<Self, MetalBf16Error> {
        let device = Device::system_default().ok_or(MetalBf16Error::Device)?;
        let source = include_str!("bf16.metal");
        let library = device
            .new_library_with_source(source, &CompileOptions::new())
            .map_err(MetalBf16Error::Library)?;
        let linear = library
            .get_function("bf16_linear", None)
            .map_err(MetalBf16Error::Function)?;
        let rms = library
            .get_function("bf16_rms", None)
            .map_err(MetalBf16Error::Function)?;
        let linear = device
            .new_compute_pipeline_state_with_function(&linear)
            .map_err(MetalBf16Error::Pipeline)?;
        let rms = device
            .new_compute_pipeline_state_with_function(&rms)
            .map_err(MetalBf16Error::Pipeline)?;
        let attention_scores = library
            .get_function("bf16_attention_scores", None)
            .map_err(MetalBf16Error::Function)?;
        let attention_scores = device
            .new_compute_pipeline_state_with_function(&attention_scores)
            .map_err(MetalBf16Error::Pipeline)?;
        let attention_reduce = library
            .get_function("bf16_attention_reduce", None)
            .map_err(MetalBf16Error::Function)?;
        let attention_reduce = device
            .new_compute_pipeline_state_with_function(&attention_reduce)
            .map_err(MetalBf16Error::Pipeline)?;
        let attention_values = library
            .get_function("bf16_attention_values", None)
            .map_err(MetalBf16Error::Function)?;
        let attention_values = device
            .new_compute_pipeline_state_with_function(&attention_values)
            .map_err(MetalBf16Error::Pipeline)?;
        let rotary = library
            .get_function("bf16_rotary_apply", None)
            .map_err(MetalBf16Error::Function)?;
        let rotary = device
            .new_compute_pipeline_state_with_function(&rotary)
            .map_err(MetalBf16Error::Pipeline)?;
        let elementwise = library
            .get_function("bf16_elementwise", None)
            .map_err(MetalBf16Error::Function)?;
        let elementwise = device
            .new_compute_pipeline_state_with_function(&elementwise)
            .map_err(MetalBf16Error::Pipeline)?;
        let queue = device.new_command_queue();
        // Do not extend an optimization measured on M3 Max to other Apple GPUs.
        let pair_linear_rows = device.name() == "Apple M3 Max";
        Ok(Self {
            device,
            queue,
            linear,
            pair_linear_rows,
            rms,
            attention_scores,
            attention_reduce,
            attention_values,
            rotary,
            elementwise,
        })
    }

    /// Upload native BF16 encodings into one shared-storage buffer.
    /// # Errors
    /// Rejects empty input.
    pub fn upload(&self, values: &[u16]) -> Result<MetalBf16Buffer, MetalBf16Error> {
        if values.is_empty() {
            return Err(MetalBf16Error::Invalid("empty Metal buffer"));
        }
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let buffer = self.device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(MetalBf16Buffer {
            buffer,
            elements: values.len(),
        })
    }

    /// Row-major `output[m,n] = input[m,k] · weight[n,k]` in BF16.
    ///
    /// Each output accumulates in FP32 in ascending inner-index order and rounds once.
    /// # Errors
    /// Rejects mismatched/empty shapes or element counts beyond u32 kernel offsets.
    pub fn linear(
        &self,
        input: &MetalBf16Buffer,
        weight: &MetalBf16Buffer,
        rows: usize,
        columns: usize,
        inner: usize,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
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
            || input.elements != input_length
            || weight.elements != weight_length
        {
            return Err(MetalBf16Error::Shape(
                "linear buffers disagree with the requested shape".into(),
            ));
        }
        // Kernel element offsets use u32 products, not only u32 dimensions.
        if u32::try_from(input_length).is_err() || u32::try_from(weight_length).is_err() {
            return Err(MetalBf16Error::Shape(
                "linear input or weight exceeds u32 elements".into(),
            ));
        }
        let columns = u32::try_from(columns)
            .map_err(|_| MetalBf16Error::Shape("columns exceed u32".into()))?;
        let inner =
            u32::try_from(inner).map_err(|_| MetalBf16Error::Shape("inner exceeds u32".into()))?;
        let total = u32::try_from(output_length)
            .map_err(|_| MetalBf16Error::Shape("output exceeds u32 elements".into()))?;
        let output = self.device.new_buffer(
            (output_length * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.linear);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&weight.buffer), 0);
        encoder.set_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, 4, (&raw const columns).cast());
        encoder.set_bytes(4, 4, (&raw const inner).cast());
        encoder.set_bytes(5, 4, (&raw const total).cast());
        // Only the measured multi-row range uses register blocking. Decode,
        // small/tail chunks and larger unqualified batches keep the exact kernel.
        let paired = u32::from(
            self.pair_linear_rows
                && (8..=64).contains(&rows)
                && rows.is_multiple_of(2)
                && inner.is_multiple_of(4),
        );
        encoder.set_bytes(6, 4, (&raw const paired).cast());
        let grid = u64::from(if paired != 0 { total / 2 } else { total });
        encoder.dispatch_threads(
            MTLSize::new(grid, 1, 1),
            MTLSize::new(grid.min(THREADS), 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(MetalBf16Buffer {
            buffer: output,
            elements: output_length,
        })
    }

    /// Grouped RMS over rows of `width`, with an optional direct BF16 scale.
    /// # Errors
    /// Rejects empty/whole-row-mismatched shapes or an invalid epsilon.
    pub fn rms_norm(
        &self,
        input: &MetalBf16Buffer,
        rows: usize,
        width: usize,
        epsilon: f32,
        scale: Option<&MetalBf16Buffer>,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        let expected = rows
            .checked_mul(width)
            .ok_or_else(|| MetalBf16Error::Shape("RMS dimensions overflow".into()))?;
        if rows == 0
            || width == 0
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || input.elements != expected
            || scale.is_some_and(|scale| scale.elements != width)
        {
            return Err(MetalBf16Error::Shape(
                "RMS buffers disagree with the requested shape".into(),
            ));
        }
        let output = self
            .device
            .new_buffer((expected * 2) as u64, MTLResourceOptions::StorageModeShared);
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rms);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        let scale_buffer = scale.map_or(&input.buffer, |scale| &scale.buffer);
        encoder.set_buffer(1, Some(scale_buffer), 0);
        encoder.set_buffer(2, Some(&output), 0);
        let width =
            u32::try_from(width).map_err(|_| MetalBf16Error::Shape("width exceeds u32".into()))?;
        encoder.set_bytes(3, 4, (&raw const width).cast());
        encoder.set_bytes(4, 4, (&raw const epsilon).cast());
        let has_scale = u32::from(scale.is_some());
        encoder.set_bytes(5, 4, (&raw const has_scale).cast());
        encoder
            .dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(THREADS, 1, 1));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(MetalBf16Buffer {
            buffer: output,
            elements: expected,
        })
    }

    /// Causal grouped-query attention over explicit absolute spans.
    ///
    /// Score, per-row softmax reduction and weighted-value kernels read the same
    /// arithmetic as a single-pass implementation: scores and weights round to
    /// BF16, the reduction accumulates in FP32 and the output rounds once.
    /// # Errors
    /// Rejects malformed geometry, disagreeing buffer lengths, span violations or
    /// a score scratch above its fixed bound.
    pub fn attention(
        &self,
        query: &MetalBf16Buffer,
        keys: &MetalBf16Buffer,
        values: &MetalBf16Buffer,
        config: MetalAttentionConfig,
        query_start: usize,
        key_start: usize,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        let MetalAttentionConfig {
            heads,
            kv_heads,
            width,
            window,
        } = config;
        let geometry = AttentionGeometry::new(
            heads,
            kv_heads,
            width,
            window,
            query.elements,
            keys.elements,
            query_start,
            key_start,
        )?;
        let output = self.device.new_buffer(
            (query.elements * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scores_buffer = self.device.new_buffer(
            u64::from(geometry.scores) * 2,
            MTLResourceOptions::StorageModeShared,
        );
        let summary_buffer = self.device.new_buffer(
            u64::from(geometry.rows) * 8,
            MTLResourceOptions::StorageModeShared,
        );
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.attention_scores);
        encoder.set_buffer(0, Some(&query.buffer), 0);
        encoder.set_buffer(1, Some(&keys.buffer), 0);
        encoder.set_buffer(2, Some(&scores_buffer), 0);
        geometry.set_bytes(encoder, 3, false);
        encoder.set_bytes(10, 4, (&raw const geometry.scores).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(geometry.scores), 1, 1),
            MTLSize::new(u64::from(geometry.scores).min(THREADS), 1, 1),
        );
        encoder.set_compute_pipeline_state(&self.attention_reduce);
        encoder.set_buffer(0, Some(&scores_buffer), 0);
        encoder.set_buffer(1, Some(&summary_buffer), 0);
        geometry.set_bytes(encoder, 2, true);
        encoder.set_bytes(7, 4, (&raw const geometry.rows).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(geometry.rows), 1, 1),
            MTLSize::new(u64::from(geometry.rows).min(THREADS), 1, 1),
        );
        encoder.set_compute_pipeline_state(&self.attention_values);
        encoder.set_buffer(0, Some(&values.buffer), 0);
        encoder.set_buffer(1, Some(&scores_buffer), 0);
        encoder.set_buffer(2, Some(&summary_buffer), 0);
        encoder.set_buffer(3, Some(&output), 0);
        geometry.set_bytes(encoder, 4, false);
        encoder.set_bytes(11, 4, (&raw const geometry.values).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(geometry.values), 1, 1),
            MTLSize::new(u64::from(geometry.values).min(THREADS), 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(MetalBf16Buffer {
            buffer: output,
            elements: query.elements,
        })
    }

    /// Split-half rotation with prepared BF16 factors `[rows, width]`.
    ///
    /// Channel `j` pairs with `j + width/2`; each product rounds to BF16 before
    /// the FP32 addition, and both factor halves are read.
    /// # Errors
    /// Rejects odd/zero widths, non-whole heads or disagreeing buffer lengths.
    pub fn rotary_apply(
        &self,
        input: &MetalBf16Buffer,
        cosine: &MetalBf16Buffer,
        sine: &MetalBf16Buffer,
        rows: usize,
        heads: usize,
        width: usize,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        if rows == 0 || heads == 0 || width == 0 || !width.is_multiple_of(2) {
            return Err(MetalBf16Error::Shape(
                "rotary needs positive rows/heads and an even width".into(),
            ));
        }
        let columns = heads
            .checked_mul(width)
            .ok_or_else(|| MetalBf16Error::Shape("rotary head stride overflow".into()))?;
        let input_length = rows
            .checked_mul(columns)
            .ok_or_else(|| MetalBf16Error::Shape("rotary input overflow".into()))?;
        let factor_length = rows
            .checked_mul(width)
            .ok_or_else(|| MetalBf16Error::Shape("rotary factor overflow".into()))?;
        let half = width / 2;
        let pairs = rows
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(half))
            .ok_or_else(|| MetalBf16Error::Shape("rotary pair count overflow".into()))?;
        if input.elements != input_length
            || cosine.elements != factor_length
            || sine.elements != factor_length
        {
            return Err(MetalBf16Error::Shape(
                "rotary buffers disagree with the requested shape".into(),
            ));
        }
        let output = self.device.new_buffer(
            (input_length * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rotary);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&cosine.buffer), 0);
        encoder.set_buffer(2, Some(&sine.buffer), 0);
        encoder.set_buffer(3, Some(&output), 0);
        let columns = u32::try_from(columns)
            .map_err(|_| MetalBf16Error::Shape("columns exceed u32".into()))?;
        let width =
            u32::try_from(width).map_err(|_| MetalBf16Error::Shape("width exceeds u32".into()))?;
        let pairs = u32::try_from(pairs)
            .map_err(|_| MetalBf16Error::Shape("rotary pairs exceed u32".into()))?;
        encoder.set_bytes(4, 4, (&raw const columns).cast());
        encoder.set_bytes(5, 4, (&raw const width).cast());
        encoder.set_bytes(6, 4, (&raw const pairs).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(pairs), 1, 1),
            MTLSize::new(u64::from(pairs).min(256), 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(MetalBf16Buffer {
            buffer: output,
            elements: input_length,
        })
    }

    /// Elementwise BF16 addition with one rounding.
    /// # Errors
    /// Rejects disagreeing buffer lengths.
    pub fn add(
        &self,
        left: &MetalBf16Buffer,
        right: &MetalBf16Buffer,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        self.elementwise(0, 0.0, left, Some(right))
    }

    /// Elementwise BF16 multiplication with one rounding.
    /// # Errors
    /// Rejects disagreeing buffer lengths.
    pub fn multiply(
        &self,
        left: &MetalBf16Buffer,
        right: &MetalBf16Buffer,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        self.elementwise(1, 0.0, left, Some(right))
    }

    /// Multiply every element by one FP32 scalar with one rounding.
    /// # Errors
    /// Rejects a non-finite scalar.
    pub fn scale(
        &self,
        input: &MetalBf16Buffer,
        factor: f32,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        if !factor.is_finite() {
            return Err(MetalBf16Error::Invalid("scale factor must be finite"));
        }
        self.elementwise(2, factor, input, None)
    }

    /// Tanh-approximated GELU with one rounding.
    /// # Errors
    /// Rejects an empty buffer.
    pub fn gelu(&self, input: &MetalBf16Buffer) -> Result<MetalBf16Buffer, MetalBf16Error> {
        self.elementwise(3, 0.0, input, None)
    }

    /// Final logit soft cap: divide, tanh and multiply with BF16 rounding each step.
    /// # Errors
    /// Rejects a non-positive/non-finite cap.
    pub fn softcap(
        &self,
        input: &MetalBf16Buffer,
        cap: f32,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        if !cap.is_finite() || cap <= 0.0 {
            return Err(MetalBf16Error::Invalid(
                "soft cap must be finite and positive",
            ));
        }
        self.elementwise(4, cap, input, None)
    }

    fn elementwise(
        &self,
        operation: u32,
        scalar: f32,
        left: &MetalBf16Buffer,
        right: Option<&MetalBf16Buffer>,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        if left.elements == 0 || right.is_some_and(|right| right.elements != left.elements) {
            return Err(MetalBf16Error::Shape(
                "elementwise buffers disagree with the requested shape".into(),
            ));
        }
        let total = u32::try_from(left.elements)
            .map_err(|_| MetalBf16Error::Shape("elementwise length exceeds u32".into()))?;
        let output = self.device.new_buffer(
            (left.elements * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.elementwise);
        encoder.set_buffer(0, Some(&left.buffer), 0);
        let right_buffer = right.map_or(&left.buffer, |right| &right.buffer);
        encoder.set_buffer(1, Some(right_buffer), 0);
        encoder.set_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, 4, (&raw const operation).cast());
        encoder.set_bytes(4, 4, (&raw const scalar).cast());
        encoder.set_bytes(5, 4, (&raw const total).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(total), 1, 1),
            MTLSize::new(u64::from(total).min(THREADS), 1, 1),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        Ok(MetalBf16Buffer {
            buffer: output,
            elements: left.elements,
        })
    }
}
