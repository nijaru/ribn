//! BF16 execution with synchronous completion on one Metal command queue.

use metal::{Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLSize};
use std::error::Error;
use std::fmt;
use std::{mem::ManuallyDrop, sync::Mutex};

mod attention;
mod execution;
mod graph;
mod loading;
mod projection;
mod rotary;
mod submission;

pub use attention::MetalAttentionConfig;
pub use projection::{
    MetalAttentionProjections, MetalFfnWeights, MetalLinearShape, MetalNormalizedProjection,
    MetalProjectionNorm,
};
pub use rotary::MetalRotaryFactors;

/// Metal preparation or execution failure.
#[derive(Debug)]
pub enum MetalBf16Error {
    Device,
    Unsupported(&'static str),
    Library(String),
    Function(String),
    Pipeline(String),
    Invalid(&'static str),
    Shape(String),
    Allocation(String),
    Execution(String),
    Read(std::io::Error),
}

impl fmt::Display for MetalBf16Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device => formatter.write_str("no Metal device is available"),
            Self::Unsupported(message) => write!(formatter, "Metal unsupported: {message}"),
            Self::Library(message) => write!(formatter, "Metal library: {message}"),
            Self::Function(message) => write!(formatter, "Metal function: {message}"),
            Self::Pipeline(message) => write!(formatter, "Metal pipeline: {message}"),
            Self::Invalid(message) => formatter.write_str(message),
            Self::Shape(message) => write!(formatter, "Metal shape: {message}"),
            Self::Allocation(message) => write!(formatter, "Metal allocation: {message}"),
            Self::Execution(message) => write!(formatter, "Metal execution: {message}"),
            Self::Read(error) => write!(formatter, "Metal BF16 source: {error}"),
        }
    }
}

impl Error for MetalBf16Error {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read(error) => Some(error),
            _ => None,
        }
    }
}

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

/// BF16 linear, RMS and attention pipelines on one Metal device and queue.
pub struct MetalBf16Executor {
    execution: ManuallyDrop<Mutex<execution::Execution>>,
    ffn_shape: Option<graph::FfnShape>,
    device: Device,
    queue: CommandQueue,
    linear: ComputePipelineState,
    linear_rows4: ComputePipelineState,
    block_linear_rows: bool,
    rms: ComputePipelineState,
    attention_scores: ComputePipelineState,
    attention_reduce: ComputePipelineState,
    attention_values: ComputePipelineState,
    rotary: ComputePipelineState,
    elementwise: ComputePipelineState,
    lookup_product: ComputePipelineState,
}

impl Drop for MetalBf16Executor {
    fn drop(&mut self) {
        let owner = self
            .execution
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if owner.retire().is_ok() {
            // SAFETY: terminal retirement acknowledged. Drop exactly once; an
            // unresolved envelope instead stays permanently quarantined.
            unsafe {
                ManuallyDrop::drop(&mut self.execution);
            }
        }
    }
}

impl MetalBf16Executor {
    /// Compile the kernels at runtime on the default Apple GPU.
    /// # Errors
    /// Rejects a missing device, MSL compilation or pipeline creation failure.
    pub fn new() -> Result<Self, MetalBf16Error> {
        Self::prepare(None)
    }

    /// Prepare experimental BF16 FFN Graphs on the qualified M3 Max backend.
    ///
    /// Pair projections select 16/64 rows; Down selects 64. All other shapes
    /// remain native. No extra weight inventory is created. This is explicit
    /// opt-in, not physical-memory or serving qualification.
    /// # Errors
    /// Rejects unavailable SDK/runtime/device support, invalid geometry or
    /// compilation failure before any model weights are loaded.
    pub fn with_ffn_graphs(hidden: usize, intermediate: usize) -> Result<Self, MetalBf16Error> {
        Self::prepare(Some(graph::FfnShape {
            hidden,
            intermediate,
        }))
    }

    fn prepare(ffn_shape: Option<graph::FfnShape>) -> Result<Self, MetalBf16Error> {
        let device = Device::system_default().ok_or(MetalBf16Error::Device)?;
        if ffn_shape.is_some() {
            graph::FfnGraphs::check_available(&device)?;
        }
        let source = format!(
            "{}\n{}",
            include_str!("bf16.metal"),
            include_str!("bf16/ffn.metal")
        );
        let library = device
            .new_library_with_source(&source, &CompileOptions::new())
            .map_err(MetalBf16Error::Library)?;
        let linear = library
            .get_function("bf16_linear", None)
            .map_err(MetalBf16Error::Function)?;
        let linear_rows4 = library
            .get_function("bf16_linear_rows4", None)
            .map_err(MetalBf16Error::Function)?;
        let linear_rows4 = device
            .new_compute_pipeline_state_with_function(&linear_rows4)
            .map_err(MetalBf16Error::Pipeline)?;
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
        let lookup_product = library
            .get_function("bf16_lookup_product", None)
            .map_err(MetalBf16Error::Function)?;
        let lookup_product = device
            .new_compute_pipeline_state_with_function(&lookup_product)
            .map_err(MetalBf16Error::Pipeline)?;
        let queue = Self::command_queue(&device)?;
        // Do not extend an optimization measured on M3 Max to other Apple GPUs.
        let block_linear_rows = device.name() == "Apple M3 Max";
        let graphs = ffn_shape
            .map(|shape| graph::FfnGraphs::new(&device, &queue, shape))
            .transpose()?;
        let execution = ManuallyDrop::new(Mutex::new(execution::Execution::new(
            &device, &queue, graphs,
        )));
        Ok(Self {
            execution,
            ffn_shape,
            device,
            queue,
            linear,
            linear_rows4,
            block_linear_rows,
            rms,
            attention_scores,
            attention_reduce,
            attention_values,
            rotary,
            elementwise,
            lookup_product,
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
        let output = self.allocate_shared(left.elements * 2)?;
        let mut command = self.command()?;
        let encoder = Self::compute_encoder(&command)?;
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
        command.submit("elementwise")?;
        Ok(MetalBf16Buffer {
            buffer: output,
            elements: left.elements,
        })
    }
}
