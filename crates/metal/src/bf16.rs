//! Chunked BF16 linear and RMS execution on one Metal command queue.

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
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

/// One shared-storage BF16 buffer. Values are copied in and out; the handle owns
/// its Metal buffer and exposes no pointer.
pub struct MetalBf16Buffer {
    buffer: Buffer,
    elements: usize,
}

impl MetalBf16Buffer {
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    /// Copy the native BF16 encodings out of shared storage.
    #[must_use]
    pub fn bits(&self) -> Vec<u16> {
        let length = self.elements * 2;
        // SAFETY: the buffer was allocated for exactly this many bytes and the
        // command queue drained before publication.
        let bytes =
            unsafe { std::slice::from_raw_parts(self.buffer.contents().cast::<u8>(), length) };
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect()
    }
}

/// BF16 linear and RMS pipelines on one Metal device and queue.
pub struct MetalBf16Executor {
    device: Device,
    queue: CommandQueue,
    linear: ComputePipelineState,
    rms: ComputePipelineState,
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
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            linear,
            rms,
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
    /// One GPU thread per output element accumulates in FP32 and rounds once.
    /// # Errors
    /// Rejects dimensions that disagree with either buffer.
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
        let columns = u32::try_from(columns)
            .map_err(|_| MetalBf16Error::Shape("columns exceed u32".into()))?;
        let inner =
            u32::try_from(inner).map_err(|_| MetalBf16Error::Shape("inner exceeds u32".into()))?;
        let total = u32::try_from(output_length)
            .map_err(|_| MetalBf16Error::Shape("output exceeds u32 elements".into()))?;
        encoder.set_bytes(3, 4, (&raw const columns).cast());
        encoder.set_bytes(4, 4, (&raw const inner).cast());
        encoder.set_bytes(5, 4, (&raw const total).cast());
        let grid = u64::from(total);
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
}
