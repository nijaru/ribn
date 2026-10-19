//! Grouped BF16 RMS on the same completion-owned storage as linear execution.
use super::{
    Access, CudaBf16ComputeError, CudaBf16Error, CudaBf16Matrix, CudaBf16Weights, MatrixState,
    Pending, Program, Retirement, drain, error,
};
use cudarc::{
    driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};
use engine_core::DataType;
use std::sync::Arc;

pub(super) struct RmsKernels {
    pub(super) stream: Arc<CudaStream>,
    function: CudaFunction,
}

/// Prepared grouped RMS: FP32 variance/normalization, optional direct BF16 scale,
/// then BF16 round-to-nearest-even. This does not implement model layer ordering.
/// No per-call device allocation or global workspace is used. Compiled code and
/// driver/context overhead are not application byte-pool allocations.
pub struct CudaBf16RmsNorm {
    kernels: Option<Arc<RmsKernels>>,
    poisoned: bool,
}
impl CudaBf16RmsNorm {
    /// Compile and load the kernel on one stable associated stream.
    /// # Errors
    /// Rejects per-thread pseudo-streams and compilation/module-loading failures.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CudaBf16ComputeError> {
        if stream.cu_stream().addr() == 0x2 {
            return Err(CudaBf16Error::Invalid("RMS requires a stable stream").into());
        }
        let ptx =
            compile_ptx(include_str!("rms.cu")).map_err(|e| error("BF16 RMS compilation", e))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| error("BF16 RMS module", e))?;
        let function = module
            .load_function("bf16_rms")
            .map_err(|e| error("BF16 RMS function", e))?;
        Ok(Self {
            kernels: Some(Arc::new(RmsKernels { stream, function })),
            poisoned: false,
        })
    }

    /// Normalize contiguous groups of `width` within each matrix row. A scale slot
    /// must be exactly `[width]`; `None` performs scale-free RMS, not `1 + weight`.
    /// Input/output have identical shapes and must use this exact stream object.
    /// Successful return includes completion; output may be reused between calls.
    /// # Errors
    /// Whole-call validation preserves output. Possible enqueue/completion failure
    /// poisons this executor/output and retains code and all storage for retirement.
    pub fn normalize_into(
        &mut self,
        scale: Option<(&mut CudaBf16Weights, usize)>,
        width: usize,
        epsilon: f32,
        input: &mut CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        let kernels = self.kernels.as_ref().ok_or(CudaBf16Error::Closed)?;
        if self.poisoned {
            return Err(CudaBf16Error::Poisoned.into());
        }
        input.check_open()?;
        output.check_open()?;
        if !input.ready() {
            return Err(CudaBf16Error::NotReady.into());
        }
        let groups = input.shape.validate_rms(output.shape, width, epsilon)?;
        let (weights, weight_pointer) = if let Some((weights, slot)) = scale {
            let spec = weights.spec(slot).ok_or(CudaBf16Error::InvalidSlot(slot))?;
            if spec.dtype() != DataType::BF16 || spec.dimensions() != [width as u64] {
                return Err(CudaBf16Error::Invalid("RMS scale must be BF16 [width]").into());
            }
            let (allocation, range) = weights.access(slot)?;
            if range.end.checked_sub(range.start) != Some(width as u64 * 2)
                || range.end > allocation.allocated_bytes()
                || !Arc::ptr_eq(&kernels.stream, &allocation.transport().stream)
            {
                return Err(CudaBf16Error::Invalid("RMS scale range or stream mismatch").into());
            }
            let pointer = allocation
                .pointer()?
                .checked_add(range.start)
                .ok_or(CudaBf16Error::Invalid("RMS scale address overflow"))?;
            (Some(allocation), pointer)
        } else {
            (None, 0_u64)
        };
        if !Arc::ptr_eq(&kernels.stream, &input.allocation.transport().stream)
            || !Arc::ptr_eq(&kernels.stream, &output.allocation.transport().stream)
        {
            return Err(CudaBf16Error::Invalid("RMS requires one associated stream").into());
        }
        let x = input.allocation.pointer()?;
        let y = output.allocation.pointer()?;
        let width =
            u32::try_from(width).map_err(|_| CudaBf16Error::Invalid("RMS width exceeds u32"))?;
        let mut pending = Pending::new(
            Access {
                program: Program::Rms(Arc::clone(kernels)),
                _weights: weights,
                _input: Arc::clone(&input.allocation),
                _output: Arc::clone(&output.allocation),
            },
            drain,
        );
        self.poisoned = true;
        output.state = MatrixState::Poisoned;
        let mut launch = kernels.stream.launch_builder(&kernels.function);
        launch
            .arg(&x)
            .arg(&weight_pointer)
            .arg(&y)
            .arg(&width)
            .arg(&epsilon);
        // SAFETY: complete preflight establishes live disjoint BF16 buffers, whole
        // groups and bounded grid/addresses. The pending envelope retains function,
        // stream/context and every accessed allocation before possible enqueue.
        let enqueue = unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (groups, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| error("BF16 RMS enqueue", e));
        if let Err(cause) = enqueue.and_then(|()| pending.retire()) {
            return Err(CudaBf16ComputeError {
                cause,
                retirement: Some(Retirement::Execution(pending)),
            });
        }
        output.state = MatrixState::Ready;
        self.poisoned = false;
        Ok(())
    }

    /// Drain and relinquish compiled code references. Driver-managed code unloading
    /// follows the CUDA context; no application allocation is refunded here.
    /// # Errors
    /// Rejects retained access guards or a failed associated-stream drain.
    pub fn close(&mut self) -> Result<(), CudaBf16Error> {
        if let Some(kernels) = self.kernels.as_mut() {
            let kernels = Arc::get_mut(kernels).ok_or(CudaBf16Error::InUse)?;
            kernels
                .stream
                .synchronize()
                .map_err(|e| error("BF16 RMS close", e))?;
        }
        self.kernels = None;
        Ok(())
    }
}
