//! Apply prepared BF16 rotary factors; frequency/position preparation is separate.
use super::{
    Access, CudaBf16ComputeError, CudaBf16Error, CudaBf16Matrix, MatrixState, Operation, Pending,
    Retirement, drain, error,
};
use cudarc::{
    driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};
use std::sync::Arc;

pub(super) struct RotaryKernels {
    pub(super) stream: Arc<CudaStream>,
    function: CudaFunction,
}

/// Split-half rotation with prepared BF16 cosine/sine. Each multiplication rounds
/// to BF16 before addition and final BF16 rounding, matching eager BF16 arithmetic.
/// This does not generate factors, select positions or implement model execution.
/// No device storage is allocated per call; code/context overhead is driver-owned.
pub struct CudaBf16RotaryApply {
    kernels: Option<Arc<RotaryKernels>>,
    poisoned: bool,
}
impl CudaBf16RotaryApply {
    /// Compile/load on one stable stream, shared by all operands.
    /// # Errors
    /// Rejects per-thread pseudo-streams and compilation/module-loading failures.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CudaBf16ComputeError> {
        if stream.cu_stream().addr() == 0x2 {
            return Err(CudaBf16Error::Invalid("rotary requires a stable stream").into());
        }
        let ptx = compile_ptx(concat!(include_str!("bf16.cuh"), include_str!("rotary.cu")))
            .map_err(|e| error("BF16 rotary compilation", e))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| error("BF16 rotary module", e))?;
        let function = module
            .load_function("bf16_rotary_apply")
            .map_err(|e| error("BF16 rotary function", e))?;
        Ok(Self {
            kernels: Some(Arc::new(RotaryKernels { stream, function })),
            poisoned: false,
        })
    }

    /// Apply factors `[M,D]` to each head of input/output `[M,H*D]`, with even D.
    /// Channel j pairs with j+D/2. Both halves of the factors are read explicitly;
    /// the model, not this backend, owns any repeated or padded factor semantics.
    /// Every matrix must be ready (except output) and on this exact stream object.
    /// Successful return includes completion. Output may be preallocated/reused.
    /// # Errors
    /// Complete validation preserves output. Possible enqueue/completion failures
    /// poison this executor/output and retain all four matrices and compiled code.
    pub fn apply_into(
        &mut self,
        cos: &mut CudaBf16Matrix,
        sin: &mut CudaBf16Matrix,
        input: &mut CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        let kernels = self.kernels.as_ref().ok_or(CudaBf16Error::Closed)?;
        if self.poisoned {
            return Err(CudaBf16Error::Poisoned.into());
        }
        for matrix in [&*input, &*output, &*cos, &*sin] {
            matrix.check_open()?;
            if !Arc::ptr_eq(&kernels.stream, &matrix.allocation.transport().stream) {
                return Err(CudaBf16Error::Invalid("rotary requires one associated stream").into());
            }
        }
        if !input.ready() || !cos.ready() || !sin.ready() {
            return Err(CudaBf16Error::NotReady.into());
        }
        let blocks = input
            .shape
            .validate_rotary(cos.shape, sin.shape, output.shape)?;
        let x = input.allocation.pointer()?;
        let y = output.allocation.pointer()?;
        let c = cos.allocation.pointer()?;
        let s = sin.allocation.pointer()?;
        let columns = u32::try_from(input.shape.cols)
            .map_err(|_| CudaBf16Error::Invalid("rotary columns exceed u32"))?;
        let width = u32::try_from(cos.shape.cols)
            .map_err(|_| CudaBf16Error::Invalid("rotary width exceeds u32"))?;
        let pairs = input.shape.rows as u64 * (input.shape.cols as u64 / 2);
        let mut pending = Pending::new(
            Access {
                operation: Operation::Rotary {
                    program: Arc::clone(kernels),
                    _cos: Arc::clone(&cos.allocation),
                    _sin: Arc::clone(&sin.allocation),
                },
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
            .arg(&c)
            .arg(&s)
            .arg(&y)
            .arg(&columns)
            .arg(&width)
            .arg(&pairs);
        // SAFETY: whole-shape/live-range checks and private non-cloneable matrices
        // establish disjoint valid pointers. Grid covers every pair exactly once.
        // Pending owns code, stream/context and all four allocations before enqueue.
        let enqueue = unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| error("BF16 rotary enqueue", e));
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

    /// Drain and relinquish code references; no application storage is freed here.
    /// # Errors
    /// Rejects retained accesses or a failed associated-stream drain.
    pub fn close(&mut self) -> Result<(), CudaBf16Error> {
        if let Some(kernels) = self.kernels.as_mut() {
            let kernels = Arc::get_mut(kernels).ok_or(CudaBf16Error::InUse)?;
            kernels
                .stream
                .synchronize()
                .map_err(|e| error("BF16 rotary close", e))?;
        }
        self.kernels = None;
        Ok(())
    }
}
