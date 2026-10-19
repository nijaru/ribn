use super::{MatrixShape, pending::Pending};
use crate::{
    CudaBf16Weights,
    bf16_weights::cuda::CudaTransport,
    cuda_allocation::{Allocation, CudaBf16Error},
};
use cudarc::{
    cublas::{result, sys},
    driver::CudaStream,
};
use ribn_foundation::{AllocationId, BytePool};
use std::{error::Error, fmt, io::Cursor, sync::Arc};
#[path = "blas.rs"]
mod blas;
use blas::{Blas, error};
#[path = "rms.rs"]
mod rms;
pub use rms::CudaBf16RmsNorm;
#[path = "rotary.rs"]
mod rotary;
pub use rotary::CudaBf16RotaryApply;
#[path = "factors.rs"]
mod factors;
pub use factors::CudaBf16RotaryFactors;

pub(super) type DeviceAllocation = Arc<Allocation<CudaTransport>>;
static ALPHA: f32 = 1.0;
static BETA: f32 = 0.0;
#[derive(Clone, Copy, PartialEq, Eq)]
enum MatrixState {
    Empty,
    Ready,
    Poisoned,
    Closed,
}

/// A charged, row-major BF16 CUDA matrix on one associated stream.
/// No public raw pointer or clone permits mutation behind an outstanding access.
/// Upload and readback complete before returning; an operation's retained access
/// rejects staging access, reuse and close until its error has been retired.
/// Physical driver/context overhead remains outside the supplied pool.
pub struct CudaBf16Matrix {
    allocation: DeviceAllocation,
    shape: MatrixShape,
    state: MatrixState,
}
impl CudaBf16Matrix {
    /// Allocate uninitialized matrix storage, including trailing 16-byte padding.
    /// # Errors
    /// Rejects zero/oversized dimensions, overflow, zero staging, pool or CUDA errors.
    pub fn allocate(
        rows: usize,
        cols: usize,
        stream: Arc<CudaStream>,
        pool: &Arc<BytePool>,
        staging_bytes: usize,
    ) -> Result<Self, CudaBf16Error> {
        let shape = MatrixShape::new(rows, cols)?;
        let allocation = Allocation::allocate(
            shape.allocation_bytes,
            CudaTransport { stream },
            pool,
            staging_bytes,
        )?;
        Ok(Self {
            allocation,
            shape,
            state: MatrixState::Empty,
        })
    }
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.shape.rows, self.shape.cols)
    }
    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        self.shape.allocation_bytes
    }
    #[must_use]
    pub fn allocation_id(&self) -> AllocationId {
        self.allocation.allocation_id()
    }
    #[must_use]
    pub fn ready(&self) -> bool {
        self.state == MatrixState::Ready
    }

    fn check_open(&self) -> Result<(), CudaBf16Error> {
        match self.state {
            MatrixState::Closed => Err(CudaBf16Error::Closed),
            MatrixState::Poisoned => Err(CudaBf16Error::Poisoned),
            MatrixState::Empty | MatrixState::Ready => {
                if Arc::strong_count(&self.allocation) != 1 {
                    return Err(CudaBf16Error::InUse);
                }
                if self.shape.bytes > self.allocation.allocated_bytes() {
                    return Err(CudaBf16Error::Invalid("matrix exceeds allocation"));
                }
                self.allocation.pointer()?;
                Ok(())
            }
        }
    }
    /// Initialize the complete matrix with native BF16 bytes, without conversion.
    /// Invalid lengths do not mutate the matrix. Transfer errors poison it.
    /// # Errors
    /// Rejects invalid length, closed/poisoned/in-use storage or transfer failure.
    pub fn upload(&mut self, bytes: &[u8]) -> Result<(), CudaBf16Error> {
        self.check_open()?;
        if bytes.len() as u64 != self.shape.bytes {
            return Err(CudaBf16Error::Invalid("matrix upload length mismatch"));
        }
        let allocation = Arc::get_mut(&mut self.allocation).ok_or(CudaBf16Error::InUse)?;
        self.state = MatrixState::Poisoned;
        allocation.write_exact(0, self.shape.bytes, &mut Cursor::new(bytes))?;
        self.state = MatrixState::Ready;
        Ok(())
    }
    /// Completion-safe diagnostic readback, bounded by the prepared staging size.
    /// # Errors
    /// Rejects uninitialized/in-use/closed storage, invalid ranges or transfer errors.
    pub fn read_chunk(&mut self, offset: u64, len: usize) -> Result<Vec<u8>, CudaBf16Error> {
        self.check_open()?;
        if !self.ready() {
            return Err(CudaBf16Error::NotReady);
        }
        if offset
            .checked_add(len as u64)
            .is_none_or(|end| end > self.shape.bytes)
        {
            return Err(CudaBf16Error::Invalid("matrix read exceeds logical bytes"));
        }
        let allocation = Arc::get_mut(&mut self.allocation).ok_or(CudaBf16Error::InUse)?;
        allocation.check_read(offset, len)?;
        // A failed read may
        // leave its owner-held staging in use; never retry/reuse a poisoned matrix.
        self.state = MatrixState::Poisoned;
        let bytes = allocation.read_chunk(offset, len)?;
        self.state = MatrixState::Ready;
        Ok(bytes)
    }
    /// Drain and acknowledge physical release before refunding this matrix's grant.
    /// # Errors
    /// Rejects retained accesses, failed drains, or uncertain allocation release.
    pub fn close(&mut self) -> Result<(), CudaBf16Error> {
        if self.state == MatrixState::Closed {
            return Ok(());
        }
        let allocation = Arc::get_mut(&mut self.allocation).ok_or(CudaBf16Error::InUse)?;
        self.state = MatrixState::Poisoned;
        allocation.close()?;
        self.state = MatrixState::Closed;
        Ok(())
    }
}

// Each variant owns exactly the compiled program and parameters it can access.
enum Operation {
    Blas {
        program: Arc<Blas>,
        _weights: DeviceAllocation,
    },
    Rms {
        program: Arc<rms::RmsKernels>,
        _scale: Option<DeviceAllocation>,
    },
    Rotary {
        program: Arc<rotary::RotaryKernels>,
        _cos: DeviceAllocation,
        _sin: DeviceAllocation,
    },
}
impl Operation {
    fn stream(&self) -> &Arc<CudaStream> {
        match self {
            Self::Blas { program, .. } => program.stream(),
            Self::Rms { program, .. } => &program.stream,
            Self::Rotary { program, .. } => &program.stream,
        }
    }
}
/// Each access shape retains exactly the resources its launch can touch.
enum Access {
    Matrices {
        operation: Operation,
        _input: DeviceAllocation,
        _output: DeviceAllocation,
    },
    Factors {
        kernels: Arc<factors::FactorKernels>,
        _inverse_frequencies: DeviceAllocation,
        _positions: DeviceAllocation,
        _cosine: DeviceAllocation,
        _sine: DeviceAllocation,
    },
}
impl Access {
    fn stream(&self) -> &Arc<CudaStream> {
        match self {
            Self::Matrices { operation, .. } => operation.stream(),
            Self::Factors { kernels, .. } => &kernels.stream,
        }
    }
}
fn drain(access: &Access) -> Result<(), CudaBf16Error> {
    access
        .stream()
        .synchronize()
        .map_err(|e| error("BF16 compute completion", e))
}
enum Retirement {
    Preparation(Box<Blas>),
    Execution(Pending<Access>),
}

/// A validation/library error plus any ownership needed after possible enqueue.
/// Dropping an unretired error drains or quarantines its complete access envelope.
pub struct CudaBf16ComputeError {
    cause: CudaBf16Error,
    retirement: Option<Retirement>,
}
impl CudaBf16ComputeError {
    #[must_use]
    pub const fn cause(&self) -> &CudaBf16Error {
        &self.cause
    }
    #[must_use]
    pub const fn retirement_pending(&self) -> bool {
        self.retirement.is_some()
    }
    /// Establish completion and release retained access guards. Matrix/weight
    /// allocations remain charged while their respective owners are still live.
    /// Failed cleanup preserves this same retirement owner for another attempt.
    /// # Errors
    /// Returns drain or preparation handle/workspace cleanup errors.
    pub fn retire(&mut self) -> Result<(), CudaBf16Error> {
        match self.retirement.as_mut() {
            Some(Retirement::Preparation(owner)) => owner.close()?,
            Some(Retirement::Execution(pending)) => pending.retire()?,
            None => {}
        }
        self.retirement = None;
        Ok(())
    }
}
impl From<CudaBf16Error> for CudaBf16ComputeError {
    fn from(cause: CudaBf16Error) -> Self {
        Self {
            cause,
            retirement: None,
        }
    }
}
impl fmt::Display for CudaBf16ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl fmt::Debug for CudaBf16ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBf16ComputeError")
            .field("cause", &self.cause)
            .field("retirement_pending", &self.retirement_pending())
            .finish_non_exhaustive()
    }
}
impl Error for CudaBf16ComputeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.cause)
    }
}

/// Prepared cuBLAS BF16 linear execution, with an explicitly charged workspace.
/// Uses FP32 accumulation and disables reduced-precision reductions. A successful
/// `linear_into` includes stream completion; this is not asynchronous submission
/// or whole-model inference, nor a performance-selected algorithm promise.
pub struct CudaBf16Gemm {
    blas: Arc<Blas>,
    poisoned: bool,
    closed: bool,
}
impl CudaBf16Gemm {
    /// Prepare one handle on the same stream object used by its operands. The
    /// workspace size is supplied explicitly, excluding library/context overhead.
    /// # Errors
    /// Returns validation/allocation/library errors, retaining partial preparation
    /// for explicit retirement when initialization has already acquired resources.
    pub fn new(
        stream: Arc<CudaStream>,
        pool: &Arc<BytePool>,
        workspace_bytes: usize,
    ) -> Result<Self, CudaBf16ComputeError> {
        let mut owner = Box::new(Blas::prepare(stream, pool, workspace_bytes)?);
        if let Err(cause) = owner.initialize() {
            return Err(CudaBf16ComputeError {
                cause,
                retirement: Some(Retirement::Preparation(owner)),
            });
        }
        Ok(Self {
            blas: Arc::from(owner),
            poisoned: false,
            closed: false,
        })
    }
    /// Compute `output[M,N] = input[M,K] × weights[slot][N,K]ᵀ` without repacking.
    /// All operands must be on this exact associated stream object. Output can be
    /// preallocated/reused; no matrix or device workspace is allocated per call.
    /// # Errors
    /// Validation does not mutate output. Enqueue/completion failures poison this
    /// executor and output and retain all accessed allocations until retirement.
    pub fn linear_into(
        &mut self,
        weights: &mut CudaBf16Weights,
        slot: usize,
        input: &mut CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        if self.closed {
            return Err(CudaBf16Error::Closed.into());
        }
        if self.poisoned {
            return Err(CudaBf16Error::Poisoned.into());
        }
        input.check_open()?;
        output.check_open()?;
        if !input.ready() {
            return Err(CudaBf16Error::NotReady.into());
        }
        let spec = weights.spec(slot).ok_or(CudaBf16Error::InvalidSlot(slot))?;
        input.shape.validate_linear(spec, output.shape)?;
        let expected_weight_bytes = spec
            .byte_size()
            .ok_or(CudaBf16Error::Invalid("weight byte overflow"))?;
        let (allocation, range) = weights.access(slot)?;
        if range.end.checked_sub(range.start) != Some(expected_weight_bytes)
            || range.end > allocation.allocated_bytes()
        {
            return Err(CudaBf16Error::Invalid("weight range exceeds allocation").into());
        }
        let stream = self.blas.stream();
        if !Arc::ptr_eq(stream, &allocation.transport().stream)
            || !Arc::ptr_eq(stream, &input.allocation.transport().stream)
            || !Arc::ptr_eq(stream, &output.allocation.transport().stream)
        {
            return Err(
                CudaBf16Error::Invalid("BF16 linear requires one associated stream").into(),
            );
        }
        let weight = allocation
            .pointer()?
            .checked_add(range.start)
            .ok_or(CudaBf16Error::Invalid("weight address overflow"))?;
        let a = usize::try_from(weight)
            .map_err(|_| CudaBf16Error::Invalid("weight pointer exceeds host width"))?;
        let b = usize::try_from(input.allocation.pointer()?)
            .map_err(|_| CudaBf16Error::Invalid("input pointer exceeds host width"))?;
        let c = usize::try_from(output.allocation.pointer()?)
            .map_err(|_| CudaBf16Error::Invalid("output pointer exceeds host width"))?;
        let handle = self.blas.handle()?;
        self.poisoned = true;
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| error("BF16 linear bind", e))?;
        let mut pending = Pending::new(
            Access::Matrices {
                operation: Operation::Blas {
                    program: Arc::clone(&self.blas),
                    _weights: allocation,
                },
                _input: Arc::clone(&input.allocation),
                _output: Arc::clone(&output.allocation),
            },
            drain,
        );
        // Poison BEFORE FFI, including unwinding. Only acknowledged completion
        // publishes output or permits another use of this executor.
        self.poisoned = true;
        output.state = MatrixState::Poisoned;
        // SAFETY: validated live, disjoint allocation ranges and exact dimensions;
        // row-major operands are transposed through column-major cuBLAS. Static
        // host scalars and every device resource remain alive through completion,
        // including failed/partial enqueue. Beta=0 never reads uninitialized output.
        let enqueue = unsafe {
            result::gemm_ex(
                handle,
                sys::cublasOperation_t::CUBLAS_OP_T,
                sys::cublasOperation_t::CUBLAS_OP_N,
                output.shape.blas_cols(),
                input.shape.blas_rows(),
                input.shape.blas_cols(),
                (&raw const ALPHA).cast(),
                a as *const _,
                sys::cudaDataType::CUDA_R_16BF,
                input.shape.blas_cols(),
                b as *const _,
                sys::cudaDataType::CUDA_R_16BF,
                input.shape.blas_cols(),
                (&raw const BETA).cast(),
                c as *mut _,
                sys::cudaDataType::CUDA_R_16BF,
                output.shape.blas_cols(),
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(|e| error("BF16 linear enqueue", e));
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
    /// Retire the handle and workspace. Retained failed accesses must settle first.
    /// # Errors
    /// Returns in-use, stream, library destruction or physical release errors.
    pub fn close(&mut self) -> Result<(), CudaBf16Error> {
        if self.closed {
            return Ok(());
        }
        let blas = Arc::get_mut(&mut self.blas).ok_or(CudaBf16Error::InUse)?;
        self.poisoned = true;
        blas.close()?;
        self.closed = true;
        Ok(())
    }
}
