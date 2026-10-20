//! Prepare BF16 rotary factors from model-owned FP32 inverse frequencies.
use super::{
    Access, CudaBf16ComputeError, CudaBf16Error, CudaBf16Matrix, DeviceAllocation, MatrixState,
    Pending, Retirement, drain, error,
};
use crate::{bf16_weights::cuda::CudaTransport, cuda_allocation::Allocation};
use cudarc::{
    driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};
use ribn_foundation::BytePool;
use std::{io::Cursor, sync::Arc};

/// Positions must convert to FP32 without rounding, as the pinned eager path does.
const MAX_EXACT_POSITION: u32 = 1 << 24;

pub(super) struct FactorKernels {
    pub(super) stream: Arc<CudaStream>,
    function: CudaFunction,
}

/// Native rotary factor preparation. The model supplies the padded FP32 inverse
/// frequencies and positions; this owner evaluates phase, cosine and sine on the
/// device, duplicates both head halves and stores BF16 round-to-nearest-even.
/// Scaling is exactly 1.0 in the pinned path, so no multiply is applied.
/// No device allocation occurs per call; code/context overhead is driver-owned.
pub struct CudaBf16RotaryFactors {
    kernels: Option<Arc<FactorKernels>>,
    inverse_frequencies: DeviceAllocation,
    positions: DeviceAllocation,
    half: u32,
    capacity_rows: usize,
    poisoned: bool,
}
impl CudaBf16RotaryFactors {
    /// Upload the complete half-head frequency vector once and reserve the position
    /// capacity. Frequencies may be zero (proportional padding) or negative, but
    /// must be finite. Compilation uses one stable associated stream.
    /// # Errors
    /// Rejects per-thread pseudo-streams, empty/oversized or non-finite frequencies,
    /// zero capacity, and compilation/allocation/upload failures.
    pub fn new(
        stream: Arc<CudaStream>,
        pool: &Arc<BytePool>,
        inverse_frequencies: &[f32],
        capacity_rows: usize,
    ) -> Result<Self, CudaBf16ComputeError> {
        if stream.cu_stream().addr() == 0x2 {
            return Err(CudaBf16Error::Invalid("rotary factors require a stable stream").into());
        }
        let half = u32::try_from(inverse_frequencies.len())
            .map_err(|_| CudaBf16Error::Invalid("rotary frequency count exceeds u32"))?;
        if half == 0 || half > i32::MAX as u32 / 2 {
            return Err(CudaBf16Error::Invalid("rotary half head out of range").into());
        }
        if !inverse_frequencies.iter().all(|value| value.is_finite()) {
            return Err(CudaBf16Error::Invalid("rotary frequencies must be finite").into());
        }
        let rows = u32::try_from(capacity_rows)
            .map_err(|_| CudaBf16Error::Invalid("rotary row capacity exceeds u32"))?;
        if rows == 0 {
            return Err(CudaBf16Error::Invalid("rotary row capacity must be positive").into());
        }
        let frequency_bytes = u64::from(half) * 4;
        let position_bytes = u64::from(rows) * 4;
        let ptx = compile_ptx(concat!(
            include_str!("bf16.cuh"),
            include_str!("factors.cu")
        ))
        .map_err(|e| error("BF16 rotary factors compilation", e))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| error("BF16 rotary factors module", e))?;
        let function = module
            .load_function("bf16_rotary_factors")
            .map_err(|e| error("BF16 rotary factors function", e))?;
        let staging = |bytes: u64| usize::try_from(bytes.min(1 << 20)).unwrap_or(1 << 20);
        let frequency_bytes_host: Vec<_> = inverse_frequencies
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut frequency_buffer = Allocation::allocate(
            frequency_bytes,
            CudaTransport {
                stream: Arc::clone(&stream),
            },
            pool,
            staging(frequency_bytes),
        )?;
        Arc::get_mut(&mut frequency_buffer)
            .ok_or(CudaBf16Error::InUse)?
            .write_exact(0, frequency_bytes, &mut Cursor::new(&frequency_bytes_host))?;
        let positions = Allocation::allocate(
            position_bytes,
            CudaTransport {
                stream: Arc::clone(&stream),
            },
            pool,
            staging(position_bytes),
        )?;
        Ok(Self {
            kernels: Some(Arc::new(FactorKernels { stream, function })),
            inverse_frequencies: frequency_buffer,
            positions,
            half,
            capacity_rows: rows as usize,
            poisoned: false,
        })
    }

    /// Width of the produced factor rows: `2 × inverse frequencies`.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.half as usize * 2
    }

    #[must_use]
    pub const fn capacity_rows(&self) -> usize {
        self.capacity_rows
    }

    /// Evaluate cosine/sine for `positions` into matching `[rows, width]` matrices,
    /// duplicating the half-head vector across both halves. Frequencies were
    /// validated at construction; positions must convert to FP32 exactly. Factors
    /// are outputs: they need not contain earlier results. Every matrix must use
    /// this exact stream object. Successful return includes completion.
    /// # Errors
    /// Whole-call validation preserves factors. Possible enqueue/completion failure
    /// poisons this preparer and both outputs and retains frequencies, positions and
    /// compiled code until retirement.
    pub fn prepare_into(
        &mut self,
        positions: &[u32],
        cosine: &mut CudaBf16Matrix,
        sine: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        let kernels = self.kernels.as_ref().ok_or(CudaBf16Error::Closed)?;
        if self.poisoned {
            return Err(CudaBf16Error::Poisoned.into());
        }
        let rows = u32::try_from(positions.len())
            .map_err(|_| CudaBf16Error::Invalid("rotary rows exceed u32"))?;
        if positions.is_empty() || positions.len() > self.capacity_rows {
            return Err(CudaBf16Error::Invalid("rotary positions exceed capacity").into());
        }
        if positions
            .iter()
            .any(|&position| position >= MAX_EXACT_POSITION)
        {
            return Err(CudaBf16Error::Invalid("rotary position is not FP32-exact").into());
        }
        for matrix in [&*cosine, &*sine] {
            matrix.check_open()?;
            if !Arc::ptr_eq(&kernels.stream, &matrix.allocation.transport().stream) {
                return Err(
                    CudaBf16Error::Invalid("rotary factors require one associated stream").into(),
                );
            }
        }
        // One factor row per position: rows bound the grid and the position reads.
        if u32::try_from(cosine.shape.rows) != Ok(rows) {
            return Err(
                CudaBf16Error::Invalid("rotary factors require one row per position").into(),
            );
        }
        let blocks = cosine.shape.validate_factors(sine.shape, self.width())?;
        let c = cosine.allocation.pointer()?;
        let s = sine.allocation.pointer()?;
        let f = self.inverse_frequencies.pointer()?;
        let p = self.positions.pointer()?;
        let bytes: Vec<_> = positions
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        // Positions are re-uploaded each call and drained before launch; the buffer
        // remains owned until completion in case a later failure retains the access.
        // A failed upload leaves this preparer unusable but does not invalidate
        // factors that were never enqueued on.
        self.poisoned = true;
        let positions_allocation = Arc::get_mut(&mut self.positions).ok_or(CudaBf16Error::InUse)?;
        positions_allocation.write_exact(0, bytes.len() as u64, &mut Cursor::new(&bytes))?;
        cosine.state = MatrixState::Poisoned;
        sine.state = MatrixState::Poisoned;
        let mut pending = Pending::new(
            Access::Factors {
                kernels: Arc::clone(kernels),
                _inverse_frequencies: Arc::clone(&self.inverse_frequencies),
                _positions: Arc::clone(&self.positions),
                _cosine: Arc::clone(&cosine.allocation),
                _sine: Arc::clone(&sine.allocation),
            },
            drain,
        );
        let mut launch = kernels.stream.launch_builder(&kernels.function);
        launch
            .arg(&f)
            .arg(&p)
            .arg(&c)
            .arg(&s)
            .arg(&rows)
            .arg(&self.half);
        // SAFETY: whole-shape/live-range checks establish disjoint valid buffers and
        // a bounded grid over rows × half. Pending retains all resources and code
        // before any possible enqueue, including failed or partial launch.
        let enqueue = unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| error("BF16 rotary factors enqueue", e));
        if let Err(cause) = enqueue.and_then(|()| pending.retire()) {
            return Err(CudaBf16ComputeError {
                cause,
                retirement: Some(Retirement::Execution(pending)),
            });
        }
        cosine.state = MatrixState::Ready(cosine.shape.rows);
        sine.state = MatrixState::Ready(sine.shape.rows);
        self.poisoned = false;
        Ok(())
    }

    /// Drain, acknowledge and release the frequency/position buffers and code.
    /// # Errors
    /// Rejects retained accesses or a failed associated-stream drain/free.
    pub fn close(&mut self) -> Result<(), CudaBf16Error> {
        if let Some(kernels) = self.kernels.as_mut() {
            let kernels = Arc::get_mut(kernels).ok_or(CudaBf16Error::InUse)?;
            kernels
                .stream
                .synchronize()
                .map_err(|e| error("BF16 rotary factors close", e))?;
        }
        Arc::get_mut(&mut self.inverse_frequencies)
            .ok_or(CudaBf16Error::InUse)?
            .close()?;
        Arc::get_mut(&mut self.positions)
            .ok_or(CudaBf16Error::InUse)?
            .close()?;
        self.kernels = None;
        Ok(())
    }
}
