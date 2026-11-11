//! Bounded same-stream commands with one terminal completion owner.
use super::{
    Access, AttentionGeometry, AttentionSpan, CudaBf16Attention, CudaBf16ComputeError,
    CudaBf16Elementwise, CudaBf16Error, CudaBf16Gemm, CudaBf16Matrix, CudaBf16RmsNorm,
    CudaBf16RotaryApply, CudaBf16Stream, CudaBf16Weights, DeviceAllocation, MatrixState, Pending,
    Retirement,
    elementwise::{Binary, Scalar, Unary},
    error,
};
use cudarc::driver::{CudaEvent, CudaStream, result, sys};
use std::{
    collections::HashMap,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

const RECORDING: u8 = 0;
const COMPLETE: u8 = 1;
const FAILED: u8 = 2;

pub(super) struct Progress(AtomicU8);
impl Progress {
    fn new() -> Self {
        Self(AtomicU8::new(RECORDING))
    }
    pub(super) fn complete(&self) -> bool {
        self.0.load(Ordering::Acquire) == COMPLETE
    }
    pub(super) fn recording(&self) -> bool {
        self.0.load(Ordering::Acquire) == RECORDING
    }
    fn fail(&self) {
        self.0.store(FAILED, Ordering::Release);
    }
}

// Borrowed submissions can survive catch_unwind. Fault shared readiness before
// unwinding returns control, even if no new access guard was installed.
struct FailureGuard<'a> {
    progress: &'a Progress,
    acknowledged: bool,
}
impl<'a> FailureGuard<'a> {
    fn new(progress: &'a Progress) -> Self {
        Self {
            progress,
            acknowledged: false,
        }
    }
    fn acknowledge(mut self) {
        self.acknowledged = true;
    }
}
impl Drop for FailureGuard<'_> {
    fn drop(&mut self) {
        if !self.acknowledged {
            self.progress.fail();
        }
    }
}

pub(super) enum ProgramState {
    Ready,
    Ordered(Arc<Progress>),
    Poisoned,
}
pub(super) enum Acknowledgement {
    Complete,
    Ordered(Arc<Progress>),
}
impl Acknowledgement {
    pub(super) fn program(&self) -> ProgramState {
        match self {
            Self::Complete => ProgramState::Ready,
            Self::Ordered(progress) => ProgramState::Ordered(Arc::clone(progress)),
        }
    }
    pub(super) fn matrix(&self, rows: usize) -> MatrixState {
        match self {
            Self::Complete => MatrixState::Ready(rows),
            Self::Ordered(progress) => MatrixState::Ordered {
                rows,
                progress: Arc::clone(progress),
            },
        }
    }
}

pub(super) struct Resources {
    stream: Arc<CudaStream>,
    event: CudaEvent,
    accesses: Vec<Access>,
}
fn drain(resources: &Resources) -> Result<(), CudaBf16Error> {
    resources
        .stream
        .synchronize()
        .map_err(|e| error("BF16 command retirement", e))
}

/// A finite sequence of dependent native BF16 operations on one exact stream.
/// Queued data is readable only by this transaction, not by staging or another
/// command owner. Every allocation and program stays retained until explicit
/// retirement successfully drains the stream, even after the terminal event completes.
/// This is not an operator graph, capture API, allocator or scheduler. Only an
/// opaque stream domain can create commands; a caller-controlled raw stream could
/// capture commands without executing them and invalidate completion ownership.
///
/// ```compile_fail
/// use cudarc::driver::CudaStream;
/// use engine_nvidia::CudaBf16Commands;
/// use std::sync::Arc;
/// fn cannot_capture_commands(stream: Arc<CudaStream>) {
///     let _ = CudaBf16Commands::new(stream, 1);
/// }
/// ```
pub struct CudaBf16Commands {
    pending: Option<Pending<Resources>>,
    progress: Arc<Progress>,
    references: HashMap<usize, usize>,
    limit: usize,
}
impl CudaBf16Commands {
    /// Reserve bounded host command metadata and create the terminal event before
    /// any operation can enqueue. Device/context metadata is outside the byte pool.
    /// # Errors
    /// Rejects zero/overflowing bounds, metadata allocation or event creation failure.
    pub fn new(
        stream: &CudaBf16Stream,
        max_operations: usize,
    ) -> Result<Self, CudaBf16ComputeError> {
        let references = max_operations
            .checked_mul(5)
            .filter(|_| max_operations > 0)
            .ok_or(CudaBf16Error::Invalid("invalid BF16 command bound"))?;
        let stream = Arc::clone(&stream.stream);
        let mut accesses = Vec::new();
        accesses
            .try_reserve_exact(max_operations)
            .map_err(|e| error("BF16 command metadata", e))?;
        let mut retained = HashMap::new();
        retained
            .try_reserve(references)
            .map_err(|e| error("BF16 access metadata", e))?;
        let event = stream
            .context()
            .new_event(None)
            .map_err(|e| error("BF16 command event", e))?;
        Ok(Self {
            pending: Some(Pending::new(
                Resources {
                    stream,
                    event,
                    accesses,
                },
                drain,
            )),
            progress: Arc::new(Progress::new()),
            references: retained,
            limit: max_operations,
        })
    }

    fn execute(
        &mut self,
        operation: impl FnOnce(&mut Self) -> Result<(), CudaBf16ComputeError>,
    ) -> Result<(), CudaBf16ComputeError> {
        if !self.progress.recording() || self.pending.is_none() {
            return Err(CudaBf16Error::Poisoned.into());
        }
        let progress = Arc::clone(&self.progress);
        let guard = FailureGuard::new(&progress);
        let result = match operation(self) {
            Ok(()) => Ok(()),
            Err(cause) => Err(self.fail(cause)),
        };
        guard.acknowledge();
        result
    }
    fn fail(&mut self, mut cause: CudaBf16ComputeError) -> CudaBf16ComputeError {
        self.progress.fail();
        // Queued primitive errors have no independent access envelope: all
        // earlier and current accesses belong to this one transaction.
        if cause.retirement.is_none() {
            cause.retirement = self.pending.take().map(Retirement::Commands);
        }
        cause
    }
    fn permits(&self, allocation: &DeviceAllocation) -> bool {
        let retained = self
            .references
            .get(&(Arc::as_ptr(allocation) as usize))
            .copied()
            .unwrap_or(0);
        Arc::strong_count(allocation) == retained + 1
    }
    fn retain(&mut self, access: Access) -> Result<(), CudaBf16Error> {
        let resources = self
            .pending
            .as_mut()
            .and_then(Pending::resources_mut)
            .ok_or(CudaBf16Error::Poisoned)?;
        if resources.accesses.len() == self.limit {
            return Err(CudaBf16Error::Invalid("BF16 command bound exceeded"));
        }
        if !Arc::ptr_eq(&resources.stream, access.stream()) {
            return Err(CudaBf16Error::Invalid(
                "commands require one associated stream",
            ));
        }
        access.visit_allocations(|allocation| {
            *self
                .references
                .entry(Arc::as_ptr(allocation) as usize)
                .or_default() += 1;
        });
        resources.accesses.push(access);
        Ok(())
    }

    /// Enqueue a shared row-major projection. Successful return is ordered, not
    /// completed; validation/partial-enqueue errors fault the entire transaction.
    /// # Errors
    /// Returns the primitive diagnostic with all prior/current accesses retained.
    pub fn linear_into(
        &mut self,
        gemm: &mut CudaBf16Gemm,
        weights: &mut CudaBf16Weights,
        slot: usize,
        input: &mut CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.execute(|commands| {
            gemm.linear(weights, slot, input, output, Dispatch::Ordered(commands))
        })
    }
    /// Enqueue grouped RMS; see [`CudaBf16RmsNorm::normalize_into`].
    /// # Errors
    /// Faults the transaction on validation or enqueue failure.
    pub fn normalize_into(
        &mut self,
        rms: &mut CudaBf16RmsNorm,
        scale: Option<(&mut CudaBf16Weights, usize)>,
        width: usize,
        epsilon: f32,
        input: &mut CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.execute(|commands| {
            rms.normalize(
                scale,
                width,
                epsilon,
                input,
                output,
                Dispatch::Ordered(commands),
            )
        })
    }
    /// Enqueue row copies, including contiguous cache-prefix extension.
    /// # Errors
    /// Faults the transaction on invalid rows, ownership or enqueue failure.
    pub fn copy_rows(
        &mut self,
        destination: &mut CudaBf16Matrix,
        start_row: usize,
        source: &CudaBf16Matrix,
        source_rows: Range<usize>,
    ) -> Result<(), CudaBf16ComputeError> {
        self.execute(|commands| {
            destination.copy_rows(start_row, source, source_rows, Dispatch::Ordered(commands))
        })
    }
    /// Enqueue addition; see [`CudaBf16Elementwise::add_into`].
    /// # Errors
    /// Faults the transaction on validation or enqueue failure.
    pub fn add_into(
        &mut self,
        elementwise: &mut CudaBf16Elementwise,
        left: &CudaBf16Matrix,
        right: &CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.execute(|commands| {
            elementwise.binary_into(
                Binary::Add,
                left,
                right,
                output,
                Dispatch::Ordered(commands),
            )
        })
    }
    /// Enqueue multiplication; see [`CudaBf16Elementwise::multiply_into`].
    /// # Errors
    /// Faults the transaction on validation or enqueue failure.
    pub fn multiply_into(
        &mut self,
        elementwise: &mut CudaBf16Elementwise,
        left: &CudaBf16Matrix,
        right: &CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.execute(|commands| {
            elementwise.binary_into(
                Binary::Multiply,
                left,
                right,
                output,
                Dispatch::Ordered(commands),
            )
        })
    }
    /// Enqueue the two-rounding PLE combine.
    /// # Errors
    /// Faults the transaction on validation or enqueue failure.
    pub fn ple_combine_into(
        &mut self,
        elementwise: &mut CudaBf16Elementwise,
        context: &CudaBf16Matrix,
        identity: &CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.execute(|commands| {
            elementwise.binary_into(
                Binary::PleCombine,
                context,
                identity,
                output,
                Dispatch::Ordered(commands),
            )
        })
    }
    /// Enqueue scaling; see [`CudaBf16Elementwise::scale_into`].
    /// # Errors
    /// Faults the transaction on invalid operands/factor or enqueue failure.
    pub fn scale_into(
        &mut self,
        elementwise: &mut CudaBf16Elementwise,
        input: &CudaBf16Matrix,
        factor: f32,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.execute(|commands| {
            elementwise.scalar_into(
                Scalar::Scale(factor),
                input,
                output,
                Dispatch::Ordered(commands),
            )
        })
    }
    /// Enqueue tanh GELU.
    /// # Errors
    /// Faults the transaction on validation or enqueue failure.
    pub fn gelu_tanh_into(
        &mut self,
        elementwise: &mut CudaBf16Elementwise,
        input: &CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.execute(|commands| {
            elementwise.unary_into(Unary::Gelu, input, output, Dispatch::Ordered(commands))
        })
    }
    /// Enqueue BF16 soft cap.
    /// # Errors
    /// Faults the transaction on invalid operands/cap or enqueue failure.
    pub fn softcap_into(
        &mut self,
        elementwise: &mut CudaBf16Elementwise,
        input: &CudaBf16Matrix,
        cap: f32,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.execute(|commands| {
            elementwise.scalar_into(
                Scalar::Softcap(cap),
                input,
                output,
                Dispatch::Ordered(commands),
            )
        })
    }
    /// Enqueue rotation using already staged/prepared factors.
    /// # Errors
    /// Faults the transaction on validation or enqueue failure.
    pub fn rotary_into(
        &mut self,
        rotary: &mut CudaBf16RotaryApply,
        cos: &mut CudaBf16Matrix,
        sin: &mut CudaBf16Matrix,
        input: &mut CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.execute(|commands| rotary.apply(cos, sin, input, output, Dispatch::Ordered(commands)))
    }
    /// Enqueue attention against the member's exact initialized history.
    /// # Errors
    /// Faults the transaction on validation or any partial attention enqueue failure.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit attention operands and their command owner"
    )]
    pub fn attend_into(
        &mut self,
        attention: &mut CudaBf16Attention,
        query: &CudaBf16Matrix,
        keys: &CudaBf16Matrix,
        values: &CudaBf16Matrix,
        geometry: AttentionGeometry,
        span: AttentionSpan,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.execute(|commands| {
            attention.attend(
                query,
                keys,
                values,
                geometry,
                span,
                output,
                Dispatch::Ordered(commands),
            )
        })
    }

    /// Abandon queued results and explicitly drain their accesses without submission.
    /// A model-level validation failure can retain this owner and retry cleanup.
    /// Successful cleanup never publishes abandoned outputs.
    /// # Errors
    /// Failed drain retains the complete access set for another attempt.
    pub fn retire(&mut self) -> Result<(), CudaBf16Error> {
        self.progress.fail();
        if let Some(pending) = self.pending.as_mut() {
            pending.retire()?;
            self.pending = None;
        }
        Ok(())
    }

    /// Record the sole terminal event after all commands. Does not wait for it.
    /// # Errors
    /// Empty/faulted transactions or event-record failures retain their whole envelope.
    pub fn submit(mut self) -> Result<CudaBf16Submission, CudaBf16ComputeError> {
        let result = (|| {
            if !self.progress.recording() {
                return Err(CudaBf16Error::Poisoned);
            }
            let resources = self
                .pending
                .as_ref()
                .and_then(Pending::resources)
                .ok_or(CudaBf16Error::Poisoned)?;
            if resources.accesses.is_empty() {
                return Err(CudaBf16Error::Invalid("empty BF16 transaction"));
            }
            resources
                .event
                .record(&resources.stream)
                .map_err(|e| error("BF16 terminal event record", e))
        })();
        if let Err(cause) = result {
            return Err(self.fail(cause.into()));
        }
        Ok(CudaBf16Submission {
            pending: self.pending.take(),
            progress: Arc::clone(&self.progress),
        })
    }
}
impl Drop for CudaBf16Commands {
    fn drop(&mut self) {
        if self.pending.is_some() {
            self.progress.fail();
        }
        // Pending drains or quarantines before accessed owners can disappear.
    }
}

/// Completion and retryable retirement for one submitted BF16 transaction.
/// Dropping an unacknowledged submission poisons its queued data and drains or
/// quarantines all accesses. A failed completion never republishes on later cleanup.
pub struct CudaBf16Submission {
    pending: Option<Pending<Resources>>,
    progress: Arc<Progress>,
}
impl CudaBf16Submission {
    /// Nonblocking terminal query; only `CUDA_ERROR_NOT_READY` means still pending.
    /// Even `true` does not release accesses or publish output: call the explicitly
    /// blocking [`Self::retire`] to acknowledge the transaction. Querying never
    /// synchronizes later work on this domain's stream.
    /// # Errors
    /// Other driver errors fault queued data and retain resources for `retire`.
    pub fn poll(&mut self) -> Result<bool, CudaBf16Error> {
        if self.progress.complete() {
            return Ok(true);
        }
        if !self.progress.recording() {
            return Err(CudaBf16Error::Poisoned);
        }
        let resources = self
            .pending
            .as_ref()
            .and_then(Pending::resources)
            .ok_or(CudaBf16Error::Poisoned)?;
        let guard = FailureGuard::new(&self.progress);
        resources
            .stream
            .context()
            .bind_to_thread()
            .map_err(|e| error("BF16 terminal query bind", e))?;
        // SAFETY: Pending owns the recorded event and its live context. No other
        // owner can destroy or re-record this transaction's terminal event.
        match unsafe { result::event::query(resources.event.cu_event()) } {
            Ok(()) => {
                guard.acknowledge();
                Ok(true)
            }
            Err(result::DriverError(sys::cudaError_enum::CUDA_ERROR_NOT_READY)) => {
                guard.acknowledge();
                Ok(false)
            }
            Err(source) => {
                self.progress.fail();
                Err(error("BF16 terminal query", source))
            }
        }
    }
    /// Drain and release access guards. Successful nonfailed execution publishes
    /// queued matrix initialization; cleanup after failure never revives results.
    /// This explicitly blocking acknowledgement drains the associated stream,
    /// including later work if the caller submitted any. A depth-one model owner
    /// must acknowledge before enqueueing its next transaction.
    /// # Errors
    /// Failed drain retains this same owner and charges for another attempt.
    pub fn retire(&mut self) -> Result<(), CudaBf16Error> {
        if let Some(pending) = self.pending.as_mut() {
            let guard = FailureGuard::new(&self.progress);
            pending.retire()?;
            self.pending = None;
            if self.progress.recording() {
                self.progress.0.store(COMPLETE, Ordering::Release);
            }
            guard.acknowledge();
        }
        Ok(())
    }
}
impl Drop for CudaBf16Submission {
    fn drop(&mut self) {
        if self.pending.is_some() {
            self.progress.fail();
        }
    }
}

pub(super) enum Dispatch<'a> {
    Blocking,
    Ordered(&'a mut CudaBf16Commands),
}
impl<'a> Dispatch<'a> {
    pub(super) fn program(&self, state: &ProgramState) -> Result<(), CudaBf16Error> {
        match state {
            ProgramState::Ready => Ok(()),
            ProgramState::Ordered(progress) if progress.complete() => Ok(()),
            ProgramState::Ordered(progress) => match self {
                Self::Ordered(commands)
                    if Arc::ptr_eq(progress, &commands.progress) && progress.recording() =>
                {
                    Ok(())
                }
                _ if progress.recording() => Err(CudaBf16Error::InUse),
                _ => Err(CudaBf16Error::Poisoned),
            },
            ProgramState::Poisoned => Err(CudaBf16Error::Poisoned),
        }
    }
    pub(super) fn matrix(&self, matrix: &CudaBf16Matrix) -> Result<(), CudaBf16Error> {
        if let Self::Ordered(commands) = self {
            match &matrix.state {
                MatrixState::Closed => return Err(CudaBf16Error::Closed),
                MatrixState::Poisoned => return Err(CudaBf16Error::Poisoned),
                MatrixState::Ordered { progress, .. } if !progress.complete() => {
                    if !progress.recording() {
                        return Err(CudaBf16Error::Poisoned);
                    }
                    if !Arc::ptr_eq(progress, &commands.progress) {
                        return Err(CudaBf16Error::InUse);
                    }
                }
                _ => {}
            }
            let resources = commands
                .pending
                .as_ref()
                .and_then(Pending::resources)
                .ok_or(CudaBf16Error::Poisoned)?;
            if !Arc::ptr_eq(&resources.stream, &matrix.allocation.transport().stream) {
                return Err(CudaBf16Error::Invalid(
                    "commands require one associated stream",
                ));
            }
            if !commands.permits(&matrix.allocation) {
                return Err(CudaBf16Error::InUse);
            }
            if matrix.shape.bytes > matrix.allocation.allocated_bytes() {
                return Err(CudaBf16Error::Invalid("matrix exceeds allocation"));
            }
            matrix.allocation.pointer()?;
            Ok(())
        } else {
            matrix.check_open()
        }
    }
    pub(super) fn rows(&self, matrix: &CudaBf16Matrix) -> usize {
        match (&matrix.state, self) {
            (MatrixState::Ordered { rows, progress }, Self::Ordered(commands))
                if Arc::ptr_eq(progress, &commands.progress) && progress.recording() =>
            {
                *rows
            }
            _ => matrix.ready_rows(),
        }
    }
    pub(super) fn ready(&self, matrix: &CudaBf16Matrix) -> bool {
        self.rows(matrix) == matrix.shape.rows
    }
    pub(super) fn weights(
        &self,
        weights: &CudaBf16Weights,
        slot: usize,
    ) -> Result<(DeviceAllocation, Range<u64>), CudaBf16Error> {
        match self {
            Self::Blocking => weights.access(slot),
            Self::Ordered(commands) => {
                weights.access_when(slot, |allocation| commands.permits(allocation))
            }
        }
    }
    pub(super) fn retain(self, access: Access) -> Result<Guard<'a>, CudaBf16Error> {
        match self {
            Self::Blocking => Ok(Guard::Blocking(Pending::new(access, super::drain))),
            Self::Ordered(commands) => {
                commands.retain(access)?;
                Ok(Guard::Ordered(EnqueueGuard {
                    commands,
                    acknowledged: false,
                }))
            }
        }
    }
}
pub(super) enum Guard<'a> {
    Blocking(Pending<Access>),
    Ordered(EnqueueGuard<'a>),
}
impl Guard<'_> {
    pub(super) fn finish(
        self,
        enqueue: Result<(), CudaBf16Error>,
    ) -> Result<Acknowledgement, CudaBf16ComputeError> {
        match self {
            Self::Blocking(mut pending) => {
                if let Err(cause) = enqueue.and_then(|()| pending.retire()) {
                    return Err(CudaBf16ComputeError {
                        cause,
                        retirement: Some(Retirement::Execution(pending)),
                    });
                }
                Ok(Acknowledgement::Complete)
            }
            Self::Ordered(mut guard) => {
                guard.acknowledged = true;
                if let Err(cause) = enqueue {
                    guard.commands.progress.fail();
                    return Err(cause.into());
                }
                Ok(Acknowledgement::Ordered(Arc::clone(
                    &guard.commands.progress,
                )))
            }
        }
    }
}
pub(super) struct EnqueueGuard<'a> {
    commands: &'a mut CudaBf16Commands,
    acknowledged: bool,
}
impl Drop for EnqueueGuard<'_> {
    fn drop(&mut self) {
        if !self.acknowledged {
            self.commands.progress.fail();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA; serialize device qualification"]
    fn terminal_driver_error_and_failed_drain_retain_the_submission_until_retry() {
        use cudarc::driver::CudaContext;
        use ribn_foundation::BytePool;

        let context = CudaContext::new(0).unwrap();
        let domain = CudaBf16Stream::new(&context).unwrap();
        let pool = BytePool::new(1024 * 1024).shared();
        let mut input = domain.matrix(2, 4, &pool, 16).unwrap();
        input.upload(&[0x3f; 16]).unwrap();
        let mut output = domain.matrix(2, 4, &pool, 16).unwrap();
        let mut commands = CudaBf16Commands::new(&domain, 1).unwrap();
        commands.copy_rows(&mut output, 0, &input, 0..2).unwrap();
        let mut submitted = commands.submit().unwrap();
        let captured = context.new_event(None).unwrap();
        // Internal fault injection only: the public domain cannot expose capture
        // or replace a terminal event. No resource-accessing command is captured.
        domain
            .stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        captured.record(&domain.stream).unwrap();
        let previous = std::mem::replace(
            &mut submitted
                .pending
                .as_mut()
                .unwrap()
                .resources_mut()
                .unwrap()
                .event,
            captured,
        );
        let query = submitted.poll();
        let drain = submitted.retire();
        let grant = pool.granted();
        drop(input);
        drop(output);
        let retained = pool.granted();
        let ended = domain.stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        drop(ended);
        let retry = submitted.retire();
        drop(previous);
        // Restore the private stream and attempt cleanup before assertions.
        assert!(query.is_err(), "a captured-event error is not NOT_READY");
        assert!(drain.is_err());
        assert_eq!(retained, grant);
        assert!(grant > 0);
        retry.unwrap();
        assert_eq!(pool.granted(), 0);
        assert!(
            submitted.poll().is_err(),
            "cleanup must not revive failed execution"
        );
    }

    #[test]
    #[ignore = "requires CUDA; serialize device qualification"]
    #[expect(
        clippy::too_many_lines,
        reason = "one real enqueue/failed drain/retry lifecycle, including capture cleanup before assertions"
    )]
    fn recording_failed_drain_retains_actual_resources_until_retry_without_publication() {
        use super::super::Operation;
        use cudarc::driver::CudaContext;
        use engine_core::{DataType, WeightTensorSpec};
        use ribn_foundation::BytePool;
        use std::io::Cursor;

        let context = CudaContext::new(0).unwrap();
        let domain = CudaBf16Stream::new(&context).unwrap();
        let pool = BytePool::new(1024 * 1024).shared();
        let spec = WeightTensorSpec::new("scale", vec![4], DataType::BF16).unwrap();
        let mut weights = domain.weights(vec![spec], &pool, 8).unwrap();
        weights
            .upload(0, &mut Cursor::new([0x80, 0x3f].repeat(4)))
            .unwrap();
        let mut rms = domain.rms_norm().unwrap();
        let mut input = domain.matrix(2, 4, &pool, 16).unwrap();
        input.upload(&[0x80, 0x3f].repeat(8)).unwrap();
        let mut output = domain.matrix(2, 4, &pool, 16).unwrap();
        output.upload(&[0x00, 0x40].repeat(8)).unwrap();
        let output_charge = output.allocated_bytes();
        let input_owner = Arc::downgrade(&input.allocation);
        let output_owner = Arc::downgrade(&output.allocation);
        let mut commands = CudaBf16Commands::new(&domain, 1).unwrap();
        commands
            .normalize_into(
                &mut rms,
                Some((&mut weights, 0)),
                4,
                1e-6,
                &mut input,
                &mut output,
            )
            .unwrap();
        // Observe the actual retained compiled program and scale allocation, not
        // dummy owners or a synthetic pool charge. Weak refs cannot extend them.
        let resources = commands.pending.as_ref().unwrap().resources().unwrap();
        let Access::Matrices {
            operation:
                Operation::Rms {
                    program,
                    _scale: Some(scale),
                },
            ..
        } = &resources.accesses[0]
        else {
            panic!("weighted RMS must retain its compiled program and scale");
        };
        let program_owner = Arc::downgrade(program);
        let scale_owner = Arc::downgrade(scale);
        let grant = pool.granted();
        assert!(grant > output_charge);
        assert!(!output.ready());

        // Private fault injection begins ONLY after real computation enqueues.
        // Nothing is enqueued during capture: no computation is captured/replayed,
        // no terminal event is replaced and opaque public authority is unchanged.
        domain
            .stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        let drain = commands.retire();
        let ended = domain.stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        // Synchronizing a captured stream may invalidate capture. cudarc returns
        // None for a null graph, or owns graph/exec destruction for Some(graph).
        let capture_ended = matches!(
            ended,
            Ok(_)
                | Err(result::DriverError(
                    sys::cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
                ))
        );
        drop(ended);
        let rejected = commands.copy_rows(&mut output, 0, &input, 0..2);
        drop(input);
        drop(weights);
        drop(rms);
        let retained = pool.granted();
        let owners_retained = input_owner.strong_count() > 0
            && scale_owner.strong_count() > 0
            && program_owner.strong_count() > 0;
        let invalid_before_retry = matches!(output.read_chunk(0, 2), Err(CudaBf16Error::Poisoned));
        let retry = commands.retire();
        // End capture, destroy any empty graph and retry before assertions, as in
        // the terminal-event fault test above; a failure must not strand capture.
        assert!(matches!(
            drain,
            Err(CudaBf16Error::Transport {
                operation: "BF16 command retirement",
                ..
            })
        ));
        assert!(capture_ended);
        let rejected = rejected.unwrap_err();
        assert!(matches!(rejected.cause(), CudaBf16Error::Poisoned));
        assert!(
            !rejected.retirement_pending(),
            "retry owner stays in commands"
        );
        assert_eq!(retained, grant, "failed drain cannot refund dropped owners");
        assert!(owners_retained);
        assert!(invalid_before_retry);
        retry.unwrap();
        assert!(commands.pending.is_none());
        assert_eq!(input_owner.strong_count(), 0);
        assert_eq!(scale_owner.strong_count(), 0);
        assert_eq!(program_owner.strong_count(), 0);
        assert_eq!(pool.granted(), output_charge);
        commands.retire().unwrap();
        assert_eq!(pool.granted(), output_charge);
        assert!(!output.ready());
        assert!(matches!(
            output.read_chunk(0, 2),
            Err(CudaBf16Error::Poisoned)
        ));
        assert!(matches!(
            output.upload(&[0x80, 0x3f].repeat(8)),
            Err(CudaBf16Error::Poisoned)
        ));
        let failure = commands
            .submit()
            .err()
            .expect("cleanup cannot revive recording");
        assert!(matches!(failure.cause(), CudaBf16Error::Poisoned));
        assert!(!failure.retirement_pending());
        drop(output);
        assert_eq!(output_owner.strong_count(), 0);
        assert_eq!(pool.granted(), 0);
    }

    #[test]
    fn failed_or_unwinding_attempt_cannot_publish_queued_data() {
        for unwind in [false, true] {
            let progress = Arc::new(Progress::new());
            let state = ProgramState::Ordered(Arc::clone(&progress));
            let attempt = std::panic::catch_unwind(|| {
                let _guard = FailureGuard::new(&progress);
                assert!(matches!(
                    Dispatch::Blocking.program(&state),
                    Err(CudaBf16Error::InUse)
                ));
                assert!(progress.recording());
                assert!(!unwind, "injected driver-wrapper unwind");
            });
            assert_eq!(attempt.is_err(), unwind);
            assert!(!progress.complete());
            assert!(!progress.recording());
            assert!(matches!(
                Dispatch::Blocking.program(&state),
                Err(CudaBf16Error::Poisoned)
            ));
        }
        let progress = Progress::new();
        FailureGuard::new(&progress).acknowledge();
        assert!(
            progress.recording(),
            "successful enqueue/query is not terminal publication"
        );
        assert!(!progress.complete());
    }
}
