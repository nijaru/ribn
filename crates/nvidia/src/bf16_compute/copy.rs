//! Completion-owned device copies into incrementally initialized cache rows.
use super::{
    Access, CudaBf16ComputeError, CudaBf16Error, CudaBf16Matrix, MatrixState, Pending, Retirement,
    drain, error,
};
use cudarc::driver::result;
use std::sync::Arc;

impl CudaBf16Matrix {
    /// Copy the complete source into this matrix starting at `start_row`, without
    /// conversion or host staging. Columns must match and both allocations must
    /// use the same associated stream. A successful return includes completion.
    ///
    /// Copies may overwrite initialized rows or extend their contiguous prefix,
    /// never leave a hole. Untouched rows are unchanged; later uninitialized rows
    /// remain unreadable. `ready()` means the entire matrix is initialized.
    /// # Errors
    /// Validation leaves this matrix unchanged. Possible enqueue/completion failure
    /// poisons it and retains both allocations until explicit retirement or quarantine.
    pub fn copy_rows_from(
        &mut self,
        start_row: usize,
        source: &Self,
    ) -> Result<(), CudaBf16ComputeError> {
        self.check_open()?;
        source.check_open()?;
        if !source.ready() {
            return Err(CudaBf16Error::NotReady.into());
        }
        let (offset, initialized) =
            self.shape
                .validate_copy_rows(source.shape, start_row, self.ready_rows())?;
        let stream = &source.allocation.transport().stream;
        if !Arc::ptr_eq(stream, &self.allocation.transport().stream) {
            return Err(
                CudaBf16Error::Invalid("matrix copy requires one associated stream").into(),
            );
        }
        let src = source.allocation.pointer()?;
        let dst = self
            .allocation
            .pointer()?
            .checked_add(offset)
            .ok_or(CudaBf16Error::Invalid("matrix copy address overflow"))?;
        let bytes = usize::try_from(source.shape.bytes)
            .map_err(|_| CudaBf16Error::Invalid("matrix copy exceeds host width"))?;
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| error("BF16 copy bind", e))?;
        let mut pending = Pending::new(
            Access::Copy {
                source: Arc::clone(&source.allocation),
                _destination: Arc::clone(&self.allocation),
            },
            drain,
        );
        self.state = MatrixState::Poisoned;
        // SAFETY: validated disjoint, live same-stream allocations, matching row
        // widths and in-bounds byte ranges. Exclusive destination access prevents
        // aliasing. The pending owner retains both allocations and their stream/
        // context before enqueue, including failure or unwind, until proven drain.
        let enqueue = unsafe { result::memcpy_dtod_async(dst, src, bytes, stream.cu_stream()) }
            .map_err(|e| error("BF16 copy enqueue", e));
        if let Err(cause) = enqueue.and_then(|()| pending.retire()) {
            return Err(CudaBf16ComputeError {
                cause,
                retirement: Some(Retirement::Execution(pending)),
            });
        }
        self.state = MatrixState::Ready(initialized);
        Ok(())
    }
}
