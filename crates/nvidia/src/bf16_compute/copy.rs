//! Completion-owned device copies into incrementally initialized cache rows.
use super::{
    Access, CudaBf16ComputeError, CudaBf16Error, CudaBf16Matrix, Dispatch, MatrixState, error,
};
use cudarc::driver::result;
use std::{ops::Range, sync::Arc};

impl CudaBf16Matrix {
    /// Copy `source_rows` into this matrix starting at `start_row`, without
    /// conversion or host staging. The source range must be nonempty and initialized;
    /// other source rows need not be ready. Columns must match and both allocations
    /// must use the same associated stream. Success includes completion.
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
        source_rows: Range<usize>,
    ) -> Result<(), CudaBf16ComputeError> {
        self.copy_rows(start_row, source, source_rows, Dispatch::Blocking)
    }
    pub(super) fn copy_rows(
        &mut self,
        start_row: usize,
        source: &Self,
        source_rows: Range<usize>,
        dispatch: Dispatch<'_>,
    ) -> Result<(), CudaBf16ComputeError> {
        dispatch.matrix(self)?;
        dispatch.matrix(source)?;
        let copy = self.shape.validate_copy_rows(
            source.shape,
            source_rows,
            dispatch.rows(source),
            start_row,
            dispatch.rows(self),
        )?;
        let stream = &source.allocation.transport().stream;
        if !Arc::ptr_eq(stream, &self.allocation.transport().stream) {
            return Err(
                CudaBf16Error::Invalid("matrix copy requires one associated stream").into(),
            );
        }
        let src = source
            .allocation
            .pointer()?
            .checked_add(copy.source_offset)
            .ok_or(CudaBf16Error::Invalid(
                "matrix copy source address overflow",
            ))?;
        let dst = self
            .allocation
            .pointer()?
            .checked_add(copy.destination_offset)
            .ok_or(CudaBf16Error::Invalid(
                "matrix copy destination address overflow",
            ))?;
        let bytes = usize::try_from(copy.bytes)
            .map_err(|_| CudaBf16Error::Invalid("matrix copy exceeds host width"))?;
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| error("BF16 copy bind", e))?;
        let pending = dispatch.retain(Access::Copy {
            source: Arc::clone(&source.allocation),
            _destination: Arc::clone(&self.allocation),
        })?;
        self.state = MatrixState::Poisoned;
        // SAFETY: validated disjoint, live same-stream allocations, matching row
        // widths and in-bounds byte ranges. Exclusive destination access prevents
        // aliasing. The pending owner retains both allocations and their stream/
        // context before enqueue, including failure or unwind, until proven drain.
        let enqueue = unsafe { result::memcpy_dtod_async(dst, src, bytes, stream.cu_stream()) }
            .map_err(|e| error("BF16 copy enqueue", e));
        self.state = pending.finish(enqueue)?.matrix(copy.initialized);
        Ok(())
    }
}
