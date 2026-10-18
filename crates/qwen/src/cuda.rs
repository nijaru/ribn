use std::path::PathBuf;

use ribn::{
    Admission, BatchItem, ExecutionError, ExecutorInfo, GenerationExecutor, RequestId, SequenceId,
    StepCompletion, SubmissionId, TokenRequest,
};

use crate::execution::{QwenExecution, model_error};
use crate::loading::{self, Backend, MemoryReport, QwenLoadOptions};

/// Experimental prepared Qwen GGUF/CUDA text implementation.
///
/// The existing CUDA kernels are reused, but integration with the new scheduler
/// still requires hardware qualification. This is not an automatically qualified
/// execution variant, a generic model loader, or a multimodal implementation.
pub struct QwenCuda {
    info: ExecutorInfo,
    memory: MemoryReport,
    execution: Option<QwenExecution<Backend>>,
}

impl QwenCuda {
    /// Load one source's weights and tokenizer, prepare kernels, and reserve
    /// logical sequence capacity. The processor and executor are returned together
    /// so frontends do not reopen a pathname to assemble a different snapshot.
    ///
    /// # Errors
    /// Reports unsupported artifacts, invalid options, memory budget failure,
    /// or CUDA preparation errors before an engine can accept requests.
    pub fn load_gguf(
        path: impl Into<PathBuf>,
        options: QwenLoadOptions,
    ) -> Result<(Self, engine_gguf::GgufTokenizer), ExecutionError> {
        let (execution, memory, tokenizer) = loading::load(path.into(), options)?;
        Ok((
            Self {
                info: execution.info().clone(),
                execution: Some(execution),
                memory,
            },
            tokenizer,
        ))
    }

    #[must_use]
    pub const fn memory_report(&self) -> MemoryReport {
        self.memory
    }

    /// Observe future CUDA serving enqueue decisions without retaining the device
    /// owner. Opt-in; counters are not device timings or logical output usage.
    ///
    /// # Errors
    /// Returns an error if the device owner is unavailable.
    pub fn observe_dispatch(&mut self) -> Result<engine_nvidia::DispatchObserver, ExecutionError> {
        Ok(self
            .execution()?
            .backend
            .dispatcher_mut()
            .observe_dispatch())
    }

    fn execution(&mut self) -> Result<&mut QwenExecution<Backend>, ExecutionError> {
        self.execution
            .as_mut()
            .ok_or_else(|| ExecutionError::new("Qwen device owner is unavailable"))
    }
}

impl GenerationExecutor for QwenCuda {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }
    fn admit(
        &mut self,
        _request_id: RequestId,
        id: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        self.execution()?.admit(id, request)
    }
    fn prepare(&mut self, batch: &[BatchItem]) -> Result<ribn::BatchPreparation, ExecutionError> {
        self.execution()?.prepare(batch)
    }
    fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
        self.execution()?.abandon_preparation();
        Ok(())
    }
    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        self.execution()?.submit(batch)
    }
    fn poll(&mut self, id: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        self.execution()?.poll(id)
    }
    fn release(&mut self, id: SequenceId) -> Result<(), ExecutionError> {
        self.execution()?.release(id)
    }
    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        let execution = self.execution()?;
        // The dispatcher owns both the stream barrier and quarantine settlement.
        // Synchronizing its stream alone cannot authorize failed-drain release.
        execution
            .backend
            .dispatcher_mut()
            .synchronize()
            .map_err(model_error)?;
        execution.drain_after_barrier()
    }
}

impl Drop for QwenCuda {
    fn drop(&mut self) {
        if self.execution.is_some()
            && self.synchronize().is_err()
            && let Some(execution) = self.execution.take()
        {
            // No barrier proof: retain the single owner instead of releasing
            // allocations, pinned output, or streams still visible to the GPU.
            std::mem::forget(execution);
        }
    }
}
