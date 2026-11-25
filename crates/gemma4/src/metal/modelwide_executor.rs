//! Continuous-batching `GenerationExecutor` over `Gemma4MetalModelWide`.

use super::modelwide::{Gemma4MetalModelWide, PackedStepItem, PackedStepOutput};
use crate::paged_executor::{PagedModelWideBackend, PagedModelWideExecutor};
use crate::{Gemma4Config, Gemma4Error};
use ribn::{
    Admission, BatchItem, BatchPreparation, ExecutionError, ExecutorInfo, GenerationExecutor,
    PrefixCacheStats, RequestId, SequenceId, StepCompletion, SubmissionId, TokenRequest,
};

impl PagedModelWideBackend for Gemma4MetalModelWide {
    fn executor_name(&self) -> &'static str {
        "gemma4 metal modelwide"
    }

    fn config(&self) -> &Gemma4Config {
        self.config()
    }

    fn max_sequences(&self) -> usize {
        self.limits().max_sequences
    }

    fn max_positions(&self) -> usize {
        self.limits().max_positions
    }

    fn max_batch_tokens(&self) -> usize {
        self.limits().max_batch_tokens
    }

    fn active_slots(&self) -> usize {
        self.active_slots()
    }

    fn prefix_cache_stats(&self) -> PrefixCacheStats {
        self.prefix_cache_stats()
    }

    fn kv_pool_total_blocks(&self) -> usize {
        self.kv_pool_total_blocks()
    }

    fn kv_pool_pinned_blocks(&self) -> usize {
        self.kv_pool_pinned_blocks()
    }

    fn kv_pool_available_blocks(&self) -> usize {
        self.kv_pool_available_blocks()
    }

    fn peek_prefix_pages(&self, prompt: &[u32]) -> (usize, usize) {
        self.peek_prefix_pages(prompt)
    }

    fn allocate_slot_with_prefix(&mut self, prompt: &[u32]) -> Result<(usize, usize), Gemma4Error> {
        self.allocate_slot_with_prefix(prompt)
    }

    fn free_slot(&mut self, slot: usize) -> Result<(), Gemma4Error> {
        self.free_slot(slot)
    }

    fn slot_position(&self, slot: usize) -> Option<usize> {
        self.slot_position(slot)
    }

    fn slot_allocated_blocks(&self, slot: usize) -> usize {
        self.slot_allocated_blocks(slot)
    }

    fn try_reserve_slot_blocks(
        &mut self,
        slot: usize,
        end_pos: usize,
    ) -> Result<Option<usize>, Gemma4Error> {
        self.try_reserve_slot_blocks(slot, end_pos)
    }

    fn rollback_slot_blocks(&mut self, slot: usize, prev_len: usize) -> Result<(), Gemma4Error> {
        self.rollback_slot_blocks(slot, prev_len)
    }

    fn step_packed(
        &mut self,
        items: &[PackedStepItem<'_>],
    ) -> Result<PackedStepOutput<'_>, Gemma4Error> {
        self.step_packed(items)
    }

    fn retire_failure(&self) -> Result<(), Gemma4Error> {
        self.retire_failure()
    }
}

/// Model-wide continuous-batching Metal executor for Gemma 4.
///
/// Packs all offered prefill and decode rows into a single Metal command buffer,
/// keeping KV caches device-resident, admitting sequences on their initial prompt
/// envelope, negotiating KV block growth during pre-submit preparation, and opting
/// into runtime-managed completion-safe recomputation preemption under pool pressure.
pub struct Gemma4MetalModelWideExecutor(PagedModelWideExecutor<Gemma4MetalModelWide>);

impl Gemma4MetalModelWideExecutor {
    /// Bind a prepared `Gemma4MetalModelWide` to the `GenerationExecutor` contract.
    /// # Errors
    /// Rejects invalid limits or a faulted model.
    pub fn new(model: Gemma4MetalModelWide) -> Result<Self, Gemma4Error> {
        PagedModelWideExecutor::new(model).map(Self)
    }
}

impl GenerationExecutor for Gemma4MetalModelWideExecutor {
    fn info(&self) -> &ExecutorInfo {
        self.0.info()
    }

    fn supports_sampling(&self) -> bool {
        self.0.supports_sampling()
    }

    fn supports_preemption(&self) -> bool {
        self.0.supports_preemption()
    }

    fn prefix_cache_stats(&self) -> Option<PrefixCacheStats> {
        self.0.prefix_cache_stats()
    }

    fn admitted_prefix(&self, sequence: SequenceId) -> u32 {
        self.0.admitted_prefix(sequence)
    }

    fn admit(
        &mut self,
        request_id: RequestId,
        sequence: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        self.0.admit(request_id, sequence, request)
    }

    fn prepare(&mut self, batch: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
        self.0.prepare(batch)
    }

    fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
        self.0.abandon_preparation()
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        self.0.submit(batch)
    }

    fn poll(
        &mut self,
        submission: SubmissionId,
    ) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        self.0.poll(submission)
    }

    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        self.0.release(sequence)
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.0.synchronize()
    }
}
