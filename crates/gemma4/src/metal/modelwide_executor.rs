//! Continuous-batching `GenerationExecutor` over `Gemma4MetalModelWide`.

use super::kv_pool::KV_PAGE_TOKENS;
use super::modelwide::{Gemma4MetalModelWide, PackedStepItem};
use crate::Gemma4Error;
use crate::generation::{BoundRow, LogicalSequence, RowInput, bind_batch};
use ribn::{
    Admission, BatchItem, BatchPreparation, ExecutionError, ExecutorInfo, GenerationExecutor,
    GenerationLimits, Readiness, RequestId, Sampling, SequenceId, StepCompletion, SubmissionId,
    TokenRequest, init_rng_state, sample_bf16_logits,
};
use std::collections::HashMap;
use std::mem::ManuallyDrop;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Prepared,
    Submitted(SubmissionId),
    Failed,
    Closed,
}

struct SequenceEntry {
    slot: usize,
    logical: LogicalSequence,
    max_reach_pages: usize,
    sampling: Sampling,
    rng_state: u64,
}

struct Execution {
    model: Gemma4MetalModelWide,
    states: HashMap<SequenceId, SequenceEntry>,
    rows: Vec<BoundRow>,
    readiness: Readiness,
    phase: Phase,
    next: u64,
}

impl Execution {
    fn reserved_future_blocks(&self) -> usize {
        self.states
            .values()
            .map(|entry| {
                entry
                    .max_reach_pages
                    .saturating_sub(self.model.slot_allocated_blocks(entry.slot))
            })
            .sum()
    }

    fn close(&mut self) -> Result<(), ExecutionError> {
        if self.phase == Phase::Closed {
            return Ok(());
        }
        self.phase = Phase::Failed;
        self.model
            .retire_failure()
            .map_err(|error| ExecutionError::new(error.to_string()))?;
        self.states.clear();
        self.rows.clear();
        self.phase = Phase::Closed;
        self.readiness.publish();
        Ok(())
    }
}

/// Model-wide continuous-batching Metal executor for Gemma 4.
///
/// Packs all offered prefill and decode rows into a single Metal command buffer,
/// keeping KV caches device-resident and supporting both greedy GPU argmax and
/// deterministic stochastic sampling (`temperature`, `top_p`, `top_k`, `seed`).
pub struct Gemma4MetalModelWideExecutor {
    info: ExecutorInfo,
    execution: Option<ManuallyDrop<Box<Execution>>>,
}

impl Gemma4MetalModelWideExecutor {
    /// Bind a prepared `Gemma4MetalModelWide` to the `GenerationExecutor` contract.
    /// # Errors
    /// Rejects invalid limits or a faulted model.
    pub fn new(model: Gemma4MetalModelWide) -> Result<Self, Gemma4Error> {
        if model.active_slots() != 0 {
            return Err(Gemma4Error::new(
                "Metal executor",
                "physical continuations already exist",
            ));
        }
        let limits = model.limits();
        let context_tokens = u32::try_from(limits.max_positions)
            .map_err(|_| Gemma4Error::new("Metal executor", "context exceeds u32"))?;
        let max_batch_tokens = u32::try_from(limits.max_batch_tokens)
            .map_err(|_| Gemma4Error::new("Metal executor", "max_batch_tokens exceeds u32"))?;
        let max_sequences = limits.max_sequences;
        Ok(Self {
            info: ExecutorInfo {
                name: "gemma4 metal modelwide".into(),
                limits: GenerationLimits {
                    context_tokens,
                    max_sequences,
                    max_batch_tokens,
                    max_decode_tokens: 1,
                },
            },
            execution: Some(ManuallyDrop::new(Box::new(Execution {
                model,
                states: HashMap::with_capacity(max_sequences),
                rows: Vec::with_capacity(max_sequences),
                readiness: Readiness::default(),
                phase: Phase::Idle,
                next: 0,
            }))),
        })
    }
}

impl Drop for Gemma4MetalModelWideExecutor {
    fn drop(&mut self) {
        if let Some(mut owner) = self.execution.take()
            && owner.close().is_ok()
        {
            drop(ManuallyDrop::into_inner(owner));
        }
    }
}

fn execution_error(error: &Gemma4Error) -> ExecutionError {
    ExecutionError::new(error.to_string())
}

impl GenerationExecutor for Gemma4MetalModelWideExecutor {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn supports_sampling(&self) -> bool {
        true
    }

    fn prefix_cache_stats(&self) -> Option<ribn::PrefixCacheStats> {
        self.execution
            .as_ref()
            .map(|owner| owner.model.prefix_cache_stats())
    }

    fn admitted_prefix(&self, sequence: SequenceId) -> u32 {
        self.execution
            .as_ref()
            .and_then(|owner| owner.states.get(&sequence))
            .map_or(0, |entry| entry.logical.cursor)
    }

    fn admit(
        &mut self,
        request_id: RequestId,
        sequence: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        let owner = self.execution.as_mut().expect("live Metal execution");
        if owner.phase != Phase::Idle {
            return Err(ExecutionError::new(
                "Metal execution is busy, faulted or closed",
            ));
        }
        if owner.states.contains_key(&sequence) {
            owner.phase = Phase::Failed;
            return Err(ExecutionError::new("duplicate Metal sequence incarnation"));
        }
        let mut logical = match LogicalSequence::new_sampled(
            owner.model.config(),
            self.info.limits.context_tokens,
            request,
        ) {
            Ok(logical) => logical,
            Err(error) => return Ok(Admission::Rejected(error)),
        };

        let reach_tokens = request
            .tokens
            .len()
            .checked_add(request.options.max_output_tokens as usize)
            .ok_or_else(|| ExecutionError::new("Metal reachable token count overflow"))?;
        let total_pages = reach_tokens.div_ceil(KV_PAGE_TOKENS);
        let pool_blocks = owner.model.kv_pool_total_blocks();
        if total_pages > pool_blocks {
            return Ok(Admission::Rejected(ExecutionError::new(format!(
                "request needs {total_pages} KV pages, exceeding shared Metal pool capacity of {pool_blocks}"
            ))));
        }

        // Register before testing transient slot/KV-pool capacity (zero lost-wakeup window).
        let wait = owner.readiness.register();
        if owner.states.len() >= self.info.limits.max_sequences {
            return Ok(Admission::Deferred(wait));
        }

        let (matched_pages, newly_pinned) = owner.model.peek_prefix_pages(&request.tokens);
        let future_for_new = total_pages.saturating_sub(matched_pages);
        let worst_case_blocks = owner
            .model
            .kv_pool_pinned_blocks()
            .saturating_add(newly_pinned)
            .saturating_add(owner.reserved_future_blocks())
            .saturating_add(future_for_new);
        if worst_case_blocks > pool_blocks {
            return Ok(Admission::Deferred(wait));
        }

        let (slot, matched_prefix) = match owner.model.allocate_slot_with_prefix(&request.tokens) {
            Ok(result) => result,
            Err(error) => return Ok(Admission::Rejected(execution_error(&error))),
        };
        logical.cursor = u32::try_from(matched_prefix)
            .map_err(|_| ExecutionError::new("cached prefix exceeds u32"))?;
        let seed = init_rng_state(request.options.sampling.seed, request_id);
        owner.states.insert(
            sequence,
            SequenceEntry {
                slot,
                logical,
                max_reach_pages: total_pages,
                sampling: request.options.sampling,
                rng_state: seed,
            },
        );
        Ok(Admission::Ready)
    }

    fn prepare(&mut self, batch: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
        let owner: &mut Execution = self.execution.as_mut().expect("live Metal execution");
        if owner.phase != Phase::Idle {
            return Err(ExecutionError::new(
                "Metal execution is busy, faulted or closed",
            ));
        }
        let Execution {
            model,
            states,
            rows,
            phase,
            ..
        } = owner;
        bind_batch(
            batch,
            self.info.limits,
            |id| states.get(&id).map(|entry| &entry.logical),
            rows,
        )?;
        for row in rows.iter() {
            let entry = states.get(&row.work().sequence).expect("bound sequence");
            if model.slot_position(entry.slot) != Some(row.work().prefix as usize) {
                rows.clear();
                return Err(ExecutionError::new(
                    "Metal physical/logical prefix mismatch",
                ));
            }
        }
        *phase = Phase::Prepared;
        Ok(BatchPreparation::Ready)
    }

    fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
        let owner = self.execution.as_mut().expect("live Metal execution");
        match owner.phase {
            Phase::Prepared => {
                owner.rows.clear();
                owner.phase = Phase::Idle;
            }
            Phase::Idle | Phase::Closed => {}
            Phase::Submitted(_) | Phase::Failed => {
                return Err(ExecutionError::new(
                    "cannot abandon submitted or failed Metal work",
                ));
            }
        }
        Ok(())
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        let owner = self.execution.as_mut().expect("live Metal execution");
        if owner.phase != Phase::Prepared
            || !owner
                .rows
                .iter()
                .map(BoundRow::work)
                .eq(batch.iter().copied())
        {
            owner.phase = Phase::Failed;
            return Err(ExecutionError::new(
                "Metal submission differs from preparation",
            ));
        }
        owner.next = owner.next.wrapping_add(1);
        let id = SubmissionId::new(owner.next);
        owner.phase = Phase::Submitted(id);
        Ok(id)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "atomic packed-batch submission, sampling, commit, and KV-reservation wakeup"
    )]
    fn poll(
        &mut self,
        submission: SubmissionId,
    ) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        let owner: &mut Execution = self.execution.as_mut().expect("live Metal execution");
        if owner.phase != Phase::Submitted(submission) {
            return Err(ExecutionError::new("unknown Metal submission"));
        }
        owner.phase = Phase::Failed;
        let pinned_before = owner.model.kv_pool_pinned_blocks();
        let reserved_before = owner.reserved_future_blocks();
        let Execution {
            model,
            states,
            rows,
            readiness,
            phase,
            ..
        } = owner;

        // Build packed items borrowing token slices from bound logical sequences.
        let mut decode_tokens: Vec<u32> = Vec::with_capacity(rows.len());
        let mut packed_specs: Vec<(usize, SequenceId, u32, bool)> = Vec::with_capacity(rows.len());
        for row in rows.iter() {
            let work = row.work();
            let entry = states.get(&work.sequence).expect("bound sequence");
            let need_logits = work.output_budget != 0;
            if let RowInput::Decode(tok) = row.input() {
                decode_tokens.push(*tok);
            }
            packed_specs.push((entry.slot, work.sequence, row.end(), need_logits));
        }

        let mut completions = Vec::with_capacity(rows.len());
        {
            let mut decode_cursor = 0_usize;
            let mut packed_items: Vec<PackedStepItem<'_>> = Vec::with_capacity(rows.len());
            for (idx, row) in rows.iter().enumerate() {
                let (slot, sequence, _, need_logits) = packed_specs[idx];
                let tokens: &[u32] = match row.input() {
                    RowInput::Prefill(range) => {
                        let entry = states.get(&sequence).expect("bound sequence");
                        &entry.logical.prompt[range.clone()]
                    }
                    RowInput::Decode(_) => {
                        let slice = &decode_tokens[decode_cursor..=decode_cursor];
                        decode_cursor += 1;
                        slice
                    }
                };
                packed_items.push(PackedStepItem {
                    slot,
                    tokens,
                    need_logits,
                });
            }

            // Single Metal command buffer executes ALL rows together!
            let output = model
                .step_packed(&packed_items)
                .map_err(|error| execution_error(&error))?;

            let mut out_idx = 0_usize;
            for &(_, sequence, end, need_logits) in &packed_specs {
                let tokens = if need_logits {
                    let entry = states.get_mut(&sequence).expect("bound sequence");
                    let token = if entry.sampling.temperature == 0.0 {
                        output.argmax[out_idx]
                    } else {
                        sample_bf16_logits(
                            output.logits_row(out_idx),
                            entry.sampling,
                            &mut entry.rng_state,
                        )?
                    };
                    out_idx += 1;
                    vec![token]
                } else {
                    Vec::new()
                };
                completions.push(StepCompletion {
                    sequence,
                    prefix: end,
                    tokens,
                });
            }
        }

        // Validate physical/logical sync and commit all rows atomically.
        for completion in &completions {
            let entry = states.get(&completion.sequence).expect("bound sequence");
            if model.slot_position(entry.slot) != Some(completion.prefix as usize) {
                return Err(ExecutionError::new("Metal physical/logical end mismatch"));
            }
        }
        for completion in &completions {
            states
                .get_mut(&completion.sequence)
                .expect("bound sequence")
                .logical
                .commit(completion.prefix, completion.tokens.first().copied());
        }
        rows.clear();
        *phase = Phase::Idle;
        let pinned_after = model.kv_pool_pinned_blocks();
        let reserved_after: usize = states
            .values()
            .map(|entry| {
                entry
                    .max_reach_pages
                    .saturating_sub(model.slot_allocated_blocks(entry.slot))
            })
            .sum();
        if pinned_after + reserved_after < pinned_before + reserved_before {
            readiness.publish();
        }
        Ok(Some(completions))
    }

    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        let owner = self.execution.as_mut().expect("live Metal execution");
        if owner.phase == Phase::Closed {
            return Ok(());
        }
        if owner.phase == Phase::Failed
            || owner.rows.iter().any(|row| row.work().sequence == sequence)
        {
            return Err(ExecutionError::new(
                "cannot release prepared, submitted or failed Metal state",
            ));
        }
        if let Some(entry) = owner.states.get(&sequence) {
            let slot = entry.slot;
            let previous = owner.phase;
            owner.phase = Phase::Failed;
            owner
                .model
                .free_slot(slot)
                .map_err(|error| execution_error(&error))?;
            owner.states.remove(&sequence);
            owner.phase = previous;
            owner.readiness.publish();
        }
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.execution
            .as_mut()
            .expect("live Metal execution")
            .close()
    }
}
