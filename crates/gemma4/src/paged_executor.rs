//! Paged continuous-batching `GenerationExecutor` with partial-envelope admission,
//! pre-submit KV block growth negotiation, and runtime-managed recomputation preemption.

use crate::generation::{BoundRow, LogicalSequence, RowInput, bind_batch};
use crate::{Gemma4Config, Gemma4Error};
use ribn::{
    Admission, BatchItem, BatchPreparation, ExecutionError, ExecutorInfo, GenerationExecutor,
    GenerationLimits, PrefixCacheStats, PreparedRow, Readiness, RequestId, Sampling, SequenceId,
    StepCompletion, StepKind, SubmissionId, TokenRequest, init_rng_state, sample_bf16_logits,
};
use std::collections::HashMap;
use std::mem::ManuallyDrop;

/// Physical page granularity (tokens per KV block) for model-wide paged execution.
pub const KV_PAGE_TOKENS: usize = 16;

/// One packed sequence slice in a batched model-wide forward pass.
#[derive(Clone, Copy, Debug)]
pub struct PackedStepItem<'a> {
    pub slot: usize,
    pub tokens: &'a [u32],
    pub need_logits: bool,
}

/// Zero-copy view over the completed outputs of a packed forward pass.
pub struct PackedStepOutput<'a> {
    pub argmax: &'a [u32],
    pub logits: &'a [u16],
    pub vocab_size: usize,
}

impl PackedStepOutput<'_> {
    /// Borrow the soft-capped BF16 logits row for the `index`-th output-producing item.
    #[must_use]
    pub fn logits_row(&self, index: usize) -> &[u16] {
        let start = index * self.vocab_size;
        &self.logits[start..start + self.vocab_size]
    }
}

/// Internal seam separating backend-specific device storage/execution from the
/// shared paged admission, KV growth preparation, rollback, and preemption owner.
pub(crate) trait PagedModelWideBackend: Send {
    fn executor_name(&self) -> &'static str;
    fn config(&self) -> &Gemma4Config;
    fn max_sequences(&self) -> usize;
    fn max_positions(&self) -> usize;
    fn max_batch_tokens(&self) -> usize;
    fn active_slots(&self) -> usize;
    fn prefix_cache_stats(&self) -> PrefixCacheStats;
    fn kv_pool_total_blocks(&self) -> usize;
    fn kv_pool_pinned_blocks(&self) -> usize;
    fn kv_pool_available_blocks(&self) -> usize;
    fn peek_prefix_pages(&self, prompt: &[u32]) -> (usize, usize);
    fn allocate_slot_with_prefix(&mut self, prompt: &[u32]) -> Result<(usize, usize), Gemma4Error>;
    fn free_slot(&mut self, slot: usize) -> Result<(), Gemma4Error>;
    fn slot_position(&self, slot: usize) -> Option<usize>;
    fn slot_allocated_blocks(&self, slot: usize) -> usize;
    fn try_reserve_slot_blocks(
        &mut self,
        slot: usize,
        end_pos: usize,
    ) -> Result<Option<usize>, Gemma4Error>;
    fn rollback_slot_blocks(&mut self, slot: usize, prev_len: usize) -> Result<(), Gemma4Error>;
    fn step_packed(
        &mut self,
        items: &[PackedStepItem<'_>],
    ) -> Result<PackedStepOutput<'_>, Gemma4Error>;
    fn retire_failure(&self) -> Result<(), Gemma4Error>;
}

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
    prompt_pages: usize,
    sampling: Sampling,
    rng_state: u64,
}

struct Execution<B: PagedModelWideBackend> {
    model: B,
    states: HashMap<SequenceId, SequenceEntry>,
    rows: Vec<BoundRow>,
    prepared_growth: Vec<(usize, usize)>,
    readiness: Readiness,
    phase: Phase,
    next: u64,
}

impl<B: PagedModelWideBackend> Execution<B> {
    /// Unallocated prompt pages still required by already-admitted sequences
    /// before their initial prefill completes. Keeps admission from over-committing
    /// simultaneous prompts beyond pool capacity while allowing decode growth to
    /// share the remaining pool dynamically.
    fn reserved_prompt_blocks(&self) -> usize {
        self.states
            .values()
            .map(|entry| {
                entry
                    .prompt_pages
                    .saturating_sub(self.model.slot_allocated_blocks(entry.slot))
            })
            .sum()
    }

    fn rollback_prepared_growth(&mut self) -> Result<bool, ExecutionError> {
        let released_any = !self.prepared_growth.is_empty();
        let mut first_error = None;
        for (slot, old_len) in self.prepared_growth.drain(..).rev() {
            if let Err(err) = self.model.rollback_slot_blocks(slot, old_len)
                && first_error.is_none()
            {
                first_error = Some(err);
            }
        }
        if let Some(err) = first_error {
            self.phase = Phase::Failed;
            return Err(execution_error(&err));
        }
        Ok(released_any)
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
        self.prepared_growth.clear();
        self.phase = Phase::Closed;
        self.readiness.publish();
        Ok(())
    }
}

pub(crate) struct PagedModelWideExecutor<B: PagedModelWideBackend> {
    info: ExecutorInfo,
    execution: Option<ManuallyDrop<Box<Execution<B>>>>,
}

impl<B: PagedModelWideBackend> PagedModelWideExecutor<B> {
    pub(crate) fn new(model: B) -> Result<Self, Gemma4Error> {
        if model.active_slots() != 0 {
            return Err(Gemma4Error::new(
                "paged executor",
                "physical continuations already exist",
            ));
        }
        let context_tokens = u32::try_from(model.max_positions())
            .map_err(|_| Gemma4Error::new("paged executor", "context exceeds u32"))?;
        let max_batch_tokens = u32::try_from(model.max_batch_tokens())
            .map_err(|_| Gemma4Error::new("paged executor", "max_batch_tokens exceeds u32"))?;
        let max_sequences = model.max_sequences();
        let name = model.executor_name().into();
        Ok(Self {
            info: ExecutorInfo {
                name,
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
                prepared_growth: Vec::with_capacity(max_sequences),
                readiness: Readiness::default(),
                phase: Phase::Idle,
                next: 0,
            }))),
        })
    }
}

impl<B: PagedModelWideBackend> Drop for PagedModelWideExecutor<B> {
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

fn rollback_growth<B: PagedModelWideBackend>(
    model: &mut B,
    prepared_growth: &mut Vec<(usize, usize)>,
) {
    for (slot, old_len) in prepared_growth.drain(..).rev() {
        let _ = model.rollback_slot_blocks(slot, old_len);
    }
}

impl<B: PagedModelWideBackend> GenerationExecutor for PagedModelWideExecutor<B> {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn supports_sampling(&self) -> bool {
        true
    }

    fn supports_preemption(&self) -> bool {
        true
    }

    fn prefix_cache_stats(&self) -> Option<PrefixCacheStats> {
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
        let owner = self.execution.as_mut().expect("live paged execution");
        if owner.phase != Phase::Idle {
            return Err(ExecutionError::new(
                "paged execution is busy, faulted or closed",
            ));
        }
        if owner.states.contains_key(&sequence) {
            owner.phase = Phase::Failed;
            return Err(ExecutionError::new("duplicate paged sequence incarnation"));
        }
        let mut logical = match LogicalSequence::new_sampled(
            owner.model.config(),
            self.info.limits.context_tokens,
            request,
        ) {
            Ok(logical) => logical,
            Err(error) => return Ok(Admission::Rejected(error)),
        };

        // Individually impossible requests whose full reachable KV span exceeds the
        // entire shared pool must reject locally rather than entering endless preemption.
        // The last generated token does not write a new KV entry, so max KV reach is
        // `prompt.len() + max_output_tokens - 1`.
        let kv_reach_tokens = request
            .tokens
            .len()
            .checked_add(request.options.max_output_tokens.saturating_sub(1) as usize)
            .ok_or_else(|| ExecutionError::new("reachable KV token count overflow"))?;
        let max_reach_pages = kv_reach_tokens.div_ceil(KV_PAGE_TOKENS);
        let pool_blocks = owner.model.kv_pool_total_blocks();
        if max_reach_pages > pool_blocks {
            return Ok(Admission::Rejected(ExecutionError::new(format!(
                "request needs {max_reach_pages} KV pages, exceeding shared KV pool capacity of {pool_blocks}"
            ))));
        }

        // Register before testing transient slot/KV-pool capacity (zero lost-wakeup window).
        let wait = owner.readiness.register();
        if owner.states.len() >= self.info.limits.max_sequences {
            return Ok(Admission::Deferred(wait));
        }

        // Partial-envelope admission: require blocks for the sequence's initial prompt
        // (minus already-pinned shared prefix pages) plus uncommitted prompt budgets of
        // already-admitted sequences, rather than worst-case `prompt + max_output_tokens`.
        // Mid-generation KV page growth is negotiated in `prepare()` and resolved via
        // completion-safe recomputation preemption if the pool saturates.
        let prompt_pages = request.tokens.len().div_ceil(KV_PAGE_TOKENS);
        let (matched_pages, newly_pinned) = owner.model.peek_prefix_pages(&request.tokens);
        let new_prompt_blocks = prompt_pages.saturating_sub(matched_pages);
        let initial_blocks = owner
            .model
            .kv_pool_pinned_blocks()
            .saturating_add(newly_pinned)
            .saturating_add(owner.reserved_prompt_blocks())
            .saturating_add(new_prompt_blocks);
        if initial_blocks > pool_blocks {
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
                prompt_pages,
                sampling: request.options.sampling,
                rng_state: seed,
            },
        );
        Ok(Admission::Ready)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "atomic whole-batch validation, KV block growth negotiation, prefill shrinking, and rollback"
    )]
    fn prepare(&mut self, batch: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
        let owner: &mut Execution<B> = self.execution.as_mut().expect("live paged execution");
        if owner.phase != Phase::Idle {
            return Err(ExecutionError::new(
                "paged execution is busy, faulted or closed",
            ));
        }
        let Execution {
            model,
            states,
            rows,
            prepared_growth,
            readiness,
            phase,
            ..
        } = owner;

        prepared_growth.clear();
        let mut bound = Vec::with_capacity(batch.len());
        bind_batch(
            batch,
            self.info.limits,
            |id| states.get(&id).map(|entry| &entry.logical),
            &mut bound,
        )?;
        for row in &bound {
            let entry = states.get(&row.work().sequence).expect("bound sequence");
            if model.slot_position(entry.slot) != Some(row.work().prefix as usize) {
                return Err(ExecutionError::new("physical/logical prefix mismatch"));
            }
        }

        rows.clear();
        let mut decisions = Vec::with_capacity(bound.len());
        let mut all_unchanged = true;

        for row in bound {
            let work = row.work();
            let entry = states.get(&work.sequence).expect("bound sequence");
            let slot = entry.slot;
            let end_pos = row.end() as usize;

            match model.try_reserve_slot_blocks(slot, end_pos) {
                Ok(Some(old_len)) => {
                    if model.slot_allocated_blocks(slot) > old_len {
                        prepared_growth.push((slot, old_len));
                    }
                    rows.push(row);
                    decisions.push(PreparedRow::Ready(work));
                }
                Ok(None) => {
                    all_unchanged = false;
                    // For prefill, shorten positively if existing + available KV pages cover
                    // at least one token past `prefix`; otherwise park until capacity frees.
                    let mut shortened = None;
                    if work.kind == StepKind::Prefill {
                        let cur_blocks = model.slot_allocated_blocks(slot);
                        let avail_blocks = model.kv_pool_available_blocks();
                        let max_end = cur_blocks
                            .saturating_add(avail_blocks)
                            .saturating_mul(KV_PAGE_TOKENS)
                            .min(end_pos);
                        let prefix = work.prefix as usize;
                        if max_end > prefix {
                            let new_budget = u32::try_from(max_end - prefix).map_err(|_| {
                                rollback_growth(model, prepared_growth);
                                rows.clear();
                                *phase = Phase::Failed;
                                ExecutionError::new("shortened prefill budget exceeds u32")
                            })?;
                            let shrunk_item = BatchItem {
                                token_budget: new_budget,
                                output_budget: 0,
                                ..work
                            };
                            let shrunk_row = match entry
                                .logical
                                .bind(shrunk_item, self.info.limits.context_tokens)
                            {
                                Ok(r) => r,
                                Err(err) => {
                                    rollback_growth(model, prepared_growth);
                                    rows.clear();
                                    *phase = Phase::Failed;
                                    return Err(err);
                                }
                            };
                            match model.try_reserve_slot_blocks(slot, max_end) {
                                Ok(Some(old_len)) => {
                                    if model.slot_allocated_blocks(slot) > old_len {
                                        prepared_growth.push((slot, old_len));
                                    }
                                    shortened = Some((shrunk_row, shrunk_item));
                                }
                                Ok(None) => {}
                                Err(err) => {
                                    rollback_growth(model, prepared_growth);
                                    rows.clear();
                                    *phase = Phase::Failed;
                                    return Err(execution_error(&err));
                                }
                            }
                        }
                    }
                    if let Some((shrunk_row, shrunk_item)) = shortened {
                        rows.push(shrunk_row);
                        decisions.push(PreparedRow::Ready(shrunk_item));
                    } else {
                        decisions.push(PreparedRow::Deferred(readiness.register()));
                    }
                }
                Err(err) => {
                    rollback_growth(model, prepared_growth);
                    rows.clear();
                    *phase = Phase::Failed;
                    return Err(execution_error(&err));
                }
            }
        }

        *phase = Phase::Prepared;
        if all_unchanged {
            Ok(BatchPreparation::Ready)
        } else {
            Ok(BatchPreparation::Selected(decisions))
        }
    }

    fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
        let owner = self.execution.as_mut().expect("live paged execution");
        match owner.phase {
            Phase::Prepared => {
                owner.rows.clear();
                let released_any = owner.rollback_prepared_growth()?;
                owner.phase = Phase::Idle;
                if released_any {
                    owner.readiness.publish();
                }
            }
            Phase::Idle | Phase::Closed => {}
            Phase::Submitted(_) | Phase::Failed => {
                return Err(ExecutionError::new(
                    "cannot abandon submitted or failed paged work",
                ));
            }
        }
        Ok(())
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        let owner = self.execution.as_mut().expect("live paged execution");
        if owner.phase != Phase::Prepared
            || !owner
                .rows
                .iter()
                .map(BoundRow::work)
                .eq(batch.iter().copied())
        {
            owner.phase = Phase::Failed;
            return Err(ExecutionError::new(
                "paged submission differs from preparation",
            ));
        }
        owner.next = owner.next.wrapping_add(1);
        let id = SubmissionId::new(owner.next);
        owner.phase = Phase::Submitted(id);
        Ok(id)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "atomic packed-batch execution, sampling, commit, and deduplication wakeup"
    )]
    fn poll(
        &mut self,
        submission: SubmissionId,
    ) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        let owner: &mut Execution<B> = self.execution.as_mut().expect("live paged execution");
        if owner.phase != Phase::Submitted(submission) {
            return Err(ExecutionError::new("unknown paged submission"));
        }
        owner.phase = Phase::Failed;
        let pinned_before = owner.model.kv_pool_pinned_blocks();
        let reserved_before = owner.reserved_prompt_blocks();
        let Execution {
            model,
            states,
            rows,
            prepared_growth,
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
                return Err(ExecutionError::new("physical/logical end mismatch"));
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
        prepared_growth.clear();
        *phase = Phase::Idle;

        let pinned_after = model.kv_pool_pinned_blocks();
        let reserved_after: usize = states
            .values()
            .map(|entry| {
                entry
                    .prompt_pages
                    .saturating_sub(model.slot_allocated_blocks(entry.slot))
            })
            .sum();
        if pinned_after + reserved_after < pinned_before + reserved_before {
            readiness.publish();
        }
        Ok(Some(completions))
    }

    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        let owner = self.execution.as_mut().expect("live paged execution");
        if owner.phase == Phase::Closed {
            return Ok(());
        }
        if owner.phase == Phase::Failed
            || owner.rows.iter().any(|row| row.work().sequence == sequence)
        {
            return Err(ExecutionError::new(
                "cannot release prepared, submitted or failed paged state",
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
            .expect("live paged execution")
            .close()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        KV_PAGE_TOKENS, PackedStepItem, PackedStepOutput, PagedModelWideBackend,
        PagedModelWideExecutor,
    };
    use crate::{Gemma4Config, Gemma4Error};
    use ribn::{
        Admission, BatchItem, BatchPreparation, Engine, EngineConfig, Event, FinishReason,
        GenerationExecutor, GenerationOptions, PrefixCacheStats, PreparedRow, RequestId,
        SchedulePolicy, StepKind, TokenRequest, Usage,
    };
    use ribn_foundation::{BlockPoolError, PagedBlockPool};
    use std::collections::HashMap;

    struct HostSlot {
        position: usize,
        blocks: Vec<usize>,
        tokens: Vec<u32>,
        cached_pages: usize,
        last_node: u64,
    }

    /// Pure-host backend exercising the real `PagedBlockPool<16>` and `Gemma4Config`
    /// with deterministic next-token generation (`2 + (sum(tokens) + pos) % 5`).
    struct HostPagedBackend {
        config: Gemma4Config,
        max_sequences: usize,
        max_positions: usize,
        max_batch_tokens: usize,
        kv_pool: PagedBlockPool<KV_PAGE_TOKENS>,
        slots: Vec<Option<HostSlot>>,
        argmax_buf: Vec<u32>,
        logits_buf: Vec<u16>,
        max_seen_batch_rows: usize,
        fail_next_step: bool,
        fail_retire: bool,
        faulted: bool,
    }

    impl HostPagedBackend {
        fn new(
            max_sequences: usize,
            max_positions: usize,
            max_batch_tokens: usize,
            num_kv_blocks: usize,
        ) -> Self {
            let config = Gemma4Config::from_json(&crate::weight_fixtures::tiny()).unwrap();
            let mut slots = Vec::with_capacity(max_sequences);
            slots.resize_with(max_sequences, || None);
            Self {
                config,
                max_sequences,
                max_positions,
                max_batch_tokens,
                kv_pool: PagedBlockPool::new(num_kv_blocks).unwrap(),
                slots,
                argmax_buf: Vec::new(),
                logits_buf: Vec::new(),
                max_seen_batch_rows: 0,
                fail_next_step: false,
                fail_retire: false,
                faulted: false,
            }
        }

        fn deterministic_next_token(tokens: &[u32]) -> u32 {
            let sum = tokens
                .iter()
                .fold(0_u32, |acc, &t| acc.wrapping_mul(31).wrapping_add(t));
            2 + (sum % 5)
        }
    }

    impl PagedModelWideBackend for HostPagedBackend {
        fn executor_name(&self) -> &'static str {
            "host paged gemma4"
        }
        fn config(&self) -> &Gemma4Config {
            &self.config
        }
        fn max_sequences(&self) -> usize {
            self.max_sequences
        }
        fn max_positions(&self) -> usize {
            self.max_positions
        }
        fn max_batch_tokens(&self) -> usize {
            self.max_batch_tokens
        }
        fn active_slots(&self) -> usize {
            self.slots.iter().filter(|s| s.is_some()).count()
        }
        fn prefix_cache_stats(&self) -> PrefixCacheStats {
            self.kv_pool.stats()
        }
        fn kv_pool_total_blocks(&self) -> usize {
            self.kv_pool.total_blocks()
        }
        fn kv_pool_pinned_blocks(&self) -> usize {
            self.kv_pool.pinned_blocks()
        }
        fn kv_pool_available_blocks(&self) -> usize {
            self.kv_pool.available_blocks()
        }
        fn peek_prefix_pages(&self, prompt: &[u32]) -> (usize, usize) {
            self.kv_pool.peek_prefix_match(prompt)
        }
        fn allocate_slot_with_prefix(
            &mut self,
            prompt: &[u32],
        ) -> Result<(usize, usize), Gemma4Error> {
            if self.faulted {
                return Err(Gemma4Error::new("host paged", "faulted"));
            }
            let slot = self
                .slots
                .iter()
                .position(Option::is_none)
                .ok_or_else(|| Gemma4Error::new("host paged", "slot capacity exhausted"))?;
            let (blocks, last_node) = self.kv_pool.match_and_pin_prefix(prompt);
            let cached_pages = blocks.len();
            let matched = cached_pages * KV_PAGE_TOKENS;
            self.slots[slot] = Some(HostSlot {
                position: matched,
                blocks,
                tokens: prompt[..matched].to_vec(),
                cached_pages,
                last_node,
            });
            Ok((slot, matched))
        }
        fn free_slot(&mut self, slot: usize) -> Result<(), Gemma4Error> {
            if self.faulted {
                return Err(Gemma4Error::new("host paged", "faulted"));
            }
            let entry = self
                .slots
                .get_mut(slot)
                .ok_or_else(|| Gemma4Error::new("host paged", "slot out of range"))?;
            if let Some(state) = entry.take() {
                self.kv_pool.release_slot_blocks(state.blocks);
            }
            Ok(())
        }
        fn slot_position(&self, slot: usize) -> Option<usize> {
            self.slots
                .get(slot)
                .and_then(Option::as_ref)
                .map(|s| s.position)
        }
        fn slot_allocated_blocks(&self, slot: usize) -> usize {
            self.slots
                .get(slot)
                .and_then(Option::as_ref)
                .map_or(0, |s| s.blocks.len())
        }
        fn try_reserve_slot_blocks(
            &mut self,
            slot: usize,
            end_pos: usize,
        ) -> Result<Option<usize>, Gemma4Error> {
            if self.faulted {
                return Err(Gemma4Error::new("host paged", "faulted"));
            }
            if end_pos > self.max_positions {
                return Err(Gemma4Error::new("host paged", "position exceeds max"));
            }
            let state = self
                .slots
                .get_mut(slot)
                .and_then(Option::as_mut)
                .ok_or_else(|| Gemma4Error::new("host paged", "unoccupied slot"))?;
            let old_len = state.blocks.len();
            match self
                .kv_pool
                .ensure_blocks_for_range(&mut state.blocks, end_pos)
            {
                Ok(()) => Ok(Some(old_len)),
                Err(BlockPoolError::Exhausted) => Ok(None),
                Err(e) => Err(Gemma4Error::new("host paged", e.to_string())),
            }
        }
        fn rollback_slot_blocks(
            &mut self,
            slot: usize,
            prev_len: usize,
        ) -> Result<(), Gemma4Error> {
            if self.faulted {
                return Err(Gemma4Error::new("host paged", "faulted"));
            }
            let state = self
                .slots
                .get_mut(slot)
                .and_then(Option::as_mut)
                .ok_or_else(|| Gemma4Error::new("host paged", "unoccupied slot"))?;
            let committed = state
                .position
                .div_ceil(KV_PAGE_TOKENS)
                .max(state.cached_pages);
            if prev_len < committed || prev_len > state.blocks.len() {
                return Err(Gemma4Error::new("host paged", "invalid rollback boundary"));
            }
            if prev_len < state.blocks.len() {
                let extra: Vec<usize> = state.blocks.drain(prev_len..).collect();
                self.kv_pool.release_slot_blocks(extra);
            }
            Ok(())
        }
        fn step_packed(
            &mut self,
            items: &[PackedStepItem<'_>],
        ) -> Result<PackedStepOutput<'_>, Gemma4Error> {
            if self.faulted || self.fail_next_step {
                self.faulted = true;
                return Err(Gemma4Error::new("host paged", "injected step failure"));
            }
            self.max_seen_batch_rows = self.max_seen_batch_rows.max(items.len());
            self.argmax_buf.clear();
            self.logits_buf.clear();
            let vocab = self.config.vocab_size();
            for item in items {
                let state = self.slots[item.slot]
                    .as_mut()
                    .ok_or_else(|| Gemma4Error::new("host paged", "unoccupied slot"))?;
                let end_pos = state.position + item.tokens.len();
                assert!(
                    state.blocks.len() >= end_pos.div_ceil(KV_PAGE_TOKENS),
                    "prepare() must reserve all KV blocks before step_packed()"
                );
                state.position = end_pos;
                state.tokens.extend_from_slice(item.tokens);
                self.kv_pool.publish_completed_blocks(
                    &mut state.blocks,
                    &state.tokens,
                    &mut state.cached_pages,
                    &mut state.last_node,
                );
                if item.need_logits {
                    let tok = Self::deterministic_next_token(&state.tokens);
                    self.argmax_buf.push(tok);
                    let start = self.logits_buf.len();
                    self.logits_buf.resize(start + vocab, 0);
                    self.logits_buf[start + tok as usize] = 0x3f80;
                }
            }
            Ok(PackedStepOutput {
                argmax: &self.argmax_buf,
                logits: &self.logits_buf,
                vocab_size: vocab,
            })
        }
        fn retire_failure(&self) -> Result<(), Gemma4Error> {
            if self.fail_retire {
                Err(Gemma4Error::new("host paged", "injected retire failure"))
            } else {
                Ok(())
            }
        }
    }

    fn expected_tokens(prompt: &[u32], count: usize) -> Vec<u32> {
        let mut history = prompt.to_vec();
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let next = HostPagedBackend::deterministic_next_token(&history);
            out.push(next);
            history.push(next);
        }
        out
    }

    fn make_req(prompt: Vec<u32>, max_output_tokens: u32) -> TokenRequest {
        TokenRequest::new(
            prompt,
            GenerationOptions {
                max_output_tokens,
                ..GenerationOptions::default()
            },
        )
    }

    #[test]
    fn partial_envelope_batches_concurrently_preempts_on_growth_and_replays_exact_output() {
        // Pool has 4 blocks (64 tokens). Two requests each have 20-token prompts (2 pages)
        // and 30 output tokens (total 50 tokens -> 4 pages each, worst-case 8 > 4).
        // Partial-envelope admission admits BOTH immediately (2 + 2 = 4 <= 4), batches
        // them at M=2 through position 32, preempts the newer request on page-3 growth,
        // finishes the protected survivor, then replays the victim with zero duplicate
        // tokens or usage.
        let backend = HostPagedBackend::new(2, 64, 64, 4);
        let exec = PagedModelWideExecutor::new(backend).unwrap();
        let mut engine = Engine::new(
            exec,
            EngineConfig {
                max_active_requests: 2,
                max_queued_requests: 4,
                max_buffered_events: 256,
                max_events_per_request: 64,
                ..EngineConfig::default()
            },
            SchedulePolicy {
                max_batch_tokens: 64,
                prefill_chunk_tokens: 64,
                ..SchedulePolicy::default()
            },
        )
        .unwrap();

        let prompt_a: Vec<u32> = (0_u32..20).map(|i| 2 + (i % 5)).collect();
        let prompt_b: Vec<u32> = (0_u32..20).map(|i| 2 + ((i + 2) % 5)).collect();
        let expected_a = expected_tokens(&prompt_a, 30);
        let expected_b = expected_tokens(&prompt_b, 30);

        let req_a = engine.enqueue(make_req(prompt_a, 30)).unwrap();
        let req_b = engine.enqueue(make_req(prompt_b, 30)).unwrap();

        let mut outputs: HashMap<RequestId, Vec<u32>> = HashMap::new();
        let mut finished: HashMap<RequestId, (FinishReason, Usage)> = HashMap::new();
        let mut saw_both_active = false;

        for _ in 0..400 {
            engine.step().unwrap();
            if engine.status().active_sequences == 2 {
                saw_both_active = true;
            }
            while let Some(ev) = engine.pop_event() {
                match ev {
                    Event::Token { request, token } => {
                        outputs.entry(request).or_default().push(token);
                    }
                    Event::Finished {
                        request,
                        reason,
                        usage,
                    } => {
                        finished.insert(request, (reason, usage));
                    }
                }
            }
            if finished.len() == 2 {
                break;
            }
        }

        assert!(
            saw_both_active,
            "partial-envelope admission must admit both prompts concurrently"
        );
        assert!(
            engine.preemptions() >= 1,
            "all-active growth beyond 4 blocks must trigger completion-safe preemption"
        );
        assert_eq!(outputs[&req_a], expected_a);
        assert_eq!(outputs[&req_b], expected_b);
        assert_eq!(
            finished[&req_a],
            (
                FinishReason::Length,
                Usage {
                    prompt_tokens: 20,
                    completion_tokens: 30,
                }
            )
        );
        assert_eq!(
            finished[&req_b],
            (
                FinishReason::Length,
                Usage {
                    prompt_tokens: 20,
                    completion_tokens: 30,
                }
            )
        );
        let stats = engine.prefix_cache_stats().unwrap();
        assert_eq!(stats.pinned_blocks, 0);
        engine.shutdown().unwrap();
    }

    #[test]
    fn survivor_priority_prevents_block_stealing_and_reuses_victim_prefix_cache() {
        // Pool has 4 blocks (64 tokens), 3 concurrent 16-token prompts (1 page each = 3 blocks),
        // each generating 25 tokens (reach = 40 tokens = 3 pages <= 4).
        // All 3 admit together; as they grow past 32 tokens, survivor A must finish first
        // without peers stealing freed blocks, and preempted victims reuse surviving cached
        // prefix pages on readmission.
        let backend = HostPagedBackend::new(3, 64, 64, 4);
        let exec = PagedModelWideExecutor::new(backend).unwrap();
        let mut engine = Engine::new(
            exec,
            EngineConfig {
                max_active_requests: 3,
                max_queued_requests: 4,
                max_buffered_events: 256,
                max_events_per_request: 64,
                ..EngineConfig::default()
            },
            SchedulePolicy {
                max_batch_tokens: 64,
                prefill_chunk_tokens: 64,
                ..SchedulePolicy::default()
            },
        )
        .unwrap();

        let p1 = vec![2_u32; 16];
        let p2 = vec![3_u32; 16];
        let p3 = vec![4_u32; 16];
        let e1 = expected_tokens(&p1, 25);
        let e2 = expected_tokens(&p2, 25);
        let e3 = expected_tokens(&p3, 25);

        let r1 = engine.enqueue(make_req(p1, 25)).unwrap();
        let r2 = engine.enqueue(make_req(p2, 25)).unwrap();
        let r3 = engine.enqueue(make_req(p3, 25)).unwrap();

        let mut outputs: HashMap<RequestId, Vec<u32>> = HashMap::new();
        let mut finish_order = Vec::new();
        for _ in 0..600 {
            engine.step().unwrap();
            while let Some(ev) = engine.pop_event() {
                match ev {
                    Event::Token { request, token } => {
                        outputs.entry(request).or_default().push(token);
                    }
                    Event::Finished {
                        request, reason, ..
                    } => {
                        assert_eq!(reason, FinishReason::Length);
                        finish_order.push(request);
                    }
                }
            }
            if finish_order.len() == 3 {
                break;
            }
        }

        assert_eq!(finish_order.len(), 3);
        assert_eq!(
            finish_order[0], r1,
            "oldest protected survivor r1 must finish first"
        );
        assert_eq!(outputs[&r1], e1);
        assert_eq!(outputs[&r2], e2);
        assert_eq!(outputs[&r3], e3);
        assert!(engine.preemptions() >= 1);
        assert!(
            engine.prefix_cache_stats().unwrap().prefix_hit_tokens > 0,
            "readmitted victim must reuse surviving cached prefix pages"
        );
        assert_eq!(engine.prefix_cache_stats().unwrap().pinned_blocks, 0);
        engine.shutdown().unwrap();
    }

    #[test]
    fn prepare_shortens_prefill_on_partial_blocks_and_abandon_rolls_back_growth() {
        // Pool has 3 blocks (48 tokens). Sequence A holds 1 block (16 tokens);
        // Sequence B is admitted with a 32-token prompt (2 pages: 1 + 2 = 3 <= 3),
        // then A grows to 2 blocks before B prefills, leaving only 1 free block for B.
        // `prepare` must shorten B's 32-token prefill to 16 tokens (`output_budget = 0`),
        // and `abandon_preparation` must roll back the newly reserved block without
        // disturbing A's committed pages.
        let backend = HostPagedBackend::new(2, 64, 64, 3);
        let mut exec = PagedModelWideExecutor::new(backend).unwrap();

        let [(req_a, seq_a), (req_b, seq_b)] = crate::modelwide_ids::issued::<2>();
        assert_eq!(
            exec.admit(req_a, seq_a, &make_req(vec![2; 16], 20))
                .unwrap(),
            Admission::Ready
        );
        assert_eq!(
            exec.admit(req_b, seq_b, &make_req(vec![3; 32], 4)).unwrap(),
            Admission::Ready
        );

        // Prefill A (takes 1 block -> pinned = 1), then decode A 1 step (takes 2nd block -> pinned = 2).
        let a_prefill = [BatchItem {
            sequence: seq_a,
            kind: StepKind::Prefill,
            prefix: 0,
            token_budget: 16,
            output_budget: 1,
        }];
        assert!(matches!(
            exec.prepare(&a_prefill).unwrap(),
            BatchPreparation::Ready
        ));
        let sub = exec.submit(&a_prefill).unwrap();
        assert!(exec.poll(sub).unwrap().is_some());

        let a_decode = [BatchItem {
            sequence: seq_a,
            kind: StepKind::Decode,
            prefix: 16,
            token_budget: 1,
            output_budget: 1,
        }];
        assert!(matches!(
            exec.prepare(&a_decode).unwrap(),
            BatchPreparation::Ready
        ));
        let sub = exec.submit(&a_decode).unwrap();
        assert!(exec.poll(sub).unwrap().is_some());
        assert_eq!(exec.prefix_cache_stats().unwrap().pinned_blocks, 2);

        // Now only 1 block is free. Offering B's full 32-token prefill must shorten to 16!
        let b_prefill = [BatchItem {
            sequence: seq_b,
            kind: StepKind::Prefill,
            prefix: 0,
            token_budget: 32,
            output_budget: 1,
        }];
        let prep = exec.prepare(&b_prefill).unwrap();
        let BatchPreparation::Selected(rows) = prep else {
            panic!("expected Selected with shortened prefill");
        };
        assert_eq!(rows.len(), 1);
        assert!(matches!(
            rows[0],
            PreparedRow::Ready(BatchItem {
                sequence,
                kind: StepKind::Prefill,
                prefix: 0,
                token_budget: 16,
                output_budget: 0,
            }) if sequence == seq_b
        ));
        assert_eq!(
            exec.prefix_cache_stats().unwrap().pinned_blocks,
            3,
            "prepare owns the newly reserved block for B"
        );

        // Abandoning must release B's uncommitted block while preserving A's 2 committed blocks.
        exec.abandon_preparation().unwrap();
        assert_eq!(
            exec.prefix_cache_stats().unwrap().pinned_blocks,
            2,
            "abandon_preparation must roll back only newly reserved growth"
        );
        exec.synchronize().unwrap();
    }

    #[test]
    fn impossible_request_rejects_locally_and_stalled_consumer_is_preempted() {
        let backend = HostPagedBackend::new(2, 96, 64, 4);
        let exec = PagedModelWideExecutor::new(backend).unwrap();
        let mut engine = Engine::new(
            exec,
            EngineConfig {
                max_active_requests: 2,
                max_queued_requests: 4,
                max_buffered_events: 8,
                max_events_per_request: 3, // stalls after 2 buffered tokens
                ..EngineConfig::default()
            },
            SchedulePolicy {
                max_batch_tokens: 64,
                prefill_chunk_tokens: 64,
                ..SchedulePolicy::default()
            },
        )
        .unwrap();

        // 1. Permanently impossible request (65 prompt + 1 output -> 5 pages > 4 pool blocks)
        let bad = engine.enqueue(make_req(vec![2; 65], 1)).unwrap();
        engine.step().unwrap();
        assert!(matches!(
            engine.pop_event_for(bad),
            Some(Event::Finished {
                reason: FinishReason::Failed(_),
                ..
            })
        ));

        // 2. Request A (prompt=32, 2 pages) stalls because we do not drain its mailbox;
        //    Request B (prompt=48, 3 pages) arrives and needs A's blocks -> A must be
        //    completion-safely preempted so healthy B completes!
        let p_a = vec![2_u32; 32];
        let p_b = vec![3_u32; 48];
        let expected_a = expected_tokens(&p_a, 6);
        let expected_b = expected_tokens(&p_b, 4);

        let req_a = engine.enqueue(make_req(p_a, 6)).unwrap();
        // Run A until its per-request mailbox (capacity 3, reserves 1 for terminal) is full.
        for _ in 0..8 {
            engine.step().unwrap();
        }
        let req_b = engine.enqueue(make_req(p_b, 4)).unwrap();

        let mut b_tokens = Vec::new();
        let mut b_done = false;
        for _ in 0..64 {
            engine.step().unwrap();
            while let Some(ev) = engine.pop_event_for(req_b) {
                match ev {
                    Event::Token { token, .. } => b_tokens.push(token),
                    Event::Finished { reason, .. } => {
                        assert_eq!(reason, FinishReason::Length);
                        b_done = true;
                    }
                }
            }
            if b_done {
                break;
            }
        }
        assert!(b_done, "healthy peer B must finish despite stalled peer A");
        assert_eq!(b_tokens, expected_b);
        assert!(engine.preemptions() >= 1);

        // Now drain A's consumer so A wakes, replays, and finishes with exact tokens & usage.
        let mut a_tokens = Vec::new();
        let mut a_done = None;
        for _ in 0..64 {
            while let Some(ev) = engine.pop_event_for(req_a) {
                match ev {
                    Event::Token { token, .. } => a_tokens.push(token),
                    Event::Finished { reason, usage, .. } => a_done = Some((reason, usage)),
                }
            }
            if a_done.is_some() {
                break;
            }
            engine.step().unwrap();
        }
        assert_eq!(a_tokens, expected_a);
        assert_eq!(
            a_done,
            Some((
                FinishReason::Length,
                Usage {
                    prompt_tokens: 32,
                    completion_tokens: 6,
                }
            ))
        );
        engine.shutdown().unwrap();
    }

    #[test]
    fn step_failure_faults_owner_and_retains_state_until_synchronize() {
        let mut backend = HostPagedBackend::new(2, 64, 64, 4);
        backend.fail_next_step = true;
        backend.fail_retire = true;
        let mut exec = PagedModelWideExecutor::new(backend).unwrap();

        let [(req_1, seq), (req_2, seq_2)] = crate::modelwide_ids::issued::<2>();
        assert_eq!(
            exec.admit(req_1, seq, &make_req(vec![2; 16], 4)).unwrap(),
            Admission::Ready
        );
        let item = [BatchItem {
            sequence: seq,
            kind: StepKind::Prefill,
            prefix: 0,
            token_budget: 16,
            output_budget: 1,
        }];
        assert!(matches!(
            exec.prepare(&item).unwrap(),
            BatchPreparation::Ready
        ));
        let sub = exec.submit(&item).unwrap();
        assert!(exec.poll(sub).is_err());

        // Faulted owner refuses release and new admission, retains pinned block, and fails synchronize while retire fails.
        assert!(exec.release(seq).is_err());
        assert!(exec.admit(req_2, seq_2, &make_req(vec![3; 16], 2)).is_err());
        assert_eq!(exec.prefix_cache_stats().unwrap().pinned_blocks, 1);
        assert!(exec.synchronize().is_err());

        // Allow clean retirement on retry so Drop does not intentionally leak in test.
        exec.execution.as_mut().unwrap().model.fail_retire = false;
        exec.synchronize().unwrap();
    }
}
