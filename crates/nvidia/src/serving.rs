use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::NvidiaDispatcher;
use crate::dispatch_diagnostics::{DecodeFallback, DispatchObserver, DispatchRoute, DispatchWork};
use cudarc::driver::{CudaContext, CudaEvent, CudaStream, PinnedHostSlice};
use engine_core::{
    BackendError, BackendSubmissionId, ExecutionBatch, ExecutionOutcome, ExecutionPhase,
    ExecutionSegment, InferenceStateSet, WeightBinding,
};

use crate::decode::{CudaQwen35BatchDecode, CudaQwen35Decode};
use crate::state::{CudaHybridState, CudaStateKey, CudaStateRegistry};

/// Which dispatcher-owned pinned output slot backs one scheduler row.
#[derive(Clone, Copy)]
struct PinnedSlotLease {
    index: usize,
}

/// Retained resources for one asynchronously in-flight scheduler batch.
struct PendingSubmission {
    /// Recorded after every kernel and pinned copy of this submission, in
    /// stream order.
    completion: CudaEvent,
    /// One leased output slot per row, in scheduler order. Rows that did not
    /// request sampling hold `None`, keeping row order aligned with batch
    /// order.
    outputs: Vec<Option<PinnedSlotLease>>,
    /// Logical state keys whose physical registry entries this submission's
    /// queued work still accesses. Retained until completion so an early
    /// `release_inference_state` cannot free device state under live work.
    state_keys: Vec<CudaStateKey>,
}

/// Row results collected while queueing one scheduler batch.
struct QueuedRows {
    outputs: Vec<Option<PinnedSlotLease>>,
    state_keys: Vec<CudaStateKey>,
}

impl QueuedRows {
    /// Restore scheduler positions after independently owned rows run together.
    fn scatter_group(&mut self, indices: &[usize], group: Self) {
        debug_assert_eq!(indices.len(), group.outputs.len());
        for (&index, output) in indices.iter().zip(group.outputs) {
            self.outputs[index] = output;
        }
        self.state_keys.extend(group.state_keys);
    }
}

/// One-stream asynchronous Qwen3.8 CUDA serving dispatcher.
///
/// Compatible decode rows execute together through prepared batch lanes;
/// other rows use the single-row executor on the same stream. Rows that request
/// sampling copy their greedy selection into dispatcher-owned pinned host
/// slots, one completion event is recorded per submission, and outcomes are
/// published later through [`NvidiaDispatcher::poll_batch`] once the event
/// completes.
///
/// Scratch reuse stays safe because all launches are stream-ordered: a later
/// row or submission cannot observe an earlier row's scratch or state until
/// the stream has finished that work.
///
/// Error contract, matching the `NvidiaDispatcher` seam:
/// - ordinary enqueue errors flush partial work before recycling leases;
///   a failed flush poisons the dispatcher and blocks physical release until
///   teardown, preserving ownership while completion is uncertain;
/// - `poll_batch` returns `Ok(None)` while the completion event is pending.
///   A device fault poisons the dispatcher. Pending records are retired after
///   a successful drain or quarantined until teardown if the drain fails.
///   Quarantined state cannot be reported as successfully released.
pub struct CudaQwen35ServingDispatcher {
    executor: CudaQwen35Decode,
    states: CudaStateRegistry,
    stream: Arc<CudaStream>,
    /// Reusable pinned one-`u32` output slots.
    pinned_outputs: Vec<PinnedHostSlice<u32>>,
    free_outputs: Vec<usize>,
    submissions: crate::submissions::Submissions<PendingSubmission>,
    /// State keys released while still referenced by pending submissions.
    /// Their physical entries stay alive until the referencing submission
    /// completes, then they are released for real.
    deferred_releases: Vec<CudaStateKey>,
    /// Supported execution lanes are prepared before accepting submissions.
    /// Compiled kernels and validated model bindings are shared by all lanes.
    batched: HashMap<usize, CudaQwen35BatchDecode>,
    /// Optional same-sequence prefill lane: prompt tokens of one request run
    /// through one batched forward, which amortizes weight reads across
    /// positions the way the decode lanes amortize them across requests.
    ///
    /// This is deliberately independent of the decode lanes and of
    /// `max_pending_rows`: prefill concurrency is not request concurrency, and
    /// one long prompt benefits even when only one request is live.
    prefill_lane: Option<CudaQwen35BatchDecode>,
    /// Unqualified subgroup selection is opt-in; defaults retain whole-batch selection.
    grouped_decode: bool,
    diagnostics: Option<DispatchObserver>,
}

/// How many leading tokens of one prefill segment can run through a
/// same-sequence lane of `members` tokens, and whether the final sampling
/// token rides in the last chunk.
///
/// Only whole chunks qualify: the lane executes exactly `members` tokens and
/// leaves the remainder to the serial path. A sampling segment normally keeps
/// its final token out of a chunk, because the output path reads logits from
/// the row that token produces. When the lane can sample that row directly
/// (`sample_final`) and the prompt is an exact whole number of chunks, the
/// final token joins the last chunk instead: state advances once and only the
/// output head remains.
fn chunked_prefix_len(
    token_count: usize,
    members: usize,
    requests_sampling: bool,
    sample_final: bool,
) -> (usize, bool) {
    if members == 0 {
        return (0, false);
    }
    let whole = token_count / members * members;
    if requests_sampling && sample_final && token_count > 0 && whole == token_count {
        return (whole, true);
    }
    let chunkable = if requests_sampling {
        token_count.saturating_sub(1)
    } else {
        token_count
    };
    (chunkable / members * members, false)
}

/// Prompt work consumed by the leading whole chunks of one segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PrefillChunks {
    /// Tokens consumed, always a whole number of chunks.
    consumed: usize,
    /// The last chunk's final row holds the residual of the final sampling
    /// token, so the caller must run only the output head for it.
    sampled_final: bool,
}

/// Run the leading whole chunks of one prefill segment through the
/// same-sequence lane and report how many tokens it consumed.
///
/// Chunking performs the same computation the serial path performs: the same
/// physical state advances in the same position order, and recurrent layers
/// still walk their convolution and GDN state one token at a time. What it
/// changes is that projections and feed-forward work for `members` positions
/// run as one batched forward, so weight reads are amortized across prompt
/// positions. State position advances once per completed chunk, so a chunk
/// that fails leaves the recorded position at its start rather than claiming
/// progress the device did not finish.
fn run_prefill_chunks(
    lane: &mut CudaQwen35BatchDecode,
    physical: &mut CudaHybridState,
    segment: &ExecutionSegment,
    tokens: &[u32],
) -> Result<PrefillChunks, BackendError> {
    let members = lane.members();
    let (chunked, sampled_final) = chunked_prefix_len(
        tokens.len(),
        members,
        segment.requests_sampling(),
        lane.can_sample_final_chunk(),
    );
    let mut offset = 0;
    while offset < chunked {
        let position = segment
            .state_position()
            .checked_add(u32::try_from(offset).map_err(|_| {
                BackendError::ExecutionFailed("prefill offset overflowed".to_owned())
            })?)
            .ok_or_else(|| {
                BackendError::ExecutionFailed("prefill position overflowed".to_owned())
            })?;
        lane.prefill_chunk(physical, &tokens[offset..offset + members], position)
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
        let next = position
            .checked_add(u32::try_from(members).map_err(|_| {
                BackendError::ExecutionFailed("prefill chunk length overflowed".to_owned())
            })?)
            .ok_or_else(|| {
                BackendError::ExecutionFailed("prefill position overflowed".to_owned())
            })?;
        physical
            .advance_to(next)
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
        offset += members;
    }
    Ok(PrefillChunks {
        consumed: chunked,
        sampled_final,
    })
}

/// Where one scheduler row's sampled output goes.
enum RowOutput<'a> {
    /// Blocking greedy readback, used by the eager dispatch path.
    Readback,
    /// Stream-ordered greedy copy into dispatcher-owned pinned memory.
    Pinned(&'a mut PinnedHostSlice<u32>),
    /// The row did not request sampling, so only its state advances.
    Unsampled,
}

/// Run the one model step that produces a row's output token: the final
/// prefill token or the decode token.
fn run_sample_step(
    executor: &mut CudaQwen35Decode,
    physical: &mut CudaHybridState,
    token: u32,
    position: u32,
    output: &mut RowOutput<'_>,
) -> Result<Option<u32>, BackendError> {
    let result = match output {
        RowOutput::Readback => executor.decode_step(physical, token, position).map(Some),
        RowOutput::Pinned(destination) => executor
            .decode_step_into_pinned(physical, token, position, destination)
            .map(|()| None),
        RowOutput::Unsampled => executor
            .prefill_step(physical, token, position)
            .map(|()| None),
    };
    result.map_err(|error| BackendError::ExecutionFailed(error.to_string()))
}

/// Sample a final prompt token from the chunk row that already advanced its
/// state, without consuming the token a second time.
fn run_chunk_sample(
    executor: &mut CudaQwen35Decode,
    lane: &CudaQwen35BatchDecode,
    member: usize,
    output: &mut RowOutput<'_>,
) -> Result<Option<u32>, BackendError> {
    let result = match output {
        RowOutput::Readback => executor.sample_prefill_chunk_row(lane, member).map(Some),
        RowOutput::Pinned(destination) => executor
            .sample_prefill_chunk_row_into_pinned(lane, member, destination)
            .map(|()| None),
        RowOutput::Unsampled => Ok(None),
    };
    result.map_err(|error| BackendError::ExecutionFailed(error.to_string()))
}

/// Run one prefill segment: leading whole chunks, the serial remainder, and
/// the optional sampled final chunk row.
fn run_prefill_row(
    executor: &mut CudaQwen35Decode,
    mut prefill_lane: Option<&mut CudaQwen35BatchDecode>,
    physical: &mut CudaHybridState,
    segment: &ExecutionSegment,
    work: Option<&mut DispatchWork>,
    output: &mut RowOutput<'_>,
) -> Result<Option<u32>, BackendError> {
    let mut output_token = None;

    let tokens = segment
        .token_input()
        .and_then(engine_core::ExecutionTokenInput::prompt_slice)
        .ok_or_else(|| {
            BackendError::ExecutionFailed(
                "Qwen prefill segment requires prompt token input".to_owned(),
            )
        })?;
    // Whole leading chunks run through the same-sequence lane; the
    // remainder, and most sampling tokens, stay on the serial path.
    let chunks = match prefill_lane.as_deref_mut() {
        Some(lane) => run_prefill_chunks(lane, physical, segment, tokens)?,
        None => PrefillChunks {
            consumed: 0,
            sampled_final: false,
        },
    };
    let (chunked, members) = (
        chunks.consumed,
        prefill_lane.as_ref().map(|lane| lane.members()),
    );
    if chunks.sampled_final {
        // The last chunk already advanced state through the final
        // prompt token; only its output head remains. The lane is
        // present because sampled_final requires one.
        let lane = prefill_lane.as_deref().ok_or_else(|| {
            BackendError::ExecutionFailed(
                "sampled prefill chunk requires a configured lane".to_owned(),
            )
        })?;
        // Batch rows are chunk-local, so the sampled row is the lane's last
        // row, not the prompt's last offset.
        output_token = run_chunk_sample(executor, lane, lane.members().saturating_sub(1), output)?;
    }
    for (offset, &token) in tokens.iter().enumerate().skip(chunked) {
        let offset = u32::try_from(offset)
            .map_err(|_| BackendError::ExecutionFailed("prefill offset overflowed".to_owned()))?;
        let position = segment
            .state_position()
            .checked_add(offset)
            .ok_or_else(|| {
                BackendError::ExecutionFailed("prefill position overflowed".to_owned())
            })?;
        let requests_output = segment.requests_sampling() && offset + 1 == segment.token_count();
        if requests_output {
            output_token = run_sample_step(executor, physical, token, position, output)?;
        } else {
            executor
                .prefill_step(physical, token, position)
                .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
        }
        let next = position.checked_add(1).ok_or_else(|| {
            BackendError::ExecutionFailed("prefill position overflowed".to_owned())
        })?;
        physical
            .advance_to(next)
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
    }
    if let Some(work) = work {
        work.merge(DispatchWork::prefill(tokens.len(), chunked, members));
    }

    Ok(output_token)
}

/// Run one scheduler row's model steps.
///
/// `output` selects how the row's sampled token leaves the executor: a
/// blocking readback, a stream-ordered copy into pinned memory, or nothing
/// when the row does not request sampling.
///
/// Returns the row's sampled token, if this call read it back, and host
/// wall-clock enqueue interval. Optional work accounting stays
/// submission-local until the entire enqueue and completion-event recording
/// succeed.
fn run_row(
    executor: &mut CudaQwen35Decode,
    prefill_lane: Option<&mut CudaQwen35BatchDecode>,
    registry: &mut CudaStateRegistry,
    segment: &ExecutionSegment,
    state: &InferenceStateSet,
    work: Option<&mut DispatchWork>,
    mut output: RowOutput<'_>,
) -> Result<Option<u32>, BackendError> {
    if let Some(sampling) = segment.sampling()
        && sampling.temperature() != 0.0
    {
        return Err(BackendError::Unsupported(
            "non-greedy Qwen3.8 CUDA sampling on the correctness path",
        ));
    }

    let physical = registry
        .get_or_create(state)
        .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;

    let output_token = match segment.phase() {
        ExecutionPhase::Prefill => {
            run_prefill_row(executor, prefill_lane, physical, segment, work, &mut output)?
        }
        ExecutionPhase::Decode => {
            let token = segment
                .token_input()
                .and_then(engine_core::ExecutionTokenInput::decode_token)
                .ok_or_else(|| {
                    BackendError::ExecutionFailed(
                        "Qwen decode segment requires one decode token".to_owned(),
                    )
                })?;
            let output_token = run_sample_step(
                executor,
                physical,
                token,
                segment.state_position(),
                &mut output,
            )?;
            let next = segment.state_position().checked_add(1).ok_or_else(|| {
                BackendError::ExecutionFailed("decode position overflowed".to_owned())
            })?;
            physical
                .advance_to(next)
                .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
            if let Some(work) = work {
                work.merge(DispatchWork::decode(1));
            }
            output_token
        }
        ExecutionPhase::SpecDraft
        | ExecutionPhase::SpecVerify
        | ExecutionPhase::Encoder
        | ExecutionPhase::MoEExpert => {
            return Err(BackendError::Unsupported(
                "this Qwen3.8 CUDA serving execution phase",
            ));
        }
    };

    Ok(output_token)
}

impl CudaQwen35ServingDispatcher {
    /// Build the dispatcher with a pool of `max_pending_rows` pinned output
    /// slots. The pool bounds the number of sampling rows that can be
    /// simultaneously in flight across all live submissions.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::ExecutionFailed`] when the pinned pool cannot
    /// be allocated or is empty.
    pub fn new(
        context: &Arc<CudaContext>,
        executor: CudaQwen35Decode,
        stream: Arc<CudaStream>,
        max_pending_rows: usize,
    ) -> Result<Self, BackendError> {
        if context.as_ref() != stream.context().as_ref() || !Arc::ptr_eq(&stream, executor.stream())
        {
            return Err(BackendError::ExecutionFailed(
                "dispatcher and executor must share one CUDA stream/context".to_owned(),
            ));
        }
        if max_pending_rows == 0 {
            return Err(BackendError::ExecutionFailed(
                "the serving dispatcher requires at least one pinned output slot".to_owned(),
            ));
        }
        let mut pinned_outputs = Vec::with_capacity(max_pending_rows);
        for _ in 0..max_pending_rows {
            // Safety: a slot is written only by the stream-ordered copy of
            // the submission that leased it, and is read only after that
            // submission's completion event covers the copy. Slots in the
            // free pool are never read.
            let slot = unsafe { context.alloc_pinned::<u32>(1) }
                .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
            pinned_outputs.push(slot);
        }
        let mut batched = HashMap::new();
        for members in 2..=max_pending_rows.min(crate::quantized::MAX_BATCH_MEMBERS) {
            let batch = CudaQwen35BatchDecode::from_decode(&executor, members)
                .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
            batched.insert(members, batch);
        }
        Ok(Self {
            executor,
            states: CudaStateRegistry::new(stream.clone()),
            stream,
            pinned_outputs,
            free_outputs: (0..max_pending_rows).rev().collect(),
            submissions: crate::submissions::Submissions::default(),
            deferred_releases: Vec::new(),
            batched,
            prefill_lane: None,
            grouped_decode: false,
            diagnostics: None,
        })
    }

    /// Enable bounded observation of future asynchronous submit attempts. Repeated
    /// calls share the same lifetime counters. Does not enable another execution lane.
    #[must_use]
    pub fn observe_dispatch(&mut self) -> DispatchObserver {
        self.diagnostics
            .get_or_insert_with(DispatchObserver::new)
            .clone()
    }

    /// Opt into experimental decode subgroups within heterogeneous/mixed offers.
    /// Device and performance qualification is pending; disabled by default.
    #[must_use]
    pub fn with_grouped_decode(mut self, enabled: bool) -> Self {
        self.grouped_decode = enabled;
        self
    }

    /// Enable same-sequence prefill chunking with `members` tokens per chunk.
    ///
    /// The lane is a copy of the single-row executor's model bindings: it
    /// shares compiled kernels and staged weights and adds only the per-lane
    /// scratch for `members` positions. It does not change the scheduler
    /// contract, the state registry, or the output path - a prefill segment is
    /// still one segment with one outcome, and only the leading whole chunks
    /// leave the serial path.
    ///
    /// Chunk sizes above the backend's `MAX_BATCH_MEMBERS` limit are rejected
    /// because no qualified lane exists for them.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::ExecutionFailed`] for a chunk smaller than two
    /// tokens, an unsupported member count, or an allocation failure.
    pub fn with_prefill_chunk(mut self, members: usize) -> Result<Self, BackendError> {
        if members < 2 {
            return Err(BackendError::ExecutionFailed(
                "prefill chunking requires at least two tokens per chunk; leave the lane unset for serial prefill"
                    .to_owned(),
            ));
        }
        let lane = CudaQwen35BatchDecode::from_decode(&self.executor, members)
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
        self.prefill_lane = Some(lane);
        Ok(self)
    }

    /// Prefer sampling the final prompt token from the last whole prefill chunk
    /// instead of running that token again on the serial path.
    ///
    /// This is a preference, not a capability request. The lane honours it only
    /// while it runs warp projections without the chunk-parallel recurrent scan;
    /// any other mode keeps the previous serial final step. Without a configured
    /// prefill lane there is nothing to sample from, so the preference is inert.
    #[must_use]
    pub fn with_sampled_prefill_final(mut self, enabled: bool) -> Self {
        if let Some(lane) = self.prefill_lane.as_mut() {
            lane.set_sampled_final_chunk(enabled);
        }
        self
    }

    /// The configured same-sequence prefill chunk size, if any.
    #[must_use]
    pub const fn prefill_chunk_members(&self) -> Option<usize> {
        match &self.prefill_lane {
            Some(lane) => Some(lane.members()),
            None => None,
        }
    }

    #[must_use]
    pub const fn state_registry(&self) -> &CudaStateRegistry {
        &self.states
    }

    #[must_use]
    pub fn pending_submissions(&self) -> usize {
        self.submissions.pending.len()
    }

    fn check_health(&self) -> Result<(), BackendError> {
        match self.submissions.fault() {
            Some(fault) => Err(BackendError::ExecutionFailed(format!(
                "CUDA dispatcher is unusable after a device failure: {fault}"
            ))),
            None => Ok(()),
        }
    }

    /// Establish completion and settle failed-drain quarantine for shutdown.
    /// Healthy pending outcomes remain available to poll; a faulted dispatcher
    /// stays closed to execution even when its resources become safe to retire.
    ///
    /// # Errors
    /// A failed barrier retains quarantine and prohibits physical release.
    pub fn synchronize(&mut self) -> Result<(), BackendError> {
        self.stream
            .synchronize()
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
        let retired = self.submissions.retire_after_barrier();
        self.retire_submissions(retired);
        Ok(())
    }

    /// A driver fault ends normal polling, but not necessarily device access.
    /// Drain before retiring leases; quarantine them if completion is uncertain.
    fn fail_device(&mut self, error: BackendError) -> BackendError {
        let flush = self.stream.synchronize();
        let retired = self.submissions.fail(error.to_string(), flush.is_ok());
        if let Err(flush_error) = flush {
            return BackendError::ExecutionFailed(format!(
                "{error}; CUDA drain failed: {flush_error}; resources quarantined until a successful drain"
            ));
        }
        self.retire_submissions(retired);
        error
    }

    fn retire_submissions(&mut self, retired: Vec<PendingSubmission>) {
        for record in retired {
            for lease in record.outputs.into_iter().flatten() {
                self.recycle_output(lease);
            }
        }
        self.drain_deferred_releases();
    }

    fn lease_output(&mut self) -> Result<PinnedSlotLease, BackendError> {
        let index = self.free_outputs.pop().ok_or_else(|| {
            BackendError::ExecutionFailed(
                "pinned Qwen output slots exhausted; too many concurrent pending rows".to_owned(),
            )
        })?;
        Ok(PinnedSlotLease { index })
    }

    fn recycle_output(&mut self, lease: PinnedSlotLease) {
        self.free_outputs.push(lease.index);
    }

    /// Query one submission's completion event. `is_complete` swallows the
    /// distinction between "not ready" and "device fault", so the raw query
    /// re-distinguishes them: only `NOT_READY` maps to pending.
    fn completion_is_ready(event: &CudaEvent) -> Result<bool, BackendError> {
        if event.is_complete() {
            return Ok(true);
        }
        match unsafe { cudarc::driver::result::event::query(event.cu_event()) } {
            Ok(()) => Ok(true),
            Err(error) if error.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => {
                Ok(false)
            }
            Err(error) => Err(BackendError::ExecutionFailed(error.to_string())),
        }
    }

    /// Eager per-row execution: run the row through the blocking decode step
    /// and synchronize the stream, so no device work outlives the call.
    fn dispatch_segment_eager(
        &mut self,
        segment: &ExecutionSegment,
        state: &InferenceStateSet,
    ) -> Result<ExecutionOutcome, BackendError> {
        let Self {
            executor,
            prefill_lane,
            states: registry,
            ..
        } = self;
        let result = run_row(
            executor,
            prefill_lane.as_mut(),
            registry,
            segment,
            state,
            None,
            RowOutput::Readback,
        );
        if let Err(error) = self.stream.synchronize() {
            return Err(self.fail_device(BackendError::ExecutionFailed(error.to_string())));
        }
        let output_token = result?;
        let outcome = ExecutionOutcome::new();
        Ok(match output_token {
            Some(token) => outcome.with_output_token(token),
            None => outcome,
        })
    }

    /// KV model layout, excluding independently owned token capacities.
    fn decode_kv_layout(
        state: &InferenceStateSet,
    ) -> Option<(u16, u16, u16, engine_core::DataType)> {
        state.kv().map(|kv| {
            let spec = kv.spec();
            (
                spec.layer_count(),
                spec.kv_heads(),
                spec.head_dim(),
                spec.dtype(),
            )
        })
    }

    /// First failed whole-batch decode selection condition. Single rows retain
    /// the batch-1 oracle; mixed phases, unsupported decode shapes and incompatible
    /// KV model layouts retain the row-wise path. Sampling values and other
    /// model/device validation still belong to enqueue.
    fn whole_decode_ineligibility(
        batch: &ExecutionBatch,
        states: &[InferenceStateSet],
    ) -> Option<DecodeFallback> {
        if batch.len() == 1 {
            return Some(DecodeFallback::SingleRow);
        }
        if batch.len() > crate::quantized::MAX_BATCH_MEMBERS {
            return Some(DecodeFallback::RowLimit);
        }
        let kv_layout = states.first().and_then(Self::decode_kv_layout);
        for (segment, state) in batch.segments().iter().zip(states) {
            if segment.phase() != ExecutionPhase::Decode {
                return Some(DecodeFallback::NonDecodePhase);
            }
            if segment.token_count() != 1
                || !segment.requests_sampling()
                || segment
                    .token_input()
                    .and_then(engine_core::ExecutionTokenInput::decode_token)
                    .is_none()
            {
                return Some(DecodeFallback::DecodeShape);
            }
            if state.token_position().is_none() {
                return Some(DecodeFallback::MissingPosition);
            }
            if Self::decode_kv_layout(state) != kv_layout {
                return Some(DecodeFallback::IncompatibleKv);
            }
        }
        None
    }

    /// Gather compatible decode rows in stable scheduler order. Requests have
    /// independent state, so device execution may group nonadjacent rows while
    /// completion stays in original row order. A prefill never enters this lane.
    fn decode_group(
        segments: &[ExecutionSegment],
        states: &[InferenceStateSet],
        scheduled: &[bool],
        first: usize,
    ) -> Vec<usize> {
        let eligible = |index: usize| {
            let segment = &segments[index];
            segment.phase() == ExecutionPhase::Decode
                && segment.token_count() == 1
                && segment.requests_sampling()
                && segment
                    .token_input()
                    .and_then(engine_core::ExecutionTokenInput::decode_token)
                    .is_some()
                && states[index].token_position().is_some()
                && states[index].kv().is_some()
        };
        if !eligible(first) {
            return vec![first];
        }
        let layout = Self::decode_kv_layout(&states[first]);
        (first..segments.len())
            .filter(|&index| {
                !scheduled[index]
                    && eligible(index)
                    && Self::decode_kv_layout(&states[index]) == layout
            })
            .take(crate::quantized::MAX_BATCH_MEMBERS)
            .collect()
    }

    /// Queue one fully-decode batch through the batched executor.
    ///
    /// Member physical states are taken out of the registry for the step
    /// (the registry allows one mutable borrow at a time), run as one
    /// submission's worth of batched launches, and re-inserted on every
    /// path so no state is lost. Pinned copies stay stream-ordered behind
    /// the step's kernels; the caller records the completion event.
    ///
    /// Greedy is the only supported sampling on this path, matching the
    /// batch-1 row seam.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::ExecutionFailed`] when member geometry,
    /// construction, or launch fails; [`BackendError::Unsupported`] for
    /// non-greedy sampling. On failure the stream is flushed and all
    /// leased slots recycled.
    #[allow(
        clippy::too_many_lines,
        reason = "one full batched submit seam; splitting it hides the registry take/reinsert contract"
    )]
    fn enqueue_batched_decode(
        &mut self,
        segments: &[&ExecutionSegment],
        states: &[&InferenceStateSet],
        work: Option<&mut DispatchWork>,
    ) -> Result<QueuedRows, BackendError> {
        let row_count = segments.len();
        let mut leased: Vec<Option<PinnedSlotLease>> = Vec::with_capacity(row_count);
        let mut state_keys = Vec::with_capacity(row_count);

        // Member physical states live outside the step closure so every
        // return path - success, launch failure, even a panic-free error
        // after the take - can re-insert them; dropping a taken state would
        // silently lose the request's device history.
        let mut members: Vec<CudaHybridState> = Vec::with_capacity(row_count);

        let result = (|| -> Result<(), BackendError> {
            // Reject non-greedy sampling before touching any device state.
            for segment in segments {
                if let Some(sampling) = segment.sampling()
                    && sampling.temperature() != 0.0
                {
                    return Err(BackendError::Unsupported(
                        "non-greedy Qwen3.8 CUDA sampling on the correctness path",
                    ));
                }
            }

            // Lease one pinned slot per sampling row, in batch order.
            for segment in segments {
                if segment.requests_sampling() {
                    leased.push(Some(self.lease_output()?));
                } else {
                    leased.push(None);
                }
            }

            // Take member physical states out of the registry in row order;
            // the registry allows one mutable borrow at a time.
            for state in states {
                let key = CudaStateKey::from_state_set(state);
                let Some(physical) = self.states.take(&key) else {
                    return Err(BackendError::ExecutionFailed(
                        "batched decode found no physical state for a decode row".to_owned(),
                    ));
                };
                members.push(physical);
                state_keys.push(key);
            }

            let executor = self.batched.get_mut(&row_count).ok_or_else(|| {
                BackendError::ExecutionFailed("batch shape was not prepared".to_owned())
            })?;

            // Assemble the step's tokens and positions.
            let mut tokens = Vec::with_capacity(row_count);
            let mut positions = Vec::with_capacity(row_count);
            for segment in segments {
                let token = segment
                    .token_input()
                    .and_then(engine_core::ExecutionTokenInput::decode_token)
                    .ok_or_else(|| {
                        BackendError::ExecutionFailed(
                            "Qwen decode segment requires one decode token".to_owned(),
                        )
                    })?;
                tokens.push(token);
                positions.push(segment.state_position());
            }

            // One submission's worth of batched launches, then per-member
            // pinned copies, all stream-ordered.
            let mut member_refs: Vec<&mut CudaHybridState> = members.iter_mut().collect();
            executor
                .decode_step_batch_enqueue(&mut member_refs, &tokens, &positions)
                .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
            for (member, lease) in leased.iter().enumerate() {
                let Some(lease) = lease else { continue };
                let pinned = self.pinned_outputs.get_mut(lease.index).ok_or_else(|| {
                    BackendError::ExecutionFailed(
                        "pinned Qwen output slot was out of range".to_owned(),
                    )
                })?;
                executor
                    .copy_selected_into_pinned(member, pinned)
                    .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
            }

            // Advance every member state by one position; re-insertion is
            // owned by the caller below so error paths also restore state.
            for (physical, position) in members.iter_mut().zip(positions.iter()) {
                let next = position.checked_add(1).ok_or_else(|| {
                    BackendError::ExecutionFailed("decode position overflowed".to_owned())
                })?;
                physical
                    .advance_to(next)
                    .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
            }

            Ok(())
        })();

        // Re-insert every taken member state before reporting success or
        // failure: the registry is the single owner of physical state, and a
        // dropped member would lose the request's device history.
        for (state, physical) in states.iter().zip(members) {
            let key = CudaStateKey::from_state_set(state);
            self.states.reinsert(key, physical);
        }

        match result {
            Ok(()) => {
                if let Some(work) = work {
                    work.merge(DispatchWork::decode(row_count));
                }
                Ok(QueuedRows {
                    outputs: leased,
                    state_keys,
                })
            }
            Err(error) => {
                if let Err(sync_error) = self.stream.synchronize() {
                    return Err(self.fail_device(BackendError::ExecutionFailed(format!(
                        "{error}; CUDA flush after batched enqueue failure also failed: {sync_error}"
                    ))));
                }
                for lease in leased.iter().flatten() {
                    self.recycle_output(*lease);
                }
                Err(error)
            }
        }
    }

    /// Queue every row of one scheduler batch without blocking the host.
    ///
    /// On success, all row work and pinned output copies are enqueued on the
    /// serving stream. On failure the stream is flushed and every leased slot
    /// is recycled before the error is returned, leaving no retained
    /// asynchronous ownership.
    fn enqueue_batch(
        &mut self,
        batch: &ExecutionBatch,
        states: &mut [InferenceStateSet],
        mut work: Option<&mut DispatchWork>,
    ) -> Result<QueuedRows, BackendError> {
        let row_count = batch.len();
        let segments = batch.segments();
        let mut queued_rows = QueuedRows {
            outputs: vec![None; row_count],
            state_keys: Vec::with_capacity(row_count),
        };
        let mut scheduled = vec![false; row_count];

        let enqueue_result = (|| -> Result<(), BackendError> {
            for first in 0..row_count {
                if scheduled[first] {
                    continue;
                }
                let group = self
                    .grouped_decode
                    .then(|| Self::decode_group(segments, states, &scheduled, first));
                if let Some(group) = group.filter(|group| group.len() > 1) {
                    let group_segments = group
                        .iter()
                        .map(|&index| &segments[index])
                        .collect::<Vec<_>>();
                    let group_states = group
                        .iter()
                        .map(|&index| &states[index])
                        .collect::<Vec<_>>();
                    let queued = self.enqueue_batched_decode(
                        &group_segments,
                        &group_states,
                        work.as_deref_mut(),
                    )?;
                    queued_rows.scatter_group(&group, queued);
                    for index in group {
                        scheduled[index] = true;
                    }
                    continue;
                }
                let segment = &segments[first];
                let state = &states[first];
                let lease = if segment.requests_sampling() {
                    Some(self.lease_output()?)
                } else {
                    None
                };
                // Retain the lease before any fallible launch so partial rows
                // and previously enqueued groups share the outer retirement guard.
                queued_rows.outputs[first] = lease;
                // Disjoint field borrows: the executor, the state registry,
                // and the leased pinned slot are independent of each other,
                // so one row can hold all three at once.
                let Self {
                    executor,
                    prefill_lane,
                    states: registry,
                    pinned_outputs,
                    ..
                } = self;
                let mut pinned = lease
                    .map(|lease| {
                        pinned_outputs.get_mut(lease.index).ok_or_else(|| {
                            BackendError::ExecutionFailed(
                                "pinned Qwen output slot was out of range".to_owned(),
                            )
                        })
                    })
                    .transpose()?;
                let output = match &mut pinned {
                    Some(pinned) => RowOutput::Pinned(pinned),
                    None => RowOutput::Unsampled,
                };
                let _ = run_row(
                    executor,
                    prefill_lane.as_mut(),
                    registry,
                    segment,
                    state,
                    work.as_deref_mut(),
                    output,
                )?;
                queued_rows
                    .state_keys
                    .push(CudaStateKey::from_state_set(state));
                scheduled[first] = true;
            }
            Ok(())
        })();

        if let Err(error) = enqueue_result {
            if let Err(sync_error) = self.stream.synchronize() {
                return Err(self.fail_device(BackendError::ExecutionFailed(format!(
                    "{error}; CUDA flush after enqueue failure also failed: {sync_error}"
                ))));
            }
            for lease in queued_rows.outputs.into_iter().flatten() {
                self.recycle_output(lease);
            }
            return Err(error);
        }

        Ok(queued_rows)
    }

    fn enqueue_submission(
        &mut self,
        submission: BackendSubmissionId,
        batch: &ExecutionBatch,
        states: &mut [InferenceStateSet],
        route: DispatchRoute,
        work: Option<&mut DispatchWork>,
    ) -> Result<(), BackendError> {
        let rows = if matches!(route, DispatchRoute::WholeDecode) {
            let segments = batch.segments().iter().collect::<Vec<_>>();
            let states = states.iter().collect::<Vec<_>>();
            self.enqueue_batched_decode(&segments, &states, work)?
        } else {
            self.enqueue_batch(batch, states, work)?
        };
        // Every kernel and pinned copy precedes this event in stream order.
        let completion = match self.executor.record_completion_event() {
            Ok(event) => event,
            Err(error) => {
                let flush = self.stream.synchronize().map_err(|sync_error| {
                    BackendError::ExecutionFailed(format!(
                        "event recording failed: {error}; CUDA flush also failed: {sync_error}"
                    ))
                });
                if let Err(error) = flush {
                    return Err(self.fail_device(error));
                }
                for lease in rows.outputs.iter().filter_map(|lease| *lease) {
                    self.recycle_output(lease);
                }
                return Err(BackendError::ExecutionFailed(error.to_string()));
            }
        };
        self.submissions.pending.insert(
            submission,
            PendingSubmission {
                completion,
                outputs: rows.outputs,
                state_keys: rows.state_keys,
            },
        );
        Ok(())
    }

    /// Read one leased pinned output slot. Safe only after the owning
    /// submission's completion event covers the stream-ordered copy into the
    /// slot; `as_slice` additionally synchronizes the slot's own event.
    fn read_pinned_output(&self, lease: PinnedSlotLease) -> Result<u32, BackendError> {
        let slot = self
            .pinned_outputs
            .get(lease.index)
            .ok_or_else(|| {
                BackendError::ExecutionFailed("pinned output lease was out of range".to_owned())
            })?
            .as_slice()
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()))?;
        slot.first()
            .copied()
            .ok_or_else(|| BackendError::ExecutionFailed("pinned output slot was empty".to_owned()))
    }

    /// Release every deferred state key no longer referenced by any pending
    /// submission.
    fn drain_deferred_releases(&mut self) {
        let deferred = std::mem::take(&mut self.deferred_releases);
        for key in deferred {
            let still_referenced = self
                .submissions
                .pending
                .values()
                .any(|pending| pending.state_keys.contains(&key));
            if still_referenced {
                self.deferred_releases.push(key);
            } else {
                self.states.release_key(&key);
            }
        }
    }
}

impl NvidiaDispatcher for CudaQwen35ServingDispatcher {
    fn dispatch(
        &mut self,
        segment: &ExecutionSegment,
        _weights: &WeightBinding,
        state: &mut InferenceStateSet,
    ) -> Result<ExecutionOutcome, BackendError> {
        self.check_health()?;
        self.dispatch_segment_eager(segment, state)
    }

    fn dispatch_batch(
        &mut self,
        batch: &ExecutionBatch,
        _weights: &WeightBinding,
        states: &mut [InferenceStateSet],
    ) -> Result<Vec<ExecutionOutcome>, BackendError> {
        self.check_health()?;
        if batch.len() != states.len() {
            return Err(BackendError::StateCountMismatch);
        }
        let outcomes = batch
            .segments()
            .iter()
            .zip(states.iter_mut())
            .map(|(segment, state)| self.dispatch_segment_eager(segment, state))
            .collect::<Result<Vec<_>, _>>();
        let sync = self
            .stream
            .synchronize()
            .map_err(|error| BackendError::ExecutionFailed(error.to_string()));
        match (outcomes, sync) {
            (Ok(values), Ok(())) => Ok(values),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(sync_error)) => Err(self.fail_device(sync_error)),
            (Err(error), Err(sync_error)) => Err(self.fail_device(BackendError::ExecutionFailed(
                format!("{error}; CUDA completion synchronization also failed: {sync_error}"),
            ))),
        }
    }

    fn submit_batch(
        &mut self,
        submission: BackendSubmissionId,
        batch: &ExecutionBatch,
        _weights: &WeightBinding,
        states: &mut [InferenceStateSet],
    ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
        self.check_health()?;
        if batch.len() != states.len() {
            return Err(BackendError::StateCountMismatch);
        }
        if self.submissions.pending.contains_key(&submission) {
            return Err(BackendError::ExecutionFailed(format!(
                "duplicate asynchronous submission {submission:?}"
            )));
        }

        let started = self.diagnostics.as_ref().map(|_| Instant::now());
        let route = if self.grouped_decode {
            DispatchRoute::Grouped
        } else {
            Self::whole_decode_ineligibility(batch, states)
                .map_or(DispatchRoute::WholeDecode, DispatchRoute::RowWise)
        };
        let mut work = self.diagnostics.as_ref().map(|_| DispatchWork::default());
        let result = self.enqueue_submission(submission, batch, states, route, work.as_mut());
        if let (Some(observer), Some(started)) = (&self.diagnostics, started) {
            observer.record(
                batch,
                route,
                result.as_ref().ok().and(work),
                u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            );
        }
        result.map(|()| None)
    }

    fn poll_batch(
        &mut self,
        submission: BackendSubmissionId,
    ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
        self.check_health()?;
        let ready = {
            let Some(pending) = self.submissions.pending.get(&submission) else {
                return Err(BackendError::UnknownSubmission(submission));
            };
            Self::completion_is_ready(&pending.completion)
        };
        match ready {
            Ok(false) => return Ok(None),
            Err(error) => return Err(self.fail_device(error)),
            Ok(true) => {}
        }

        // The completion event covers every kernel and pinned copy of this
        // submission; taking the record ends async ownership of its leases
        // and retained state keys even on the error paths below. The leases
        // are recycled before outcome construction so a read failure cannot
        // strand pool slots.
        let pending = self
            .submissions
            .pending
            .remove(&submission)
            .expect("presence was checked above");
        let result = pending
            .outputs
            .iter()
            .map(|lease| {
                let outcome = ExecutionOutcome::new();
                match lease {
                    Some(lease) => self
                        .read_pinned_output(*lease)
                        .map(|token| outcome.with_output_token(token)),
                    None => Ok(outcome),
                }
            })
            .collect::<Result<Vec<_>, BackendError>>();
        // Retain the removed record too if a pinned read exposes a driver fault.
        let outcomes = match result {
            Ok(outcomes) => outcomes,
            Err(error) => {
                self.submissions.quarantine(pending);
                return Err(self.fail_device(error));
            }
        };
        for lease in pending.outputs.iter().flatten() {
            self.recycle_output(*lease);
        }
        drop(pending);
        self.drain_deferred_releases();

        Ok(Some(outcomes))
    }

    fn release_inference_state(&mut self, state: &InferenceStateSet) -> Result<(), BackendError> {
        if !self.submissions.release_is_safe() {
            return Err(BackendError::ExecutionFailed(
                "CUDA resource release awaits a successful dispatcher drain".to_owned(),
            ));
        }
        // Physical release while queued device work still accesses this
        // state is deferred to the referencing submission's completion.
        let key = CudaStateKey::from_state_set(state);
        let referenced = self
            .submissions
            .pending
            .values()
            .any(|pending| pending.state_keys.contains(&key));
        if referenced {
            self.deferred_releases.push(key);
        } else {
            self.states.release_key(&key);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Arc, CudaQwen35ServingDispatcher, chunked_prefix_len};
    #[test]
    fn nonadjacent_group_slots_return_to_original_positions() {
        use super::{PinnedSlotLease, QueuedRows};
        let mut rows = QueuedRows {
            outputs: vec![None; 3],
            state_keys: Vec::new(),
        };
        rows.scatter_group(
            &[0, 2],
            QueuedRows {
                outputs: vec![
                    Some(PinnedSlotLease { index: 7 }),
                    Some(PinnedSlotLease { index: 3 }),
                ],
                state_keys: Vec::new(),
            },
        );
        assert_eq!(
            rows.outputs
                .iter()
                .map(|slot| slot.map(|slot| slot.index))
                .collect::<Vec<_>>(),
            [Some(7), None, Some(3)]
        );
    }

    #[test]
    fn decode_groups_preserve_order_and_concrete_kv_compatibility() {
        use super::DecodeFallback;
        for (capacities, prefill, expected, fallback) in [
            (vec![4], None, vec![0], Some(DecodeFallback::SingleRow)),
            (vec![4, 4], None, vec![0, 1], None),
            (vec![4, 5], None, vec![0, 1], None),
            (vec![4, 5, 4], None, vec![0, 1, 2], None),
            (
                vec![4, 4, 4],
                Some(1),
                vec![0, 2],
                Some(DecodeFallback::NonDecodePhase),
            ),
            (vec![4; 8], None, (0..8).collect(), None),
            (
                vec![4; 9],
                None,
                (0..8).collect(),
                Some(DecodeFallback::RowLimit),
            ),
        ] {
            let specs = capacities
                .into_iter()
                .map(|capacity| {
                    engine_core::KvStateSpec::new(1, 1, 2, capacity, engine_core::DataType::F16)
                        .unwrap()
                })
                .collect();
            assert_decode_selection(specs, prefill, &expected, fallback);
        }
    }

    #[test]
    fn decode_groups_reject_incompatible_kv_layouts() {
        use engine_core::{DataType, KvStateSpec};
        let base = KvStateSpec::new(1, 1, 2, 4, DataType::F16).unwrap();
        for other in [
            KvStateSpec::new(2, 1, 2, 4, DataType::F16).unwrap(),
            KvStateSpec::new(1, 2, 2, 4, DataType::F16).unwrap(),
            KvStateSpec::new(1, 1, 3, 4, DataType::F16).unwrap(),
            KvStateSpec::new(1, 1, 2, 4, DataType::F32).unwrap(),
        ] {
            assert_decode_selection(
                vec![base, other, base.with_capacity(8).unwrap()],
                None,
                &[0, 2],
                Some(super::DecodeFallback::IncompatibleKv),
            );
        }
    }

    fn assert_decode_selection(
        specs: Vec<engine_core::KvStateSpec>,
        prefill: Option<usize>,
        expected: &[usize],
        fallback: Option<super::DecodeFallback>,
    ) {
        use engine_core::{
            DeviceId, ExecutionBatch, ExecutionPhase, ExecutionSegment, ExecutionTokenInput,
            LogicalStateManager, RequestId, SamplingParams, StateRequirement,
        };
        let device = DeviceId::new(0);
        let mut manager =
            LogicalStateManager::new(device, ribn_foundation::BytePool::new(1024).shared());

        let mut states = specs
            .into_iter()
            .map(|spec| {
                manager
                    .allocate_set(&[StateRequirement::FullAttentionKv(spec)])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let positions = vec![1; states.len()];
        manager.commit_batch(&mut states, &positions).unwrap();
        let segments = states
            .iter()
            .enumerate()
            .map(|(index, state)| {
                ExecutionSegment::new(
                    RequestId::new(u64::try_from(index + 1).unwrap()).unwrap(),
                    if Some(index) == prefill {
                        ExecutionPhase::Prefill
                    } else {
                        ExecutionPhase::Decode
                    },
                    1,
                    1,
                    1,
                    state.requirements(),
                )
                .unwrap()
                .with_token_input(if Some(index) == prefill {
                    ExecutionTokenInput::prompt(Arc::from([7, 7]), 1, 1).unwrap()
                } else {
                    ExecutionTokenInput::Decode { token: 7 }
                })
                .unwrap()
                .with_sampling(SamplingParams::greedy(None))
            })
            .collect();
        let batch = ExecutionBatch::new(segments).unwrap();
        assert_eq!(
            CudaQwen35ServingDispatcher::whole_decode_ineligibility(&batch, &states),
            fallback
        );
        let mut scheduled = vec![false; states.len()];
        assert_eq!(
            CudaQwen35ServingDispatcher::decode_group(batch.segments(), &states, &scheduled, 0),
            expected
        );
        for &index in expected {
            scheduled[index] = true;
        }
        if let Some(first) = scheduled.iter().position(|&done| !done) {
            let group = CudaQwen35ServingDispatcher::decode_group(
                batch.segments(),
                &states,
                &scheduled,
                first,
            );
            assert!(group.iter().all(|&index| !scheduled[index]));
            assert_eq!(group[0], first);
        }
        for state in &states {
            manager.release_set(state).unwrap();
        }
    }

    #[test]
    fn decode_fallback_distinguishes_unsupported_rows_from_missing_state() {
        use super::DecodeFallback;
        use engine_core::{
            ExecutionBatch, ExecutionPhase, ExecutionSegment, ExecutionTokenInput,
            InferenceStateSet, RequestId, SamplingParams,
        };
        for (sample, input, tokens, expected) in [
            (false, true, 1, DecodeFallback::DecodeShape),
            (true, false, 1, DecodeFallback::DecodeShape),
            (true, false, 2, DecodeFallback::DecodeShape),
            (true, true, 1, DecodeFallback::MissingPosition),
        ] {
            let batch = ExecutionBatch::new(
                (1..=2)
                    .map(|id| {
                        let mut segment = ExecutionSegment::new(
                            RequestId::new(id).unwrap(),
                            ExecutionPhase::Decode,
                            1,
                            tokens,
                            0,
                            vec![],
                        )
                        .unwrap();
                        if sample {
                            segment = segment.with_sampling(SamplingParams::greedy(None));
                        }
                        if input {
                            segment = segment
                                .with_token_input(ExecutionTokenInput::Decode { token: 7 })
                                .unwrap();
                        }
                        segment
                    })
                    .collect(),
            )
            .unwrap();
            let states = (0..2)
                .map(|_| InferenceStateSet::new(vec![]).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                CudaQwen35ServingDispatcher::whole_decode_ineligibility(&batch, &states),
                Some(expected)
            );
        }
    }

    #[test]
    fn chunked_prefix_keeps_the_sampling_token_serial() {
        for (token_count, members, expected) in [
            (1, 8, 0),
            (7, 8, 0),
            (8, 8, 0),
            (9, 8, 8),
            (15, 8, 8),
            (16, 8, 8),
            (17, 8, 16),
            (32, 8, 24),
            (16, 4, 12),
            (3, 4, 0),
        ] {
            assert_eq!(
                chunked_prefix_len(token_count, members, true, false),
                (expected, false),
                "sampling prefill of {token_count} tokens in chunks of {members}"
            );
        }
    }

    /// Only a whole-chunk prompt on a qualified lane may put the final
    /// sampling token in the last chunk. Partial prompts still leave their
    /// final token serial, and unqualified lanes never sample a chunk row.
    #[test]
    fn sampled_final_chunk_requires_a_whole_prompt_and_a_qualified_lane() {
        for (token_count, members, sampling, sample_final, expected) in [
            (16, 8, true, true, (16, true)),
            (8, 8, true, true, (8, true)),
            (32, 8, true, true, (32, true)),
            (16, 4, true, true, (16, true)),
            (17, 8, true, true, (16, false)),
            (21, 8, true, true, (16, false)),
            (16, 8, true, false, (8, false)),
            (16, 8, false, true, (16, false)),
            (7, 8, true, true, (0, false)),
            (0, 8, true, true, (0, false)),
            (16, 0, true, true, (0, false)),
        ] {
            assert_eq!(
                chunked_prefix_len(token_count, members, sampling, sample_final),
                expected,
                "prefill of {token_count} tokens in chunks of {members}: sampling {sampling}, qualified lane {sample_final}"
            );
        }
    }

    /// Without sampling, every token is chunkable, and the result is always a
    /// whole number of chunks that does not exceed the segment.
    #[test]
    fn chunked_prefix_never_exceeds_whole_chunks() {
        for (token_count, members, expected) in [
            (0, 8, 0),
            (7, 8, 0),
            (8, 8, 8),
            (16, 8, 16),
            (20, 8, 16),
            (16, 4, 16),
            (3, 4, 0),
            (48, 8, 48),
        ] {
            assert_eq!(
                chunked_prefix_len(token_count, members, false, false),
                (expected, false),
                "unsampled prefill of {token_count} tokens in chunks of {members}"
            );
        }
    }
}
