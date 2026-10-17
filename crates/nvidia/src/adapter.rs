//! NVIDIA submission adapter. Host-reference dispatchers need no CUDA library.

use std::collections::HashMap;

use engine_core::DeviceId;
use engine_core::backend::{BackendError, BackendSubmissionId, ComputeBackend};
use engine_core::execution::{
    ExecutionBatch, ExecutionBatchEvent, ExecutionEvent, ExecutionOutcome, ExecutionSegment,
};
use engine_core::state::InferenceStateSet;
use engine_core::weights::WeightBinding;

pub trait NvidiaDispatcher: Send {
    /// Dispatch one request segment.
    ///
    /// This remains the compatibility primitive for simple/reference
    /// dispatchers. Production dispatchers can override [`Self::dispatch_batch`]
    /// to consume the scheduler's whole multi-request batch in one backend
    /// operation.
    ///
    /// # Errors
    ///
    /// Returns a backend error when CUDA dispatch or state mutation fails.
    fn dispatch(
        &mut self,
        segment: &ExecutionSegment,
        weights: &WeightBinding,
        state: &mut InferenceStateSet,
    ) -> Result<ExecutionOutcome, BackendError>;

    /// Release dispatcher-owned physical state associated with one logical
    /// inference-state set. Stateless/reference dispatchers use the no-op.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the dispatcher cannot release the state.
    /// Release must be idempotent after success: logical release can fail and
    /// cause a later retry. An error must retain resources still needed by work.
    fn release_inference_state(&mut self, _state: &InferenceStateSet) -> Result<(), BackendError> {
        Ok(())
    }

    /// Dispatch one scheduler-selected multi-request batch synchronously.
    ///
    /// The default implementation preserves reference/single-request
    /// dispatchers by visiting each segment in order. CUDA implementations
    /// can override this to consume the whole batch in one backend operation.
    /// Async-capable dispatchers should instead override [`Self::submit_batch`]
    /// and [`Self::poll_batch`].
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::StateCountMismatch`] when request/state counts
    /// differ, or propagates a segment dispatch failure. A synchronous
    /// implementation must not return while device work can still access the
    /// supplied request state.
    fn dispatch_batch(
        &mut self,
        batch: &ExecutionBatch,
        weights: &WeightBinding,
        states: &mut [InferenceStateSet],
    ) -> Result<Vec<ExecutionOutcome>, BackendError> {
        if batch.len() != states.len() {
            return Err(BackendError::StateCountMismatch);
        }
        batch
            .segments()
            .iter()
            .zip(states)
            .map(|(segment, state)| self.dispatch(segment, weights, state))
            .collect()
    }

    /// Submit one scheduler batch under the backend-owned submission identity.
    ///
    /// The compatibility implementation executes immediately and returns its
    /// outcomes. A genuinely asynchronous dispatcher returns `Ok(None)` after
    /// queueing device work and later exposes outcomes from [`Self::poll_batch`].
    /// The submission ID is stable across both calls and can key CUDA events,
    /// pinned output slots, or other backend-local completion state.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the batch cannot be accepted. Returning an
    /// error means the dispatcher did not retain asynchronous ownership: no
    /// queued work may still access the supplied request state or transient
    /// submission resources after this method returns, unless a device fault
    /// forced explicit quarantine. Quarantined ownership must prevent physical
    /// release and report an error through `release_inference_state`.
    fn submit_batch(
        &mut self,
        _submission: BackendSubmissionId,
        batch: &ExecutionBatch,
        weights: &WeightBinding,
        states: &mut [InferenceStateSet],
    ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
        self.dispatch_batch(batch, weights, states).map(Some)
    }

    /// Poll outcomes for a batch previously accepted asynchronously.
    ///
    /// Synchronous dispatchers never reach this method. Async dispatchers
    /// return `Ok(None)` while device work is pending and `Ok(Some(...))` once
    /// all outcomes for the scheduler batch are ready.
    ///
    /// # Errors
    ///
    /// Returns a backend error for terminal completion failure. An error is a
    /// terminal result for this submission, which the runtime will not poll
    /// again. The dispatcher must either end asynchronous access or retain
    /// uncertain device-owned resources in an explicit quarantine. While
    /// ownership is uncertain, `release_inference_state` must return an error
    /// so the runtime retains the logical owner for retry or teardown.
    fn poll_batch(
        &mut self,
        _submission: BackendSubmissionId,
    ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
        Err(BackendError::Unsupported(
            "asynchronous NVIDIA dispatcher completion",
        ))
    }
}

#[derive(Clone, Debug)]
struct PendingNvidiaSubmission {
    batch: ExecutionBatch,
}

/// Submission bookkeeping over one NVIDIA dispatcher.
///
/// The backend owns submission identity, completion bookkeeping and the weight binding the
/// dispatcher resolves names against. It is not a model or capability descriptor: the model
/// constructed it with the device and weights it prepared.
pub struct NvidiaBackend<D> {
    device: DeviceId,
    weights: WeightBinding,
    dispatcher: D,
    next_submission: u64,
    completed: HashMap<BackendSubmissionId, ExecutionBatchEvent>,
    pending: HashMap<BackendSubmissionId, PendingNvidiaSubmission>,
}

impl<D> NvidiaBackend<D> {
    #[must_use]
    pub fn new(device: DeviceId, weights: WeightBinding, dispatcher: D) -> Self {
        Self {
            device,
            weights,
            dispatcher,
            next_submission: 1,
            completed: HashMap::new(),
            pending: HashMap::new(),
        }
    }

    #[must_use]
    pub fn dispatcher(&self) -> &D {
        &self.dispatcher
    }

    #[must_use]
    pub fn dispatcher_mut(&mut self) -> &mut D {
        &mut self.dispatcher
    }

    #[must_use]
    pub fn into_dispatcher(self) -> D {
        self.dispatcher
    }

    fn allocate_submission(&mut self) -> Result<BackendSubmissionId, BackendError> {
        let id = BackendSubmissionId::new(self.next_submission).ok_or_else(|| {
            BackendError::ExecutionFailed("submission identity overflowed".to_owned())
        })?;
        self.next_submission = self.next_submission.checked_add(1).ok_or_else(|| {
            BackendError::ExecutionFailed("submission identity overflowed".to_owned())
        })?;
        Ok(id)
    }
}

fn completion_event(
    batch: &ExecutionBatch,
    outcomes: Vec<ExecutionOutcome>,
) -> Result<ExecutionBatchEvent, BackendError> {
    if outcomes.len() != batch.len() {
        return Err(BackendError::ExecutionFailed(format!(
            "NVIDIA dispatcher returned {} outcomes for {} request segments",
            outcomes.len(),
            batch.len()
        )));
    }

    let events = batch
        .segments()
        .iter()
        .zip(outcomes)
        .map(|(segment, outcome)| {
            let output_token = outcome.output_token();
            if segment.requests_sampling() != output_token.is_some() {
                return Err(BackendError::ExecutionFailed(format!(
                    "NVIDIA dispatcher output-token presence did not match sampling for request {}",
                    segment.request().get()
                )));
            }
            let event =
                ExecutionEvent::new(segment.request(), segment.phase(), segment.token_count())
                    .ok_or_else(|| {
                        BackendError::ExecutionFailed("segment contained no tokens".to_owned())
                    })?;
            Ok(match output_token {
                Some(token) => event.with_output_token(token),
                None => event,
            })
        })
        .collect::<Result<Vec<_>, BackendError>>()?;
    ExecutionBatchEvent::new(events).map_err(BackendError::InvalidBatch)
}

impl<D: NvidiaDispatcher> ComputeBackend for NvidiaBackend<D> {
    fn device(&self) -> DeviceId {
        self.device
    }

    fn release_inference_state(&mut self, state: &InferenceStateSet) -> Result<(), BackendError> {
        self.dispatcher.release_inference_state(state)
    }

    fn submit(
        &mut self,
        batch: &ExecutionBatch,
        states: &mut [InferenceStateSet],
    ) -> Result<BackendSubmissionId, BackendError> {
        ComputeBackend::validate_execution(self, batch, states)?;
        let submission = self.allocate_submission()?;
        let result = self
            .dispatcher
            .submit_batch(submission, batch, &self.weights, states)?;
        match result {
            Some(outcomes) => {
                let event = completion_event(batch, outcomes)?;
                self.completed.insert(submission, event);
            }
            None => {
                self.pending.insert(
                    submission,
                    PendingNvidiaSubmission {
                        batch: batch.clone(),
                    },
                );
            }
        }
        Ok(submission)
    }

    fn poll(
        &mut self,
        submission: BackendSubmissionId,
    ) -> Result<Option<ExecutionBatchEvent>, BackendError> {
        if let Some(event) = self.completed.remove(&submission) {
            return Ok(Some(event));
        }
        if !self.pending.contains_key(&submission) {
            return Err(BackendError::UnknownSubmission(submission));
        }
        let outcomes = match self.dispatcher.poll_batch(submission) {
            Ok(None) => return Ok(None),
            Ok(Some(outcomes)) => outcomes,
            Err(error) => {
                self.pending.remove(&submission);
                return Err(error);
            }
        };
        let pending = self
            .pending
            .remove(&submission)
            .ok_or(BackendError::UnknownSubmission(submission))?;
        completion_event(&pending.batch, outcomes).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::device::DeviceId;
    use engine_core::execution::ExecutionPhase;
    use engine_core::request::{RequestId, SamplingParams};
    use engine_core::state::{
        InferenceStateSet, KvStateSpec, LogicalStateManager, StateRequirement,
    };
    use engine_core::tensor::DataType;
    use std::collections::HashSet;

    struct TestDispatcher;

    impl NvidiaDispatcher for TestDispatcher {
        fn dispatch(
            &mut self,
            _segment: &ExecutionSegment,
            _weights: &WeightBinding,
            _state: &mut InferenceStateSet,
        ) -> Result<ExecutionOutcome, BackendError> {
            Ok(ExecutionOutcome::new())
        }
    }

    #[derive(Default)]
    struct BatchAwareDispatcher {
        batches: usize,
        segments: usize,
        releases: usize,
    }

    impl NvidiaDispatcher for BatchAwareDispatcher {
        fn dispatch(
            &mut self,
            _segment: &ExecutionSegment,
            _weights: &WeightBinding,
            _state: &mut InferenceStateSet,
        ) -> Result<ExecutionOutcome, BackendError> {
            self.segments += 1;
            Ok(ExecutionOutcome::new())
        }

        fn dispatch_batch(
            &mut self,
            batch: &ExecutionBatch,
            _weights: &WeightBinding,
            states: &mut [InferenceStateSet],
        ) -> Result<Vec<ExecutionOutcome>, BackendError> {
            if batch.len() != states.len() {
                return Err(BackendError::StateCountMismatch);
            }
            self.batches += 1;
            Ok((0..batch.len()).map(|_| ExecutionOutcome::new()).collect())
        }

        fn release_inference_state(
            &mut self,
            _state: &InferenceStateSet,
        ) -> Result<(), BackendError> {
            self.releases += 1;
            Ok(())
        }
    }

    #[derive(Default)]
    struct AsyncDispatcher {
        submits: usize,
        polls: usize,
        pending: HashMap<BackendSubmissionId, usize>,
    }

    impl NvidiaDispatcher for AsyncDispatcher {
        fn dispatch(
            &mut self,
            _segment: &ExecutionSegment,
            _weights: &WeightBinding,
            _state: &mut InferenceStateSet,
        ) -> Result<ExecutionOutcome, BackendError> {
            Err(BackendError::ExecutionFailed(
                "async dispatcher should not use synchronous dispatch".to_owned(),
            ))
        }

        fn submit_batch(
            &mut self,
            submission: BackendSubmissionId,
            batch: &ExecutionBatch,
            _weights: &WeightBinding,
            states: &mut [InferenceStateSet],
        ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
            if batch.len() != states.len() {
                return Err(BackendError::StateCountMismatch);
            }
            self.submits += 1;
            self.pending.insert(submission, batch.len());
            Ok(None)
        }

        fn poll_batch(
            &mut self,
            submission: BackendSubmissionId,
        ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
            self.polls += 1;
            let count = self
                .pending
                .remove(&submission)
                .ok_or(BackendError::UnknownSubmission(submission))?;
            Ok(Some((0..count).map(|_| ExecutionOutcome::new()).collect()))
        }
    }

    #[derive(Default)]
    struct FailingAsyncDispatcher {
        pending: HashSet<BackendSubmissionId>,
    }

    impl NvidiaDispatcher for FailingAsyncDispatcher {
        fn dispatch(
            &mut self,
            _segment: &ExecutionSegment,
            _weights: &WeightBinding,
            _state: &mut InferenceStateSet,
        ) -> Result<ExecutionOutcome, BackendError> {
            unreachable!("failing async dispatcher does not synchronously dispatch")
        }

        fn submit_batch(
            &mut self,
            submission: BackendSubmissionId,
            batch: &ExecutionBatch,
            _weights: &WeightBinding,
            states: &mut [InferenceStateSet],
        ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
            if batch.len() != states.len() {
                return Err(BackendError::StateCountMismatch);
            }
            self.pending.insert(submission);
            Ok(None)
        }

        fn poll_batch(
            &mut self,
            submission: BackendSubmissionId,
        ) -> Result<Option<Vec<ExecutionOutcome>>, BackendError> {
            self.pending
                .remove(&submission)
                .ok_or(BackendError::UnknownSubmission(submission))?;
            Err(BackendError::ExecutionFailed(
                "terminal async failure".to_owned(),
            ))
        }
    }

    fn empty_decode_fixture<D: NvidiaDispatcher>(
        dispatcher: D,
    ) -> (NvidiaBackend<D>, ExecutionBatch, Vec<InferenceStateSet>) {
        let device = DeviceId::new(0);
        let backend = NvidiaBackend::new(device, WeightBinding::empty(device), dispatcher);
        let batch = ExecutionBatch::new(vec![
            ExecutionSegment::new(
                RequestId::new(1).expect("request ID"),
                ExecutionPhase::Decode,
                1,
                1,
                0,
                Vec::new(),
            )
            .expect("segment"),
        ])
        .expect("batch");
        let states = vec![InferenceStateSet::new(Vec::new()).expect("state")];
        (backend, batch, states)
    }

    #[test]
    fn backend_dispatches_only_after_capability_and_state_validation() {
        let device = DeviceId::new(0);
        let mut backend = NvidiaBackend::new(device, WeightBinding::empty(device), TestDispatcher);
        let requirement = StateRequirement::FullAttentionKv(
            KvStateSpec::new(1, 1, 2, 1, DataType::F16).expect("KV spec"),
        );
        let request = RequestId::new(1).expect("request ID");
        let segment =
            ExecutionSegment::new(request, ExecutionPhase::Decode, 1, 1, 0, vec![requirement])
                .expect("segment");
        let batch = ExecutionBatch::new(vec![segment]).expect("batch");
        let spec = match requirement {
            StateRequirement::FullAttentionKv(spec) => spec,
            StateRequirement::Recurrent(_) => unreachable!(),
        };
        let mut manager =
            LogicalStateManager::new(device, ribn_foundation::BytePool::new(1024).shared());
        let kv = manager.allocate_kv(spec).expect("state allocation");
        let state = InferenceStateSet::try_new(Some(kv), None).expect("state set");
        let submission = backend.submit(&batch, &mut [state]).expect("submit");
        let _ = backend.wait(submission).expect("completion");
    }

    #[test]
    fn backend_rejects_missing_requested_output_token() {
        let device = DeviceId::new(0);
        let mut backend = NvidiaBackend::new(device, WeightBinding::empty(device), TestDispatcher);
        let segment = ExecutionSegment::new(
            RequestId::new(1).expect("request ID"),
            ExecutionPhase::Decode,
            1,
            1,
            0,
            Vec::new(),
        )
        .expect("segment")
        .with_sampling(SamplingParams::greedy(None));
        let batch = ExecutionBatch::new(vec![segment]).expect("batch");
        let mut states = vec![InferenceStateSet::new(Vec::new()).expect("state")];

        assert!(matches!(
            backend.submit(&batch, &mut states),
            Err(BackendError::ExecutionFailed(_))
        ));
    }

    #[test]
    fn backend_forwards_state_release_to_the_dispatcher() {
        let device = DeviceId::new(0);
        let mut backend = NvidiaBackend::new(
            device,
            WeightBinding::empty(device),
            BatchAwareDispatcher::default(),
        );
        let state = InferenceStateSet::new(Vec::new()).expect("state");

        backend
            .release_inference_state(&state)
            .expect("release state");
        assert_eq!(backend.dispatcher().releases, 1);
    }

    #[test]
    fn backend_forwards_the_whole_scheduler_batch_to_the_dispatcher() {
        let device = DeviceId::new(0);
        let mut backend = NvidiaBackend::new(
            device,
            WeightBinding::empty(device),
            BatchAwareDispatcher::default(),
        );
        let batch = ExecutionBatch::new(vec![
            ExecutionSegment::new(
                RequestId::new(1).expect("request ID"),
                ExecutionPhase::Decode,
                1,
                1,
                0,
                Vec::new(),
            )
            .expect("segment"),
            ExecutionSegment::new(
                RequestId::new(2).expect("request ID"),
                ExecutionPhase::Decode,
                1,
                1,
                0,
                Vec::new(),
            )
            .expect("segment"),
        ])
        .expect("batch");
        let empty_state = || InferenceStateSet::new(Vec::new()).expect("state");
        let mut states = vec![empty_state(), empty_state()];

        let submission = backend.submit(&batch, &mut states).expect("submit");
        let completed = backend.wait(submission).expect("completion");

        assert_eq!(completed.len(), 2);
        assert_eq!(
            completed
                .events()
                .iter()
                .map(|event| event.phase())
                .collect::<Vec<_>>(),
            [ExecutionPhase::Decode, ExecutionPhase::Decode]
        );
        assert_eq!(backend.dispatcher().batches, 1);
        assert_eq!(backend.dispatcher().segments, 0);
    }

    #[test]
    fn backend_supports_dispatcher_owned_async_completion() {
        let (mut backend, batch, mut states) = empty_decode_fixture(AsyncDispatcher::default());

        let submission = backend.submit(&batch, &mut states).expect("submit");
        assert_eq!(backend.dispatcher().submits, 1);
        assert_eq!(backend.dispatcher().polls, 0);

        let _ = backend
            .poll(submission)
            .expect("poll async dispatcher")
            .expect("completion");
        assert_eq!(backend.dispatcher().polls, 1);
        assert!(backend.pending.is_empty());
    }

    #[test]
    fn terminal_async_failure_releases_backend_pending_metadata() {
        let (mut backend, batch, mut states) =
            empty_decode_fixture(FailingAsyncDispatcher::default());
        let submission = backend.submit(&batch, &mut states).expect("submit");
        assert_eq!(backend.pending.len(), 1);

        assert!(matches!(
            backend.poll(submission),
            Err(BackendError::ExecutionFailed(_))
        ));
        assert!(backend.pending.is_empty());
        assert!(backend.dispatcher().pending.is_empty());
        assert!(matches!(
            backend.poll(submission),
            Err(BackendError::UnknownSubmission(actual)) if actual == submission
        ));
    }
}
