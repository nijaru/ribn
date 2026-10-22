//! Adversarial work crosses the real Gemma executor and CPU decoder boundary.
use super::*;
use ribn::{Engine, Event, GenerationOptions, RequestId, StepKind};
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "../../tests/common/weights.rs"]
mod weights;

#[derive(Default)]
struct Probe {
    calls: AtomicUsize,
    live: AtomicUsize,
}
struct CountedDecoder {
    inner: Gemma4CpuDecoder,
    probe: Arc<Probe>,
    failure: Option<Mode>,
}
impl CountedDecoder {
    fn enter(&self) -> Result<(), Gemma4Error> {
        let call = self.probe.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == 2 && self.failure.is_some() {
            assert_ne!(
                self.failure,
                Some(Mode::DropUnwind),
                "injected decoder unwind"
            );
            return Err(Gemma4Error::new("test boundary", "injected step failure"));
        }
        Ok(())
    }
}
impl Drop for CountedDecoder {
    fn drop(&mut self) {
        self.probe.live.fetch_sub(1, Ordering::SeqCst);
    }
}
impl GemmaDecode for CountedDecoder {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>, Gemma4Error> {
        self.enter()?;
        self.inner.prefill(tokens)
    }
    fn decode(&mut self, token: u32) -> Result<Vec<u16>, Gemma4Error> {
        self.enter()?;
        self.inner.decode(token)
    }
    fn position(&self) -> usize {
        self.inner.position()
    }
}

fn counted(probe: Arc<Probe>, failure: Option<Mode>) -> Gemma4Executor<CountedDecoder> {
    let dir = weights::Dir::new();
    let package = weights::package(&dir, &weights::inventory(), false);
    let source = crate::Gemma4Source::open(&package).unwrap();
    let host = Arc::new(Gemma4CpuWeights::load(&source).unwrap());
    Gemma4Executor::new(
        "counted CPU",
        host.config().max_position_embeddings(),
        host.config().clone(),
        Box::new(move || {
            let inner = Gemma4CpuDecoder::shared(host.clone())?;
            probe.live.fetch_add(1, Ordering::SeqCst);
            Ok(CountedDecoder {
                inner,
                probe: probe.clone(),
                failure,
            })
        }),
    )
    .unwrap()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Corruption {
    Prefix,
    Duplicate,
    ZeroInputs,
    FinalOutput,
    DecodeBeforePrompt,
    AggregateInputs,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Corrupt(Corruption),
    Lifecycle,
    Failure,
    DropFailure,
    DropUnwind,
}

struct Boundary {
    inner: Gemma4Executor<CountedDecoder>,
    mode: Mode,
    offered: Vec<BatchItem>,
}
impl Boundary {
    fn lifecycle_prepare(&mut self) {
        assert!(self.inner.submit(&self.offered).is_err());
        self.inner.prepare(&self.offered).unwrap();
        assert!(self.inner.prepare(&self.offered).is_err());
        assert!(self.inner.release(self.offered[0].sequence).is_err());
        let mut changed = self.offered.clone();
        changed[0].output_budget += 1;
        assert!(self.inner.submit(&changed).is_err());
        self.inner.abandon_preparation().unwrap();
        self.inner.abandon_preparation().unwrap();
    }
}
impl GenerationExecutor for Boundary {
    fn info(&self) -> &ExecutorInfo {
        self.inner.info()
    }
    fn admit(
        &mut self,
        request: RequestId,
        sequence: SequenceId,
        input: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        self.inner.admit(request, sequence, input)
    }
    fn prepare(&mut self, batch: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
        assert_eq!(batch.len(), 2);
        self.offered = batch.to_vec();
        if self.mode == Mode::Lifecycle {
            self.lifecycle_prepare();
        }
        let [first, second] = self.offered.as_mut_slice() else {
            unreachable!()
        };
        match self.mode {
            Mode::Corrupt(Corruption::Prefix) => second.prefix += 1,
            Mode::Corrupt(Corruption::Duplicate) => second.sequence = first.sequence,
            Mode::Corrupt(Corruption::ZeroInputs) => second.token_budget = 0,
            Mode::Corrupt(Corruption::FinalOutput) => second.output_budget = 0,
            Mode::Corrupt(Corruption::DecodeBeforePrompt) => {
                second.kind = StepKind::Decode;
                second.token_budget = 1;
                second.output_budget = 1;
            }
            Mode::Corrupt(Corruption::AggregateInputs) => {
                second.token_budget = 2;
                second.output_budget = 1;
            }
            _ => {}
        }
        self.inner.prepare(&self.offered)
    }
    fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
        self.inner.abandon_preparation()
    }
    fn submit(&mut self, _: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        let id = self.inner.submit(&self.offered)?;
        if self.mode == Mode::Lifecycle {
            assert!(self.inner.submit(&self.offered).is_err());
            assert!(self.inner.abandon_preparation().is_err());
            assert!(self.inner.release(self.offered[0].sequence).is_err());
            assert!(self.inner.poll(SubmissionId::new(id.get() + 1)).is_err());
        }
        Ok(id)
    }
    fn poll(&mut self, id: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        if matches!(self.mode, Mode::DropFailure | Mode::DropUnwind) {
            let mut owner = std::mem::replace(&mut self.inner, counted(Arc::default(), None));
            // Move the owner into the closure so its Drop runs during unwinding.
            return std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || owner.poll(id)))
                .unwrap_or_else(|_| Err(ExecutionError::new("caught decoder unwind")));
        }
        let result = self.inner.poll(id);
        if self.mode == Mode::Failure && result.is_err() {
            // These are the authoritative logical commitment fields, not physical
            // decoder progress: the earlier CPU decoder did execute successfully.
            let states = &self.inner.execution.as_ref().unwrap().states;
            assert!(states.values().all(|state| {
                state.cursor == 0 && state.input.is_none() && state.remaining == 2
            }));
            for item in &self.offered {
                assert!(self.inner.release(item.sequence).is_err());
            }
            assert!(self.inner.prepare(&self.offered).is_err());
            assert!(self.inner.submit(&self.offered).is_err());
            assert!(self.inner.poll(id).is_err());
        }
        let rows = result?;
        if self.mode == Mode::Lifecycle {
            assert!(self.inner.poll(id).is_err());
            let invalid_decode = rows
                .as_ref()
                .unwrap()
                .iter()
                .map(|row| BatchItem {
                    sequence: row.sequence,
                    kind: StepKind::Decode,
                    prefix: row.prefix,
                    token_budget: 2,
                    output_budget: 2,
                })
                .collect::<Vec<_>>();
            assert!(self.inner.prepare(&invalid_decode).is_err());
        }
        Ok(rows)
    }
    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        self.inner.release(sequence)?;
        self.inner.release(sequence)
    }
    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.inner.synchronize()
    }
}

fn engine(probe: Arc<Probe>, mode: Mode) -> Engine {
    let failure =
        matches!(mode, Mode::Failure | Mode::DropFailure | Mode::DropUnwind).then_some(mode);
    let mut inner = counted(probe, failure);
    if mode == Mode::Corrupt(Corruption::AggregateInputs) {
        inner.info.limits.max_batch_tokens = 4;
    }
    let mut engine = Engine::with_defaults(Boundary {
        inner,
        mode,
        offered: Vec::new(),
    })
    .unwrap();
    for prompt in [vec![1, 2, 3], vec![4, 5]] {
        engine
            .enqueue(TokenRequest::new(
                prompt,
                GenerationOptions {
                    max_output_tokens: 2,
                    ..GenerationOptions::default()
                },
            ))
            .unwrap();
    }
    engine
}

#[test]
fn malformed_later_rows_cannot_execute_an_earlier_continuation() {
    let mut violations = Vec::new();
    for corruption in [
        Corruption::Prefix,
        Corruption::Duplicate,
        Corruption::ZeroInputs,
        Corruption::FinalOutput,
        Corruption::DecodeBeforePrompt,
        Corruption::AggregateInputs,
    ] {
        let probe = Arc::new(Probe::default());
        let mut engine = engine(probe.clone(), Mode::Corrupt(corruption));
        let first = engine.step();
        let failed = first.is_err() || engine.step().is_err();
        let calls = probe.calls.load(Ordering::SeqCst);
        if calls != 0 || !failed {
            violations.push((corruption, calls, failed));
        }
        engine.shutdown().unwrap();
    }
    assert!(
        violations.is_empty(),
        "preflight violations: {violations:?}"
    );
}

#[test]
fn preparation_submission_and_poll_identity_are_one_transaction() {
    let probe = Arc::new(Probe::default());
    let mut engine = engine(probe.clone(), Mode::Lifecycle);
    let mut tokens = 0;
    for _ in 0..8 {
        engine.step().unwrap();
        while let Some(event) = engine.pop_event() {
            match event {
                Event::Token { .. } => tokens += 1,
                Event::Finished { reason, .. } => assert_eq!(reason, ribn::FinishReason::Length),
            }
        }
        if engine.status().requests == 0 {
            break;
        }
    }
    assert_eq!(tokens, 4);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 4);
    assert_eq!(probe.live.load(Ordering::SeqCst), 0);
    engine.shutdown().unwrap();
}

#[test]
fn partial_execution_failure_never_reports_cleanup_or_releases_owners() {
    let probe = Arc::new(Probe::default());
    let mut engine = engine(probe.clone(), Mode::Failure);
    engine.step().unwrap();
    assert!(engine.step().is_err());
    while let Some(event) = engine.pop_event() {
        assert!(matches!(
            event,
            Event::Finished {
                reason: ribn::FinishReason::Failed(_),
                ..
            }
        ));
    }
    for _ in 0..2 {
        assert!(engine.shutdown().is_err());
        assert_eq!(probe.live.load(Ordering::SeqCst), 2);
    }
    drop(engine);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        probe.live.load(Ordering::SeqCst),
        2,
        "complete owner quarantined"
    );
}

#[test]
fn direct_executor_drop_retains_failed_and_unwinding_batches() {
    for mode in [Mode::DropFailure, Mode::DropUnwind] {
        let probe = Arc::new(Probe::default());
        let mut engine = engine(probe.clone(), mode);
        engine.step().unwrap();
        assert!(engine.step().is_err());
        drop(engine);
        assert_eq!(probe.live.load(Ordering::SeqCst), 2, "{mode:?}");
    }
}
