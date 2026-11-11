//! Test-only observation of runtime-issued opaque identity, not GPU simulation.
use ribn::{
    Admission, BatchItem, BatchPreparation, Engine, ExecutionError, ExecutorInfo,
    GenerationExecutor, GenerationLimits, GenerationOptions, RequestId, SequenceId, StepCompletion,
    SubmissionId, TokenRequest,
};
use std::sync::{Arc, Mutex};

// Every new model-wide gate in a given test binary shares this lease. Separate
// processes still require the documented external GPU serialization.
pub static DEVICE: Mutex<()> = Mutex::new(());

// Sequence IDs are opaque runtime authority, not numbers a model may forge.
// This fixture observes only the real Engine's admission boundary; it does not
// simulate any GPU operation, cache, completion or result being tested below.
pub fn issued_ids<const N: usize>() -> [SequenceId; N] {
    issued::<N>().map(|(_, sequence)| sequence)
}

pub fn issued<const N: usize>() -> [(RequestId, SequenceId); N] {
    struct Issuer {
        info: ExecutorInfo,
        ids: Arc<Mutex<Vec<(RequestId, SequenceId)>>>,
    }
    impl GenerationExecutor for Issuer {
        fn info(&self) -> &ExecutorInfo {
            &self.info
        }
        fn admit(
            &mut self,
            request: RequestId,
            id: SequenceId,
            _: &TokenRequest,
        ) -> Result<Admission, ExecutionError> {
            self.ids.lock().unwrap().push((request, id));
            Ok(Admission::Rejected(ExecutionError::new("identity fixture")))
        }
        fn prepare(&mut self, _: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
            unreachable!("no admitted state")
        }
        fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
            Ok(())
        }
        fn submit(&mut self, _: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
            unreachable!("no admitted state")
        }
        fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
            unreachable!("no admitted state")
        }
        fn release(&mut self, _: SequenceId) -> Result<(), ExecutionError> {
            Ok(())
        }
        fn synchronize(&mut self) -> Result<(), ExecutionError> {
            Ok(())
        }
    }
    let ids = Arc::new(Mutex::new(Vec::new()));
    let mut engine = Engine::with_defaults(Issuer {
        info: ExecutorInfo {
            name: "identity boundary".into(),
            limits: GenerationLimits {
                context_tokens: 2,
                max_sequences: N,
                max_batch_tokens: 2,
                max_decode_tokens: 1,
            },
        },
        ids: Arc::clone(&ids),
    })
    .unwrap();
    for _ in 0..N {
        engine
            .enqueue(TokenRequest::new(
                vec![1],
                GenerationOptions {
                    max_output_tokens: 1,
                    ..GenerationOptions::default()
                },
            ))
            .unwrap();
    }
    for _ in 0..N + 2 {
        engine.step().unwrap();
        if ids.lock().unwrap().len() == N {
            break;
        }
    }
    let ids = ids.lock().unwrap();
    std::array::from_fn(|index| ids[index])
}
