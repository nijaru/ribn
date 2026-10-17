//! Host checks of the production CUDA assembly's runtime configuration.
//! Only the device executor is substituted; no CUDA context is initialized.

use ribn::{
    Admission, BatchItem, BatchPreparation, ExecutionError, ExecutorInfo, GenerationLimits,
    GenerationOptions, RequestId, SequenceId, StepCompletion, SubmissionId, TokenRequest,
};

use super::*;

struct PendingExecutor {
    info: ExecutorInfo,
}

impl PendingExecutor {
    fn new(max_sequences: usize) -> Self {
        Self {
            info: ExecutorInfo {
                name: "pending assembly fixture".into(),
                limits: GenerationLimits {
                    context_tokens: 32,
                    max_sequences,
                    max_batch_tokens: 3,
                    max_decode_tokens: 1,
                },
            },
        }
    }
}

impl GenerationExecutor for PendingExecutor {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn admit(
        &mut self,
        _: RequestId,
        _: SequenceId,
        _: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        Ok(Admission::Ready)
    }

    fn prepare(&mut self, _: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
        Ok(BatchPreparation::Ready)
    }

    fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        assert!(batch.iter().map(|item| item.token_budget).sum::<u32>() <= 3);
        Ok(SubmissionId::new(1))
    }

    fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        Ok(None)
    }

    fn release(&mut self, _: SequenceId) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        Ok(())
    }
}

#[test]
fn loaded_sequence_limit_controls_admission_without_a_hidden_permit_cap() {
    for sequences in [1, 8, 9, 16, 17, 32] {
        let config = engine_config(sequences).unwrap();
        let mut engine = configured_engine(PendingExecutor::new(sequences), config).unwrap();
        assert!(
            DriverConfig::for_engine(&engine).max_requests >= sequences,
            "application admission undercuts {sequences} active slots"
        );
        for _ in 0..=sequences {
            engine
                .enqueue(TokenRequest::new(
                    vec![1],
                    GenerationOptions {
                        max_output_tokens: 2,
                        ..GenerationOptions::default()
                    },
                ))
                .unwrap();
        }
        engine.step().unwrap();
        assert_eq!(engine.status().active_sequences, sequences);
        assert_eq!(engine.status().waiting, 1, "excess request remains queued");
        engine.shutdown().unwrap();
    }
}

#[test]
fn sampled_final_prefill_preference_is_on_by_default_and_can_be_disabled() {
    let defaults = LoadOptions::default();
    assert!(defaults.sampled_prefill_final);
    assert!(QwenLoadOptions::from(defaults).sampled_prefill_final);
    let disabled = QwenLoadOptions::from(LoadOptions {
        sampled_prefill_final: false,
        ..defaults
    });
    assert!(!disabled.sampled_prefill_final);
    assert_eq!(
        disabled.prefill_chunk_members,
        defaults.prefill_chunk_members
    );
}

#[test]
fn impossible_count_envelopes_reject_before_artifact_or_device_access() {
    for max_sequences in [0, usize::MAX, usize::MAX / 2] {
        let result = TextOwner::load(
            "nonexistent-capacity-fixture.gguf",
            LoadOptions {
                device: u16::MAX,
                max_sequences,
                ..LoadOptions::default()
            },
        );
        assert!(matches!(result, Err(TextError::InvalidInput(_))));
    }
}
