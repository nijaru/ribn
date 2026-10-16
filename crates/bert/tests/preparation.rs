//! Real host preparation and batch delivery; only device execution is substituted.
use std::num::NonZeroUsize;
use std::sync::Arc;

use engine_bert::{
    BertConfig, EncoderConstraint, EncoderPreparation, EncoderRequest, EncoderShape,
};
use ribn_batch::{
    BatchConfig, BatchExecutor, BatchRuntime, BatchSelection, EnqueueError, Job, JobOutput,
    Rejection, StepOutcome, Terminal,
};
use ribn_foundation::{BytePool, ParameterVersion};

struct HostEncoder {
    preparation: EncoderPreparation,
    executed: Vec<usize>,
}

impl BatchExecutor for HostEncoder {
    type Input = EncoderRequest;
    type Output = usize;
    type Error = EncoderConstraint;
    type Constraint = EncoderConstraint;

    fn parameter_version(&self) -> ParameterVersion {
        ParameterVersion::new(0)
    }

    fn max_batch_items(&self) -> usize {
        4
    }

    fn select_batch(&self, candidates: &[&EncoderRequest]) -> BatchSelection<EncoderConstraint> {
        self.preparation.select(candidates)
    }

    fn retained_bytes(&self, input: &EncoderRequest) -> u64 {
        self.preparation
            .shape(input.sequence())
            .map_or(0, EncoderShape::device_bytes)
    }

    fn execute(
        &mut self,
        jobs: Vec<Job<EncoderRequest>>,
    ) -> Result<Vec<JobOutput<usize>>, EnqueueError<EncoderConstraint>> {
        let mut results = Vec::with_capacity(jobs.len());
        for job in jobs {
            let (id, input, lease) = job.into_parts();
            let shape = self
                .preparation
                .shape_of(&input)
                .map_err(EnqueueError::Refused)?;
            self.executed.push(shape.sequence());
            results.push(JobOutput::new(id, shape.sequence(), lease));
        }
        Ok(results)
    }

    fn retire(&mut self, outputs: Vec<JobOutput<usize>>) {
        drop(outputs); // Host-only outputs have no outstanding device access.
    }
}

#[test]
fn a_backend_infeasible_request_does_not_fail_accepted_or_later_peers() {
    let config = BertConfig {
        vocab_size: 8,
        hidden_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 3,
        intermediate_size: 20,
        max_position_embeddings: 16,
        type_vocab_size: 2,
        layer_norm_eps: 1.0e-12,
    };
    let encoder = HostEncoder {
        preparation: EncoderPreparation::new(config, NonZeroUsize::new(4).unwrap()).unwrap(),
        executed: vec![],
    };
    let pool = BytePool::new(1 << 20).shared();
    let mut runtime = BatchRuntime::new(
        encoder,
        Arc::clone(&pool),
        BatchConfig {
            max_waiting_requests: 4,
            max_retained_results: 3,
        },
    )
    .unwrap();
    let before = runtime
        .submit(EncoderRequest::single_segment(vec![1; 4]))
        .unwrap();
    let bad = runtime
        .submit(EncoderRequest::single_segment(vec![1; 5]))
        .unwrap();
    let after = runtime
        .submit(EncoderRequest::single_segment(vec![1; 2]))
        .unwrap();

    assert_eq!(
        runtime.step().unwrap(),
        StepOutcome::Executed { results: 1 }
    );
    let completed = runtime.pop_completed().unwrap();
    assert_eq!(completed.request(), before);
    assert_eq!(completed.outcome(), &Terminal::Output(4));
    drop(completed);
    assert_eq!(pool.granted(), 0);

    assert_eq!(
        runtime.step().unwrap(),
        StepOutcome::Rejected { request: bad }
    );
    let rejected = runtime.pop_completed().unwrap();
    assert_eq!(rejected.request(), bad);
    assert_eq!(
        rejected.outcome(),
        &Terminal::Rejected(Rejection::Executor(
            EncoderConstraint::BackendSequenceLimit {
                sequence: 5,
                max_sequence: 4
            }
        ))
    );
    assert_eq!(pool.granted(), 0, "an infeasible head reserves nothing");

    assert_eq!(
        runtime.step().unwrap(),
        StepOutcome::Executed { results: 1 }
    );
    let completed = runtime.pop_completed().unwrap();
    assert_eq!(completed.request(), after);
    assert_eq!(completed.outcome(), &Terminal::Output(2));
    drop(completed);
    assert_eq!(runtime.executor().executed, [4, 2]);
    assert_eq!(pool.granted(), 0);
}
