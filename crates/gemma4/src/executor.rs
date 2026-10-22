//! Runtime executor over the Gemma 4 sequence decoders.
//!
//! The decoders are synchronous, so one submission completes the offered rows in
//! `poll`. Successful calls establish completion; a failed call may retain device
//! work. Weights are shared across sequences and each admitted sequence owns its
//! cache and position. Unretired failures keep the complete execution owner.

use crate::{Gemma4Config, Gemma4Error};
use ribn::{
    Admission, BatchItem, BatchPreparation, ExecutionError, ExecutorInfo, GenerationExecutor,
    GenerationLimits, SequenceId, StepCompletion, SubmissionId, TokenRequest,
};
use std::collections::HashMap;
use std::sync::Arc;

mod backend;
mod batch;
#[cfg(test)]
mod tests;

use backend::Backend;
pub use backend::GemmaDecode;

const MAX_SEQUENCES: usize = 4;

struct SequenceState<D> {
    decoder: D,
    prompt: Arc<[u32]>,
    cursor: u32,
    input: Option<u32>,
    remaining: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Prepared,
    Submitted(SubmissionId),
    Failed,
    Retiring,
    Closed,
}

/// One retirement envelope, including the factory's shared model ownership.
struct Execution<D> {
    backend: Backend<D>,
    states: HashMap<SequenceId, SequenceState<D>>,
    items: Vec<BatchItem>,
    phase: Phase,
    next: u64,
}

/// Greedy executor over one Gemma 4 package.
///
/// Whole-batch preparation precedes execution. A failed continuation cannot resume,
/// but may be destroyed once backend completion is established. Failed teardown
/// retains ownership for retry; defensive Drop quarantines only unresolved cleanup.
pub struct Gemma4Executor<D: GemmaDecode> {
    info: ExecutorInfo,
    config: Gemma4Config,
    // Present throughout ordinary use; taken only by defensive Drop.
    execution: Option<Execution<D>>,
}

impl<D: GemmaDecode> Drop for Gemma4Executor<D> {
    fn drop(&mut self) {
        if let Some(execution) = self.execution.take() {
            // Cleanup can fail or unwind. Keep the complete envelope until its
            // completion and sequence release have both been acknowledged.
            let mut execution = std::mem::ManuallyDrop::new(execution);
            if execution.retire().is_ok() {
                drop(std::mem::ManuallyDrop::into_inner(execution));
            }
        }
    }
}

impl<D: GemmaDecode> Execution<D> {
    fn retire(&mut self) -> Result<(), Gemma4Error> {
        if self.phase == Phase::Closed {
            return Ok(());
        }
        self.phase = Phase::Retiring;
        (self.backend.settle)()?;
        self.items.clear();
        for state in self.states.values_mut() {
            state.decoder.close()?;
        }
        self.states.clear();
        self.phase = Phase::Closed;
        Ok(())
    }
}

impl<D: GemmaDecode> Gemma4Executor<D> {
    fn new(
        name: &str,
        capacity: usize,
        config: Gemma4Config,
        backend: Backend<D>,
    ) -> Result<Self, Gemma4Error> {
        if capacity == 0 || capacity > config.max_position_embeddings() {
            return Err(Gemma4Error::new(
                "executor",
                "invalid prepared context capacity",
            ));
        }
        let context = u32::try_from(capacity)
            .map_err(|_| Gemma4Error::new("executor", "context exceeds u32"))?;
        Ok(Self {
            info: ExecutorInfo {
                name: name.into(),
                limits: GenerationLimits {
                    context_tokens: context,
                    max_sequences: MAX_SEQUENCES,
                    max_batch_tokens: 4096,
                    max_decode_tokens: 1,
                },
            },
            config,
            execution: Some(Execution {
                backend,
                states: HashMap::new(),
                items: Vec::with_capacity(MAX_SEQUENCES),
                phase: Phase::Idle,
                next: 0,
            }),
        })
    }
}

fn execution(error: &Gemma4Error) -> ExecutionError {
    ExecutionError::new(error.to_string())
}

impl<D: GemmaDecode> GenerationExecutor for Gemma4Executor<D> {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn admit(
        &mut self,
        _request_id: ribn::RequestId,
        sequence: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        let owner = self.execution.as_mut().expect("executor owns execution");
        if owner.phase != Phase::Idle {
            return Err(ExecutionError::new(
                "Gemma execution is busy, faulted or closed",
            ));
        }
        if request.options.sampling.temperature != 0.0 {
            return Err(ExecutionError::new(
                "Gemma executor currently supports greedy sampling only",
            ));
        }
        let prompt = u32::try_from(request.tokens.len())
            .map_err(|_| ExecutionError::new("prompt length exceeds u32"))?;
        let output = request.options.max_output_tokens;
        if prompt == 0 || output == 0 {
            return Err(ExecutionError::new(
                "prompt and output budget must be positive",
            ));
        }
        let total = u64::from(prompt) + u64::from(output);
        if total > u64::from(self.info.limits.context_tokens) {
            return Err(ExecutionError::new(
                "prompt and output exceed the context bound",
            ));
        }
        if owner.states.len() >= self.info.limits.max_sequences {
            return Err(ExecutionError::new("sequence capacity is exhausted"));
        }
        if owner.states.contains_key(&sequence) {
            return Err(ExecutionError::new("sequence is already admitted"));
        }
        self.config
            .validate_text_tokens(&request.tokens)
            .map_err(|error| execution(&error))?;
        let decoder = (owner.backend.create)().map_err(|error| execution(&error))?;
        owner.states.insert(
            sequence,
            SequenceState {
                decoder,
                prompt: request.tokens.clone(),
                cursor: 0,
                input: None,
                remaining: output,
            },
        );
        Ok(Admission::Ready)
    }

    fn prepare(&mut self, batch: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
        let owner = self.execution.as_mut().expect("executor owns execution");
        if owner.phase != Phase::Idle {
            return Err(ExecutionError::new(
                "Gemma execution is busy, faulted or closed",
            ));
        }
        owner.validate(batch, self.info.limits)?;
        owner.items.extend_from_slice(batch);
        owner.phase = Phase::Prepared;
        Ok(BatchPreparation::Ready)
    }

    fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
        let owner = self.execution.as_mut().expect("executor owns execution");
        match owner.phase {
            Phase::Prepared => {
                owner.items.clear();
                owner.phase = Phase::Idle;
            }
            Phase::Idle | Phase::Closed => {}
            Phase::Submitted(_) | Phase::Failed | Phase::Retiring => {
                return Err(ExecutionError::new(
                    "cannot abandon submitted or failed work",
                ));
            }
        }
        Ok(())
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        let owner = self.execution.as_mut().expect("executor owns execution");
        if owner.phase != Phase::Prepared || owner.items != batch {
            return Err(ExecutionError::new(
                "submission differs from its preparation",
            ));
        }
        owner.next = owner.next.wrapping_add(1);
        let id = SubmissionId::new(owner.next);
        owner.phase = Phase::Submitted(id);
        Ok(id)
    }

    fn poll(
        &mut self,
        submission: SubmissionId,
    ) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        let owner = self.execution.as_mut().expect("executor owns execution");
        if owner.phase != Phase::Submitted(submission) {
            return Err(ExecutionError::new("unknown Gemma submission"));
        }
        // Includes unwinding: no row, shared model, or cache can escape this
        // retirement envelope after a decoder may have changed physical state.
        owner.phase = Phase::Failed;
        let rows = owner.execute()?;
        for row in &rows {
            let state = owner
                .states
                .get_mut(&row.sequence)
                .expect("validated sequence");
            state.cursor = row.prefix;
            if let Some(&token) = row.tokens.first() {
                state.input = Some(token);
                state.remaining -= 1;
            }
        }
        owner.items.clear();
        owner.phase = Phase::Idle;
        Ok(Some(rows))
    }

    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        let owner = self.execution.as_mut().expect("executor owns execution");
        if matches!(owner.phase, Phase::Failed | Phase::Retiring)
            || owner.items.iter().any(|item| item.sequence == sequence)
        {
            return Err(ExecutionError::new(
                "cannot release prepared, submitted or failed state",
            ));
        }
        if let Some(state) = owner.states.get_mut(&sequence) {
            let previous = owner.phase;
            owner.phase = Phase::Retiring;
            state.decoder.close().map_err(|error| execution(&error))?;
            owner.states.remove(&sequence);
            owner.phase = previous;
        }
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        let owner = self.execution.as_mut().expect("executor owns execution");
        owner.retire().map_err(|error| execution(&error))
    }
}
