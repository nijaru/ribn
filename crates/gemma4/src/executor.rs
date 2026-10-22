//! Runtime executor over the Gemma 4 sequence decoders.
//!
//! The decoders are synchronous, so one submission completes the offered rows in
//! `poll`. Successful calls establish completion; a failed call may retain device
//! work. Weights are shared across sequences and each admitted sequence owns its
//! cache and position. Unretired failures keep the complete execution owner.

use crate::cpu::Gemma4CpuDecoder;
use crate::{Gemma4Config, Gemma4CpuWeights, Gemma4Error};
use ribn::{
    Admission, BatchItem, BatchPreparation, ExecutionError, ExecutorInfo, GenerationExecutor,
    GenerationLimits, SequenceId, StepCompletion, SubmissionId, TokenRequest,
};
use std::collections::HashMap;
use std::sync::Arc;

mod batch;
#[cfg(test)]
mod tests;

/// Sequence-level decoder operations; the executor does not inspect backend state.
/// Successful calls establish completion. An error need not establish retirement,
/// so the generic executor treats decoder errors and unwinds as owner-fatal.
pub trait GemmaDecode: Send {
    /// Append a prompt chunk and return logits for its last token.
    /// # Errors
    /// Rejects invalid tokens or a chunk past the model's context.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>, Gemma4Error>;
    /// Append one generated token and return the next logits.
    /// # Errors
    /// Rejects an invalid token or a step past the model's context.
    fn decode(&mut self, token: u32) -> Result<Vec<u16>, Gemma4Error>;
    /// Tokens currently owned by this continuation.
    fn position(&self) -> usize;
}

impl GemmaDecode for Gemma4CpuDecoder {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>, Gemma4Error> {
        Self::prefill(self, tokens)
    }
    fn decode(&mut self, token: u32) -> Result<Vec<u16>, Gemma4Error> {
        Self::decode(self, token)
    }
    fn position(&self) -> usize {
        Self::position(self)
    }
}

#[cfg(feature = "cuda")]
impl GemmaDecode for crate::Gemma4CudaDecoder {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>, Gemma4Error> {
        Self::prefill(self, tokens)
    }
    fn decode(&mut self, token: u32) -> Result<Vec<u16>, Gemma4Error> {
        Self::decode(self, token)
    }
    fn position(&self) -> usize {
        Self::position(self)
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl GemmaDecode for crate::Gemma4MetalDecoder {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>, Gemma4Error> {
        Self::prefill(self, tokens)
    }
    fn decode(&mut self, token: u32) -> Result<Vec<u16>, Gemma4Error> {
        Self::decode(self, token)
    }
    fn position(&self) -> usize {
        Self::position(self)
    }
}

type Factory<D> = Box<dyn FnMut() -> Result<D, Gemma4Error> + Send>;
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
    Closed,
}

/// One retirement envelope, including the factory's shared model ownership.
struct Execution<D> {
    factory: Factory<D>,
    states: HashMap<SequenceId, SequenceState<D>>,
    items: Vec<BatchItem>,
    phase: Phase,
    next: u64,
}

/// Greedy executor over one Gemma 4 package.
///
/// Whole-batch preparation precedes execution. Successful calls are synchronous;
/// a failed batch is retirement-only and cannot be replayed or released. Backend
/// recovery is not yet implemented: shutdown reports failure and defensive Drop
/// quarantines the complete owner rather than claiming completion.
pub struct Gemma4Executor<D> {
    info: ExecutorInfo,
    config: Gemma4Config,
    // Present throughout ordinary use; taken only by defensive Drop.
    execution: Option<Execution<D>>,
}

impl<D> Drop for Gemma4Executor<D> {
    fn drop(&mut self) {
        if self
            .execution
            .as_ref()
            .is_some_and(|e| e.phase == Phase::Failed)
        {
            std::mem::forget(self.execution.take());
        }
    }
}

impl Gemma4Executor<Gemma4CpuDecoder> {
    /// Bind the executor to host CPU weights shared by every sequence.
    /// # Errors
    /// Rejects a configuration whose context bound exceeds the runtime's u32 limits.
    pub fn cpu(weights: Arc<Gemma4CpuWeights>) -> Result<Self, Gemma4Error> {
        let config = weights.config().clone();
        Self::new(
            "gemma4 cpu",
            config.max_position_embeddings(),
            config,
            Box::new(move || Gemma4CpuDecoder::shared(weights.clone())),
        )
    }
}

#[cfg(feature = "cuda")]
impl Gemma4Executor<crate::Gemma4CudaDecoder> {
    /// Bind the executor to an uploaded CUDA model shared by every sequence.
    /// # Errors
    /// Rejects a poisoned model lock or an invalid prepared cache bound.
    pub fn cuda(model: Arc<std::sync::Mutex<crate::Gemma4CudaModel>>) -> Result<Self, Gemma4Error> {
        let (config, capacity) = {
            let model = model
                .lock()
                .map_err(|_| Gemma4Error::new("executor", "CUDA model lock poisoned"))?;
            (model.config().clone(), model.max_positions())
        };
        Self::new(
            "gemma4 cuda",
            capacity,
            config,
            Box::new(move || crate::Gemma4CudaDecoder::shared(model.clone())),
        )
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl Gemma4Executor<crate::Gemma4MetalDecoder> {
    /// Bind the executor to an uploaded Metal model shared by every sequence.
    /// # Errors
    /// Rejects a configuration whose context bound exceeds the runtime's u32 limits.
    pub fn metal(model: Arc<crate::Gemma4MetalModel>) -> Result<Self, Gemma4Error> {
        let config = model.config().clone();
        Self::new(
            "gemma4 metal",
            config.max_position_embeddings(),
            config,
            Box::new(move || crate::Gemma4MetalDecoder::shared(model.clone())),
        )
    }
}

impl<D: GemmaDecode> Gemma4Executor<D> {
    fn new(
        name: &str,
        capacity: usize,
        config: Gemma4Config,
        factory: Factory<D>,
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
                factory,
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
        let decoder = (owner.factory)().map_err(|error| execution(&error))?;
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
            Phase::Submitted(_) | Phase::Failed => {
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
        if owner.phase == Phase::Failed || owner.items.iter().any(|item| item.sequence == sequence)
        {
            return Err(ExecutionError::new(
                "cannot release prepared, submitted or failed state",
            ));
        }
        owner.states.remove(&sequence);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        let owner = self.execution.as_mut().expect("executor owns execution");
        if owner.phase == Phase::Failed {
            return Err(ExecutionError::new(
                "failed Gemma batch retained: backend retirement is not implemented",
            ));
        }
        // Submitted work has not executed until poll. Successful polls are
        // synchronous; only those cases permit ordinary state destruction.
        owner.items.clear();
        owner.states.clear();
        owner.phase = Phase::Closed;
        Ok(())
    }
}
