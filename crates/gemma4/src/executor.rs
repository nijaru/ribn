//! Runtime executor over the Gemma 4 sequence decoders.
//!
//! The decoders are synchronous, so one submission completes the offered rows in
//! `poll`; there is no queued device work. Weights are shared across sequences and
//! each admitted sequence owns only its cache and position.

use crate::cpu::Gemma4CpuDecoder;
use crate::{Gemma4Config, Gemma4CpuWeights, Gemma4Error, argmax};
use ribn::{
    Admission, BatchItem, BatchPreparation, ExecutionError, ExecutorInfo, GenerationExecutor,
    GenerationLimits, SequenceId, StepCompletion, StepKind, SubmissionId, TokenRequest,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Sequence-level decoder operations; the executor does not inspect backend state.
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

struct SequenceState<D> {
    decoder: D,
    prompt: Arc<[u32]>,
    cursor: u32,
    input: Option<u32>,
    remaining: u32,
}

/// Greedy executor over one Gemma 4 package.
///
/// Admission validates the whole prompt and output budget, sequences share the
/// loaded weights, and completed rows release their decoder state. Submissions
/// complete synchronously, so cancellation before `poll` has no device work left
/// to observe.
pub struct Gemma4Executor<D> {
    info: ExecutorInfo,
    config: Gemma4Config,
    factory: Factory<D>,
    states: HashMap<SequenceId, SequenceState<D>>,
    batch: Option<Vec<BatchItem>>,
    next: u64,
}

impl Gemma4Executor<Gemma4CpuDecoder> {
    /// Bind the executor to host CPU weights shared by every sequence.
    /// # Errors
    /// Rejects a configuration whose context bound exceeds the runtime's u32 limits.
    pub fn cpu(weights: Arc<Gemma4CpuWeights>) -> Result<Self, Gemma4Error> {
        let config = weights.config().clone();
        Self::new(
            "gemma4 cpu",
            config,
            Box::new(move || Gemma4CpuDecoder::shared(weights.clone())),
        )
    }
}

#[cfg(feature = "cuda")]
impl Gemma4Executor<crate::Gemma4CudaDecoder> {
    /// Bind the executor to an uploaded CUDA model shared by every sequence.
    /// # Errors
    /// Rejects a poisoned model lock or a configuration whose context bound
    /// exceeds the runtime's u32 limits.
    pub fn cuda(model: Arc<std::sync::Mutex<crate::Gemma4CudaModel>>) -> Result<Self, Gemma4Error> {
        let config = model
            .lock()
            .map_err(|_| Gemma4Error::new("executor", "CUDA model lock poisoned"))?
            .config()
            .clone();
        Self::new(
            "gemma4 cuda",
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
            config,
            Box::new(move || crate::Gemma4MetalDecoder::shared(model.clone())),
        )
    }
}

impl<D: GemmaDecode> Gemma4Executor<D> {
    fn new(name: &str, config: Gemma4Config, factory: Factory<D>) -> Result<Self, Gemma4Error> {
        let context = u32::try_from(config.max_position_embeddings())
            .map_err(|_| Gemma4Error::new("executor", "context exceeds u32"))?;
        Ok(Self {
            info: ExecutorInfo {
                name: name.into(),
                limits: GenerationLimits {
                    context_tokens: context,
                    max_sequences: 4,
                    max_batch_tokens: 4096,
                    max_decode_tokens: 1,
                },
            },
            config,
            factory,
            states: HashMap::new(),
            batch: None,
            next: 0,
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
        if self.states.len() >= self.info.limits.max_sequences {
            return Err(ExecutionError::new("sequence capacity is exhausted"));
        }
        if self.states.contains_key(&sequence) {
            return Err(ExecutionError::new("sequence is already admitted"));
        }
        if self.config.vocab_size() < request.tokens.iter().copied().max().unwrap_or(0) as usize {
            return Err(ExecutionError::new("prompt token exceeds the vocabulary"));
        }
        let decoder = (self.factory)().map_err(|error| execution(&error))?;
        self.states.insert(
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

    fn prepare(&mut self, _batch: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
        Ok(BatchPreparation::Ready)
    }

    fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        if self.batch.is_some() {
            return Err(ExecutionError::new("a submission is already outstanding"));
        }
        self.batch = Some(batch.to_vec());
        self.next = self.next.wrapping_add(1);
        Ok(SubmissionId::new(self.next))
    }

    fn poll(
        &mut self,
        _submission: SubmissionId,
    ) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        let Some(batch) = self.batch.take() else {
            return Ok(None);
        };
        let mut rows = Vec::with_capacity(batch.len());
        for item in &batch {
            let state = self
                .states
                .get_mut(&item.sequence)
                .ok_or_else(|| ExecutionError::new("submitted sequence is not admitted"))?;
            if item.prefix != state.cursor {
                return Err(ExecutionError::new(
                    "submitted prefix disagrees with continuation",
                ));
            }
            if state.decoder.position() != state.cursor as usize {
                return Err(ExecutionError::new(
                    "decoder position disagrees with prefix",
                ));
            }
            let (logits, prefix, produced) = match item.kind {
                StepKind::Prefill => {
                    let start = state.cursor as usize;
                    let end = start
                        .checked_add(item.token_budget as usize)
                        .ok_or_else(|| ExecutionError::new("prefill range overflow"))?;
                    if end > state.prompt.len() {
                        return Err(ExecutionError::new("prefill range exceeds the prompt"));
                    }
                    let logits = state
                        .decoder
                        .prefill(&state.prompt[start..end])
                        .map_err(|error| execution(&error))?;
                    let prefix = u32::try_from(end)
                        .map_err(|_| ExecutionError::new("prefix exceeds u32"))?;
                    (logits, prefix, u32::from(end == state.prompt.len()))
                }
                StepKind::Decode => {
                    let input = state
                        .input
                        .ok_or_else(|| ExecutionError::new("decode before the first token"))?;
                    let logits = state
                        .decoder
                        .decode(input)
                        .map_err(|error| execution(&error))?;
                    (logits, item.prefix + 1, 1)
                }
            };
            let produced = produced.min(item.output_budget).min(state.remaining);
            let tokens = if produced == 0 {
                Vec::new()
            } else {
                let token =
                    argmax(&logits).ok_or_else(|| ExecutionError::new("logits are empty"))?;
                state.input = Some(token);
                state.remaining -= 1;
                vec![token]
            };
            state.cursor = prefix;
            rows.push(StepCompletion {
                sequence: item.sequence,
                prefix,
                tokens,
            });
        }
        Ok(Some(rows))
    }

    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        self.states.remove(&sequence);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.batch = None;
        self.states.clear();
        Ok(())
    }
}
