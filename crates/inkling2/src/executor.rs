//! Model-local `ribn::GenerationExecutor` over the Inkling 2 CPU reference model.
//!
//! Shared physical weights (`Arc<Inkling2CpuModel>`) serve multiple admitted
//! sequences concurrently; each admitted sequence owns its own `n_slots` KV and
//! short-convolution history (`Inkling2CpuState`). Whole-offer validation runs
//! before any forward execution, and logical cursors commit only after every
//! offered row succeeds.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use ribn::{
    Admission, BatchItem, BatchPreparation, ExecutionError, ExecutorInfo, GenerationExecutor,
    GenerationLimits, Readiness, RequestId, Sampling, SequenceId, StepCompletion, StepKind,
    SubmissionId, TokenRequest, init_rng_state, sample_f32_logits,
};

use crate::Inkling2Error;
use crate::continuation::KvCachePrecision;
use crate::cpu::{Inkling2CpuModel, Inkling2CpuState};

struct LogicalSequence {
    prompt: Arc<[u32]>,
    cursor: u32,
    input: Option<u32>,
    remaining: u32,
}

enum RowInput {
    Prefill(Range<usize>),
    Decode(u32),
}

struct BoundRow {
    work: BatchItem,
    input: RowInput,
}

impl BoundRow {
    const fn end(&self) -> u32 {
        self.work.prefix + self.work.token_budget
    }
}

struct SequenceState {
    logical: LogicalSequence,
    cpu: Inkling2CpuState,
    sampling: Sampling,
    rng_state: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Idle,
    Prepared,
    Submitted(SubmissionId),
    Failed,
    Closed,
}

/// Autoregressive `GenerationExecutor` backed by [`Inkling2CpuModel`] with slot-capacity
/// deferral (`Admission::Deferred`) and deterministic greedy/stochastic sampling.
pub struct Inkling2Executor {
    info: ExecutorInfo,
    model: Arc<Inkling2CpuModel>,
    kv_precision: KvCachePrecision,
    states: HashMap<SequenceId, SequenceState>,
    rows: Vec<BoundRow>,
    readiness: Readiness,
    phase: Phase,
    next_submission: u64,
}

impl Inkling2Executor {
    /// Construct a bounded greedy executor sharing `model` across up to `max_sequences`
    /// concurrent sequences with context ceiling `context_tokens`.
    ///
    /// # Errors
    /// Rejects zero limits or `context_tokens` exceeding the model's context length or `u32`.
    pub fn new(
        model: Arc<Inkling2CpuModel>,
        max_sequences: usize,
        context_tokens: usize,
        kv_precision: KvCachePrecision,
    ) -> Result<Self, Inkling2Error> {
        if max_sequences == 0 {
            return Err(Inkling2Error::new(
                "executor",
                "max_sequences must be positive",
            ));
        }
        if context_tokens == 0 || context_tokens > model.config().context_length() {
            return Err(Inkling2Error::new(
                "executor",
                format!(
                    "context_tokens {context_tokens} must be within 1..={}",
                    model.config().context_length()
                ),
            ));
        }
        let context_u32 = u32::try_from(context_tokens)
            .map_err(|_| Inkling2Error::new("executor", "context_tokens exceeds u32"))?;
        let max_batch_tokens = context_u32.min(4096);
        Ok(Self {
            info: ExecutorInfo {
                name: "inkling2-cpu".into(),
                limits: GenerationLimits {
                    context_tokens: context_u32,
                    max_sequences,
                    max_batch_tokens,
                    max_decode_tokens: 1,
                },
            },
            model,
            kv_precision,
            states: HashMap::with_capacity(max_sequences),
            rows: Vec::with_capacity(max_sequences),
            readiness: Readiness::default(),
            phase: Phase::Idle,
            next_submission: 0,
        })
    }

    #[must_use]
    pub fn active_sequences(&self) -> usize {
        self.states.len()
    }

    fn bind_batch(&mut self, items: &[BatchItem]) -> Result<(), ExecutionError> {
        self.rows.clear();
        let limits = self.info.limits;
        let total_tokens = items
            .iter()
            .try_fold(0_u32, |sum, item| sum.checked_add(item.token_budget));
        if items.is_empty()
            || items.len() > limits.max_sequences
            || total_tokens.is_none_or(|n| n > limits.max_batch_tokens)
        {
            return Err(ExecutionError::new(
                "Inkling 2 batch exceeds prepared limits",
            ));
        }

        let bind_result: Result<(), ExecutionError> = (|| {
            for (index, item) in items.iter().enumerate() {
                if items[..index]
                    .iter()
                    .any(|prior| prior.sequence == item.sequence)
                {
                    return Err(ExecutionError::new(
                        "duplicate Inkling 2 sequence in offered batch",
                    ));
                }
                let seq = self
                    .states
                    .get(&item.sequence)
                    .ok_or_else(|| ExecutionError::new("offered sequence is not admitted"))?;
                if seq.cpu.position() != seq.logical.cursor as usize {
                    return Err(ExecutionError::new(
                        "CPU state position disagrees with logical cursor",
                    ));
                }
                let end = item
                    .prefix
                    .checked_add(item.token_budget)
                    .ok_or_else(|| ExecutionError::new("Inkling 2 prefix overflow"))?;
                if item.token_budget == 0
                    || item.prefix != seq.logical.cursor
                    || end > limits.context_tokens
                    || seq.logical.remaining == 0
                {
                    return Err(ExecutionError::new(
                        "work disagrees with continuation or capacity",
                    ));
                }
                let input = match item.kind {
                    StepKind::Prefill => {
                        if end as usize > seq.logical.prompt.len()
                            || item.output_budget
                                != u32::from(end as usize == seq.logical.prompt.len())
                        {
                            return Err(ExecutionError::new(
                                "prefill boundary or output_budget mismatch",
                            ));
                        }
                        RowInput::Prefill(item.prefix as usize..end as usize)
                    }
                    StepKind::Decode => {
                        if item.token_budget != 1
                            || item.output_budget != 1
                            || (item.prefix as usize) < seq.logical.prompt.len()
                        {
                            return Err(ExecutionError::new("invalid decode row"));
                        }
                        let token = seq.logical.input.ok_or_else(|| {
                            ExecutionError::new("decode row has no previous sampled token")
                        })?;
                        RowInput::Decode(token)
                    }
                };
                self.rows.push(BoundRow { work: *item, input });
            }
            Ok(())
        })();

        if bind_result.is_err() {
            self.rows.clear();
        }
        bind_result
    }
}

impl GenerationExecutor for Inkling2Executor {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn supports_sampling(&self) -> bool {
        true
    }

    fn admit(
        &mut self,
        request_id: RequestId,
        sequence: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        if self.phase != Phase::Idle {
            return Err(ExecutionError::new(
                "Inkling 2 executor is busy, faulted, or closed",
            ));
        }
        if self.states.contains_key(&sequence) {
            return Err(ExecutionError::new("sequence is already admitted"));
        }
        let sampling = request.options.sampling;
        if !sampling.temperature.is_finite() || sampling.temperature < 0.0 {
            return Ok(Admission::Rejected(ExecutionError::new(
                "sampling temperature must be finite and nonnegative",
            )));
        }
        if !sampling.top_p.is_finite() || sampling.top_p <= 0.0 || sampling.top_p > 1.0 {
            return Ok(Admission::Rejected(ExecutionError::new(
                "sampling top_p must be in (0, 1]",
            )));
        }
        let Ok(prompt_len) = u32::try_from(request.tokens.len()) else {
            return Ok(Admission::Rejected(ExecutionError::new(
                "prompt length exceeds u32",
            )));
        };
        let output_budget = request.options.max_output_tokens;
        if prompt_len == 0 || output_budget == 0 {
            return Ok(Admission::Rejected(ExecutionError::new(
                "prompt and max_output_tokens must be positive",
            )));
        }
        if u64::from(prompt_len) + u64::from(output_budget)
            > u64::from(self.info.limits.context_tokens)
        {
            return Ok(Admission::Rejected(ExecutionError::new(
                "prompt plus output exceeds prepared context capacity",
            )));
        }
        let vocab = self.model.config().vocab_unpadded();
        if request
            .tokens
            .iter()
            .any(|&tok| usize::try_from(tok).is_ok_and(|id| id >= vocab))
        {
            return Ok(Admission::Rejected(ExecutionError::new(
                "prompt contains token ID outside unpadded vocabulary",
            )));
        }

        // Register before testing transient slot capacity (zero lost-wakeup window).
        let wait = self.readiness.register();
        if self.states.len() >= self.info.limits.max_sequences {
            return Ok(Admission::Deferred(wait));
        }

        let cpu = self.model.new_state(self.kv_precision);
        self.states.insert(
            sequence,
            SequenceState {
                logical: LogicalSequence {
                    prompt: Arc::clone(&request.tokens),
                    cursor: 0,
                    input: None,
                    remaining: output_budget,
                },
                cpu,
                sampling,
                rng_state: init_rng_state(sampling.seed, request_id),
            },
        );
        Ok(Admission::Ready)
    }

    fn prepare(&mut self, batch: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
        if self.phase != Phase::Idle {
            return Err(ExecutionError::new(
                "Inkling 2 executor is busy, faulted, or closed",
            ));
        }
        self.bind_batch(batch)?;
        self.phase = Phase::Prepared;
        Ok(BatchPreparation::Ready)
    }

    fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
        match self.phase {
            Phase::Prepared => {
                self.rows.clear();
                self.phase = Phase::Idle;
                Ok(())
            }
            Phase::Idle | Phase::Closed => Ok(()),
            Phase::Submitted(_) | Phase::Failed => Err(ExecutionError::new(
                "cannot abandon submitted or failed Inkling 2 work",
            )),
        }
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        if self.phase != Phase::Prepared
            || !self
                .rows
                .iter()
                .map(|row| row.work)
                .eq(batch.iter().copied())
        {
            return Err(ExecutionError::new(
                "submission differs from prepared Inkling 2 batch",
            ));
        }
        self.next_submission = self.next_submission.wrapping_add(1);
        let id = SubmissionId::new(self.next_submission);
        self.phase = Phase::Submitted(id);
        Ok(id)
    }

    fn poll(
        &mut self,
        submission: SubmissionId,
    ) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        if self.phase != Phase::Submitted(submission) {
            return Err(ExecutionError::new("unknown Inkling 2 submission"));
        }
        // Defensive fault transition before mutating sequence states.
        self.phase = Phase::Failed;

        let mut completions = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let item = row.work;
            let state = self
                .states
                .get_mut(&item.sequence)
                .expect("bound sequence exists");
            let logits = match &row.input {
                RowInput::Prefill(range) => {
                    let slice = &state.logical.prompt[range.clone()];
                    self.model
                        .forward_tokens(&mut state.cpu, slice, item.output_budget > 0)
                }
                RowInput::Decode(tok) => self.model.forward_tokens(&mut state.cpu, &[*tok], true),
            }
            .map_err(|error| ExecutionError::new(error.to_string()))?;

            if state.cpu.position() != row.end() as usize {
                return Err(ExecutionError::new(
                    "Inkling 2 forward did not advance to expected prefix",
                ));
            }
            let tokens = match logits {
                Some(row_logits) => {
                    vec![sample_f32_logits(
                        &row_logits,
                        state.sampling,
                        &mut state.rng_state,
                    )?]
                }
                None => Vec::new(),
            };
            completions.push(StepCompletion {
                sequence: item.sequence,
                prefix: row.end(),
                tokens,
            });
        }

        // Commit logical state only after every row in the offer succeeded.
        for completion in &completions {
            let state = self
                .states
                .get_mut(&completion.sequence)
                .expect("validated sequence");
            state.logical.cursor = completion.prefix;
            if let Some(&tok) = completion.tokens.first() {
                state.logical.input = Some(tok);
                state.logical.remaining -= 1;
            }
        }

        self.rows.clear();
        self.phase = Phase::Idle;
        Ok(Some(completions))
    }

    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        if self.phase == Phase::Failed || self.rows.iter().any(|row| row.work.sequence == sequence)
        {
            return Err(ExecutionError::new(
                "cannot release prepared, submitted, or faulted sequence state",
            ));
        }
        if self.states.remove(&sequence).is_some() {
            self.readiness.publish();
        }
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.rows.clear();
        self.states.clear();
        self.phase = Phase::Closed;
        self.readiness.publish();
        Ok(())
    }
}
