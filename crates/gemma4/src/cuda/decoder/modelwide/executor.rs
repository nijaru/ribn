//! Depth-one AR binding over the model-wide physical CUDA owner.
use super::{Gemma4CudaModelWide, MAX_SEQUENCES};
use crate::{
    Gemma4Error, argmax,
    generation::{BoundRow, LogicalSequence, RowInput, bind_batch},
};
use ribn::{
    Admission, BatchItem, BatchPreparation, ExecutionError, ExecutorInfo, GenerationExecutor,
    GenerationLimits, RequestId, SequenceId, StepCompletion, SubmissionId, TokenRequest,
};
use std::{collections::HashMap, mem::ManuallyDrop};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Prepared,
    Submitted(SubmissionId),
    Failed,
    Closed,
}
struct Execution {
    model: Gemma4CudaModelWide,
    states: HashMap<SequenceId, LogicalSequence>,
    rows: Vec<BoundRow>,
    results: Vec<Option<StepCompletion>>,
    ready: Vec<StepCompletion>,
    active: Option<[usize; 2]>,
    phase: Phase,
    next: u64,
}

/// Experimental model-wide CUDA implementation of the ordinary AR contract.
/// This owns one physical model, not a factory of per-request decoders. The
/// ordinary text loader retains its qualified serial backend until promotion.
pub struct Gemma4CudaExecutor {
    info: ExecutorInfo,
    execution: Option<ManuallyDrop<Box<Execution>>>,
}
impl Gemma4CudaExecutor {
    /// Bind a prepared physical model to bounded logical generation state.
    /// # Errors
    /// Reports metadata allocation/limit errors; physical resources remain owned
    /// through cleanup by the supplied model even if construction fails.
    pub fn new(model: Gemma4CudaModelWide) -> Result<Self, Gemma4Error> {
        model.owner().idle()?;
        if !model.owner().states.is_empty() {
            return Err(Gemma4Error::new(
                "CUDA executor",
                "physical continuations already exist; this is not a restoration API",
            ));
        }
        let context_tokens = u32::try_from(model.capacity())
            .map_err(|_| Gemma4Error::new("CUDA executor", "context exceeds u32"))?;
        let mut states = HashMap::new();
        let mut rows = Vec::new();
        let mut results = Vec::new();
        let mut ready = Vec::new();
        let metadata = |error: std::collections::TryReserveError| {
            Gemma4Error::new("CUDA executor metadata", error.to_string())
        };
        states.try_reserve(MAX_SEQUENCES).map_err(metadata)?;
        rows.try_reserve_exact(MAX_SEQUENCES).map_err(metadata)?;
        results.try_reserve_exact(MAX_SEQUENCES).map_err(metadata)?;
        ready.try_reserve_exact(MAX_SEQUENCES).map_err(metadata)?;
        Ok(Self {
            info: ExecutorInfo {
                name: "Gemma model-wide CUDA".into(),
                limits: GenerationLimits {
                    context_tokens,
                    max_sequences: MAX_SEQUENCES,
                    max_batch_tokens: 4096,
                    max_decode_tokens: 1,
                },
            },
            execution: Some(ManuallyDrop::new(Box::new(Execution {
                model,
                states,
                rows,
                results,
                ready,
                active: None,
                phase: Phase::Idle,
                next: 0,
            }))),
        })
    }
}
impl Execution {
    fn completion(
        &self,
        index: usize,
        logits: Option<Vec<u16>>,
    ) -> Result<StepCompletion, ExecutionError> {
        let row = &self.rows[index];
        let work = row.work();
        if self.model.position(work.sequence) != Some(row.end() as usize) {
            return Err(ExecutionError::new("CUDA physical/logical end mismatch"));
        }
        let tokens = if let Some(logits) = logits {
            if work.output_budget != 1
                || logits.len() != self.model.config().vocab_size()
                || logits.iter().any(|bits| bits & 0x7f80 == 0x7f80)
            {
                return Err(ExecutionError::new(
                    "invalid CUDA output shape, finiteness or budget",
                ));
            }
            vec![argmax(&logits).ok_or_else(|| ExecutionError::new("CUDA logits have no winner"))?]
        } else {
            if work.output_budget != 0 {
                return Err(ExecutionError::new("CUDA omitted a required output"));
            }
            Vec::new()
        };
        Ok(StepCompletion {
            sequence: work.sequence,
            prefix: row.end(),
            tokens,
        })
    }

    // This is physical lane selection within an already accepted offer, not a
    // second admission scheduler. Results retain their original offer slots.
    fn advance(&mut self) -> Result<(), ExecutionError> {
        while let Some(first) = self.results.iter().position(Option::is_none) {
            if let RowInput::Decode(a) = self.rows[first].input()
                && let Some(second) = (first + 1..self.rows.len()).find(|&index| {
                    self.results[index].is_none()
                        && matches!(self.rows[index].input(), RowInput::Decode(_))
                })
            {
                let RowInput::Decode(b) = self.rows[second].input() else {
                    unreachable!("selected decode lane")
                };
                self.model
                    .start_pair(
                        [
                            self.rows[first].work().sequence,
                            self.rows[second].work().sequence,
                        ],
                        [*a, *b],
                    )
                    .map_err(|error| execution(&error))?;
                self.active = Some([first, second]);
                return Ok(());
            }
            let row = &self.rows[first];
            let work = row.work();
            let input = match row.input() {
                RowInput::Prefill(range) => {
                    &self
                        .states
                        .get(&work.sequence)
                        .expect("bound sequence")
                        .prompt[range.clone()]
                }
                RowInput::Decode(token) => std::slice::from_ref(token),
            };
            let logits = self
                .model
                .prefill(work.sequence, input, work.output_budget != 0)
                .map_err(|error| execution(&error))?;
            self.results[first] = Some(self.completion(first, logits)?);
        }
        Ok(())
    }
    fn close(&mut self) -> Result<(), ExecutionError> {
        if self.phase == Phase::Closed {
            return Ok(());
        }
        self.phase = Phase::Failed;
        self.model
            .close()
            .map_err(|error| ExecutionError::new(error.to_string()))?;
        self.states.clear();
        self.rows.clear();
        self.results.clear();
        self.ready.clear();
        self.active = None;
        self.phase = Phase::Closed;
        Ok(())
    }
}
fn execution(error: &Gemma4Error) -> ExecutionError {
    ExecutionError::new(error.to_string())
}
impl GenerationExecutor for Gemma4CudaExecutor {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }
    fn admit(
        &mut self,
        _: RequestId,
        sequence: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        let owner: &mut Execution = self.execution.as_mut().expect("live CUDA execution");
        if owner.phase != Phase::Idle {
            return Err(ExecutionError::new(
                "CUDA execution is busy, faulted or closed",
            ));
        }
        if owner.states.contains_key(&sequence) {
            owner.phase = Phase::Failed;
            return Err(ExecutionError::new("duplicate CUDA sequence incarnation"));
        }
        if owner.states.len() >= MAX_SEQUENCES {
            return Ok(Admission::Rejected(ExecutionError::new(
                "CUDA sequence capacity exhausted",
            )));
        }
        let logical = match LogicalSequence::new(
            owner.model.config(),
            self.info.limits.context_tokens,
            request,
        ) {
            Ok(logical) => logical,
            Err(error) => return Ok(Admission::Rejected(error)),
        };
        let reach = logical.prompt.len() + logical.remaining as usize;
        owner.phase = Phase::Failed;
        let admission = owner
            .model
            .admit(sequence, reach)
            .map_err(|error| execution(&error))?;
        if matches!(admission, Admission::Ready) {
            owner.states.insert(sequence, logical);
        }
        owner.phase = Phase::Idle;
        Ok(admission)
    }
    fn prepare(&mut self, batch: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
        let owner: &mut Execution = self.execution.as_mut().expect("live CUDA execution");
        if owner.phase != Phase::Idle {
            return Err(ExecutionError::new(
                "CUDA execution is busy, faulted or closed",
            ));
        }
        owner.phase = Phase::Failed;
        bind_batch(
            batch,
            self.info.limits,
            |id| owner.states.get(&id),
            &mut owner.rows,
        )?;
        for row in &owner.rows {
            if owner.model.position(row.work().sequence) != Some(row.work().prefix as usize) {
                return Err(ExecutionError::new("CUDA physical/logical prefix mismatch"));
            }
        }
        owner
            .ready
            .try_reserve_exact(batch.len())
            .map_err(|error| ExecutionError::new(error.to_string()))?;
        owner.results.resize_with(batch.len(), || None);
        owner.phase = Phase::Prepared;
        Ok(BatchPreparation::Ready)
    }
    fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
        let owner: &mut Execution = self.execution.as_mut().expect("live CUDA execution");
        match owner.phase {
            Phase::Prepared => {
                owner.rows.clear();
                owner.results.clear();
                owner.phase = Phase::Idle;
            }
            Phase::Idle | Phase::Closed => {}
            Phase::Submitted(_) | Phase::Failed => {
                return Err(ExecutionError::new(
                    "cannot abandon submitted or failed CUDA work",
                ));
            }
        }
        Ok(())
    }
    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        let owner: &mut Execution = self.execution.as_mut().expect("live CUDA execution");
        if owner.phase != Phase::Prepared
            || !owner
                .rows
                .iter()
                .map(BoundRow::work)
                .eq(batch.iter().copied())
        {
            owner.phase = Phase::Failed;
            return Err(ExecutionError::new(
                "CUDA submission differs from preparation",
            ));
        }
        owner.phase = Phase::Failed;
        owner.next = owner.next.wrapping_add(1);
        let id = SubmissionId::new(owner.next);
        owner.advance()?;
        owner.phase = Phase::Submitted(id);
        Ok(id)
    }
    fn poll(
        &mut self,
        submission: SubmissionId,
    ) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        let owner: &mut Execution = self.execution.as_mut().expect("live CUDA execution");
        if owner.phase != Phase::Submitted(submission) {
            return Err(ExecutionError::new("unknown CUDA submission"));
        }
        owner.phase = Phase::Failed;
        if let Some(indices) = owner.active {
            let Some(logits) = owner.model.poll_pair().map_err(|error| execution(&error))? else {
                owner.phase = Phase::Submitted(submission);
                return Ok(None);
            };
            for (index, logits) in indices.into_iter().zip(logits) {
                owner.results[index] = Some(owner.completion(index, Some(logits))?);
            }
            owner.active = None;
        }
        owner.advance()?;
        if owner.active.is_some() {
            owner.phase = Phase::Submitted(submission);
            return Ok(None);
        }
        // Every physical row has been validated, including later fallback rows.
        // No cursor, next input or output credit changes before this boundary.
        for result in &mut owner.results {
            owner
                .ready
                .push(result.take().expect("complete accepted offer"));
        }
        for row in &owner.ready {
            owner
                .states
                .get_mut(&row.sequence)
                .expect("bound sequence")
                .commit(row.prefix, row.tokens.first().copied());
        }
        owner.rows.clear();
        owner.results.clear();
        owner.phase = Phase::Idle;
        Ok(Some(std::mem::take(&mut owner.ready)))
    }
    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        let owner: &mut Execution = self.execution.as_mut().expect("live CUDA execution");
        if owner.phase == Phase::Closed {
            return Ok(());
        }
        if owner.phase == Phase::Failed
            || owner.rows.iter().any(|row| row.work().sequence == sequence)
        {
            return Err(ExecutionError::new(
                "cannot release prepared, submitted or failed CUDA state",
            ));
        }
        let previous = owner.phase;
        owner.phase = Phase::Failed;
        owner
            .model
            .release(sequence)
            .map_err(|error| execution(&error))?;
        owner.states.remove(&sequence);
        owner.phase = previous;
        Ok(())
    }
    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.execution
            .as_mut()
            .expect("live CUDA execution")
            .close()
    }
}
impl Drop for Gemma4CudaExecutor {
    fn drop(&mut self) {
        if let Some(mut owner) = self.execution.take()
            && owner.close().is_ok()
        {
            drop(ManuallyDrop::into_inner(owner));
        }
    }
}
