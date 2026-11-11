//! Model-local logical generation state and accepted input binding.
//!
//! This module knows no decoder, device allocation or execution lane. Binding an
//! entire offer changes neither continuation nor physical state.
use crate::Gemma4Config;
use ribn::{BatchItem, ExecutionError, GenerationLimits, SequenceId, StepKind, TokenRequest};
use std::{ops::Range, sync::Arc};

pub(crate) struct LogicalSequence {
    pub(crate) prompt: Arc<[u32]>,
    pub(crate) cursor: u32,
    pub(crate) input: Option<u32>,
    pub(crate) remaining: u32,
}

pub(crate) enum RowInput {
    Prefill(Range<usize>),
    Decode(u32),
}

pub(crate) struct BoundRow {
    work: BatchItem,
    input: RowInput,
}

impl BoundRow {
    pub(crate) fn work(&self) -> BatchItem {
        self.work
    }

    pub(crate) fn input(&self) -> &RowInput {
        &self.input
    }

    pub(crate) fn end(&self) -> u32 {
        // Only whole-offer binding constructs a row, after checked range validation.
        self.work.prefix + self.work.token_budget
    }
}

impl LogicalSequence {
    pub(crate) fn new(
        config: &Gemma4Config,
        context_tokens: u32,
        request: &TokenRequest,
    ) -> Result<Self, ExecutionError> {
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
        if u64::from(prompt) + u64::from(output) > u64::from(context_tokens) {
            return Err(ExecutionError::new(
                "prompt and output exceed the context bound",
            ));
        }
        config
            .validate_text_tokens(&request.tokens)
            .map_err(|error| ExecutionError::new(error.to_string()))?;
        Ok(Self {
            prompt: Arc::clone(&request.tokens),
            cursor: 0,
            input: None,
            remaining: output,
        })
    }

    fn bind(&self, item: BatchItem, context_tokens: u32) -> Result<BoundRow, ExecutionError> {
        let end = item
            .prefix
            .checked_add(item.token_budget)
            .ok_or_else(|| ExecutionError::new("Gemma prefix overflow"))?;
        if item.token_budget == 0
            || item.prefix != self.cursor
            || end > context_tokens
            || self.remaining == 0
        {
            return Err(ExecutionError::new(
                "work disagrees with continuation or capacity",
            ));
        }
        let input = match item.kind {
            StepKind::Prefill => {
                if end as usize > self.prompt.len()
                    || item.output_budget != u32::from(end as usize == self.prompt.len())
                {
                    return Err(ExecutionError::new("prefill boundary/output mismatch"));
                }
                RowInput::Prefill(item.prefix as usize..end as usize)
            }
            StepKind::Decode => {
                if item.token_budget != 1
                    || item.output_budget != 1
                    || (item.prefix as usize) < self.prompt.len()
                {
                    return Err(ExecutionError::new("invalid ordinary decode row"));
                }
                RowInput::Decode(
                    self.input
                        .ok_or_else(|| ExecutionError::new("ordinary decode has no input"))?,
                )
            }
        };
        Ok(BoundRow { work: item, input })
    }

    pub(crate) fn commit(&mut self, prefix: u32, token: Option<u32>) {
        self.cursor = prefix;
        if let Some(token) = token {
            self.input = Some(token);
            self.remaining -= 1;
        }
    }
}

/// Reuse bounded prepared storage, preserving offer order. Failure publishes no
/// bound rows, including when a later row is malformed; no cursor advances here.
pub(crate) fn bind_batch<'a>(
    items: &[BatchItem],
    limits: GenerationLimits,
    lookup: impl Fn(SequenceId) -> Option<&'a LogicalSequence>,
    rows: &mut Vec<BoundRow>,
) -> Result<(), ExecutionError> {
    rows.clear();
    let tokens = items
        .iter()
        .try_fold(0_u32, |sum, item| sum.checked_add(item.token_budget));
    if items.is_empty()
        || items.len() > limits.max_sequences
        || tokens.is_none_or(|n| n > limits.max_batch_tokens)
    {
        return Err(ExecutionError::new("Gemma batch exceeds prepared limits"));
    }
    let result = (|| {
        for (index, item) in items.iter().enumerate() {
            if items[..index]
                .iter()
                .any(|old| old.sequence == item.sequence)
            {
                return Err(ExecutionError::new("duplicate Gemma sequence in batch"));
            }
            let sequence = lookup(item.sequence)
                .ok_or_else(|| ExecutionError::new("submitted sequence is not admitted"))?;
            rows.push(sequence.bind(*item, limits.context_tokens)?);
        }
        Ok(())
    })();
    if result.is_err() {
        rows.clear();
    }
    result
}
