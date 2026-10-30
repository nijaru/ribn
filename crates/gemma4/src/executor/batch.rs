//! Whole-offer validation and execution over a bounded, owned Gemma batch.
use super::{Execution, GemmaDecode, execution};
use crate::argmax;
use ribn::{BatchItem, ExecutionError, GenerationLimits, StepCompletion, StepKind};

impl<D: GemmaDecode> Execution<D> {
    pub(super) fn validate(
        &self,
        items: &[BatchItem],
        limits: GenerationLimits,
    ) -> Result<(), ExecutionError> {
        let tokens = items
            .iter()
            .try_fold(0_u32, |sum, item| sum.checked_add(item.token_budget));
        if items.is_empty()
            || items.len() > limits.max_sequences
            || tokens.is_none_or(|n| n > limits.max_batch_tokens)
        {
            return Err(ExecutionError::new("Gemma batch exceeds prepared limits"));
        }
        for (index, item) in items.iter().enumerate() {
            if items[..index]
                .iter()
                .any(|old| old.sequence == item.sequence)
            {
                return Err(ExecutionError::new("duplicate Gemma sequence in batch"));
            }
            let state = self
                .states
                .get(&item.sequence)
                .ok_or_else(|| ExecutionError::new("submitted sequence is not admitted"))?;
            let end = item
                .prefix
                .checked_add(item.token_budget)
                .ok_or_else(|| ExecutionError::new("Gemma prefix overflow"))?;
            if item.token_budget == 0
                || item.prefix != state.cursor
                || state.decoder.position() != state.cursor as usize
                || end > limits.context_tokens
                || state.remaining == 0
            {
                return Err(ExecutionError::new(
                    "work disagrees with continuation or capacity",
                ));
            }
            match item.kind {
                StepKind::Prefill => {
                    if end as usize > state.prompt.len()
                        || item.output_budget != u32::from(end as usize == state.prompt.len())
                    {
                        return Err(ExecutionError::new("prefill boundary/output mismatch"));
                    }
                }
                StepKind::Decode => {
                    if item.token_budget != 1
                        || item.output_budget != 1
                        || (item.prefix as usize) < state.prompt.len()
                        || state.input.is_none()
                    {
                        return Err(ExecutionError::new("invalid ordinary decode row"));
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn execute(&mut self) -> Result<Vec<StepCompletion>, ExecutionError> {
        let mut rows = Vec::with_capacity(self.items.len());
        for item in &self.items {
            let state = self
                .states
                .get_mut(&item.sequence)
                .expect("validated sequence");
            let prefix = item.prefix + item.token_budget;
            let logits = match item.kind {
                StepKind::Prefill => {
                    let tokens = &state.prompt[item.prefix as usize..prefix as usize];
                    if item.output_budget == 0 {
                        state.decoder.advance_prefill(tokens).map(|()| None)
                    } else {
                        state.decoder.prefill(tokens).map(Some)
                    }
                }
                StepKind::Decode => state
                    .decoder
                    .decode(state.input.expect("validated decode input"))
                    .map(Some),
            }
            .map_err(|error| execution(&error))?;
            if state.decoder.position() != prefix as usize {
                return Err(ExecutionError::new(
                    "decoder did not complete the accepted range",
                ));
            }
            let tokens = match logits {
                Some(logits) => {
                    vec![argmax(&logits).ok_or_else(|| ExecutionError::new("logits are empty"))?]
                }
                None => Vec::new(),
            };
            rows.push(StepCompletion {
                sequence: item.sequence,
                prefix,
                tokens,
            });
        }
        // No logical cursor/input/output-budget mutation until every row succeeds.
        Ok(rows)
    }
}
