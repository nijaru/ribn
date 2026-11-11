//! Serial reference execution over model-local, whole-offer bound rows.
use super::{Execution, GemmaDecode, execution};
use crate::{
    argmax,
    generation::{RowInput, bind_batch},
};
use ribn::{BatchItem, ExecutionError, GenerationLimits, StepCompletion};

impl<D: GemmaDecode> Execution<D> {
    pub(super) fn bind(
        &mut self,
        items: &[BatchItem],
        limits: GenerationLimits,
    ) -> Result<(), ExecutionError> {
        bind_batch(
            items,
            limits,
            |sequence| self.states.get(&sequence).map(|state| &state.logical),
            &mut self.rows,
        )?;
        // Logical binding is independent of physical storage. The serial oracle
        // additionally proves each decoder agrees before any row can execute.
        for row in &self.rows {
            let state = self
                .states
                .get(&row.work().sequence)
                .expect("bound sequence");
            if state.decoder.position() != state.logical.cursor as usize {
                self.rows.clear();
                return Err(ExecutionError::new(
                    "decoder disagrees with logical continuation",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn execute(&mut self) -> Result<Vec<StepCompletion>, ExecutionError> {
        let mut rows = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let item = row.work();
            let state = self.states.get_mut(&item.sequence).expect("bound sequence");
            let logits = match row.input() {
                RowInput::Prefill(range) => {
                    let tokens = &state.logical.prompt[range.clone()];
                    if item.output_budget == 0 {
                        state.decoder.advance_prefill(tokens).map(|()| None)
                    } else {
                        state.decoder.prefill(tokens).map(Some)
                    }
                }
                RowInput::Decode(token) => state.decoder.decode(*token).map(Some),
            }
            .map_err(|error| execution(&error))?;
            if state.decoder.position() != row.end() as usize {
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
                prefix: row.end(),
                tokens,
            });
        }
        // No logical cursor/input/output-budget mutation until every row succeeds.
        Ok(rows)
    }
}
