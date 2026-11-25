//! Selection over engine-owned queues; no duplicate request or resource owner.
use super::Engine;
use crate::{BatchItem, PrefillScheduling, StepKind};

impl Engine {
    pub(super) fn build_batch(&mut self) {
        self.batch.clear();
        self.batch_slots.clear();
        let mut budget = self.policy.max_batch_tokens;
        let prefill_cap = self.policy.prefill_chunk_tokens;
        let scheduling = self.policy.prefill_scheduling;
        if let Some(survivor) = self.protected_survivor {
            if self.try_schedule_survivor_prefill(survivor, &mut budget, prefill_cap)
                && matches!(scheduling, PrefillScheduling::Separate { .. })
            {
                return;
            }
            self.promote_in_queue(StepKind::Decode, survivor);
        }
        let force_prefill = self.batch.is_empty()
            && !self.prefill.is_empty()
            && self.decode_only_steps >= self.policy.max_decode_only_steps;
        if force_prefill {
            self.schedule_kind(StepKind::Prefill, &mut budget, 1, prefill_cap);
            if !self.batch.is_empty() && matches!(scheduling, PrefillScheduling::Separate { .. }) {
                return;
            }
        }
        self.schedule_kind(StepKind::Decode, &mut budget, usize::MAX, prefill_cap);
        match scheduling {
            PrefillScheduling::Mixed => {
                self.schedule_kind(StepKind::Prefill, &mut budget, usize::MAX, prefill_cap);
            }
            PrefillScheduling::Separate { idle_chunk_tokens } if self.batch.is_empty() => {
                self.schedule_kind(StepKind::Prefill, &mut budget, 1, idle_chunk_tokens);
            }
            PrefillScheduling::Separate { .. } => {}
        }
    }

    fn promote_in_queue(&mut self, kind: StepKind, target: usize) {
        let queue = match kind {
            StepKind::Prefill => &mut self.prefill,
            StepKind::Decode => &mut self.decode,
        };
        if let Some(pos) = queue.iter().position(|&idx| idx == target)
            && pos != 0
        {
            queue.remove(pos);
            queue.push_front(target);
        }
    }

    fn try_schedule_survivor_prefill(
        &mut self,
        survivor: usize,
        budget: &mut u32,
        prefill_cap: u32,
    ) -> bool {
        if *budget == 0 || self.batch.len() >= self.config.max_active_requests {
            return false;
        }
        let Some(pos) = self.prefill.iter().position(|&idx| idx == survivor) else {
            return false;
        };
        self.prefill.remove(pos);
        if self.try_offer_slot(StepKind::Prefill, survivor, budget, prefill_cap) {
            true
        } else {
            self.prefill.push_back(survivor);
            false
        }
    }

    pub(super) fn record_scheduled_batch(&mut self) {
        if self.batch.iter().any(|item| item.kind == StepKind::Prefill) {
            self.decode_only_steps = 0;
        } else if !self.batch.is_empty() && !self.prefill.is_empty() {
            self.decode_only_steps = self.decode_only_steps.saturating_add(1);
        }
    }

    fn schedule_kind(&mut self, kind: StepKind, budget: &mut u32, limit: usize, prefill_cap: u32) {
        let queue_len = match kind {
            StepKind::Prefill => self.prefill.len(),
            StepKind::Decode => self.decode.len(),
        };
        let mut selected = 0;
        // Visit each ready row at most once. A full client mailbox does not
        // prevent later rows from using their own output credits.
        for _ in 0..queue_len {
            if selected >= limit
                || self.batch.len() >= self.config.max_active_requests
                || *budget == 0
            {
                break;
            }
            let index = match kind {
                StepKind::Prefill => self.prefill.pop_front(),
                StepKind::Decode => self.decode.pop_front(),
            }
            .expect("nonempty ready queue");
            if self.try_offer_slot(kind, index, budget, prefill_cap) {
                selected += 1;
            } else {
                match kind {
                    StepKind::Prefill => self.prefill.push_back(index),
                    StepKind::Decode => self.decode.push_back(index),
                }
            }
        }
    }

    fn try_offer_slot(
        &mut self,
        kind: StepKind,
        index: usize,
        budget: &mut u32,
        prefill_cap: u32,
    ) -> bool {
        let sequence = self.slots[index].as_mut().expect("runnable slot exists");
        let credits = self.output.credits(sequence.output);
        let (tokens, outputs) = match kind {
            StepKind::Prefill => {
                let tokens = (sequence.replay_tokens - sequence.prefix)
                    .min(prefill_cap)
                    .min(*budget);
                (
                    tokens,
                    u32::from(sequence.prefix + tokens == sequence.replay_tokens),
                )
            }
            StepKind::Decode => {
                let output_credits = u32::try_from(credits.saturating_sub(1)).unwrap_or(u32::MAX);
                let tokens = self
                    .policy
                    .decode_tokens
                    .min(sequence.options.max_output_tokens - sequence.generated)
                    .min(*budget)
                    .min(output_credits);
                (tokens, tokens)
            }
        };
        if tokens == 0 || u64::from(outputs) + 1 > credits as u64 {
            return false;
        }
        self.output.reserve(sequence.output, outputs as usize + 1);
        self.batch.push(BatchItem {
            sequence: sequence.id,
            kind,
            prefix: sequence.prefix,
            token_budget: tokens,
            output_budget: outputs,
        });
        self.batch_slots.push(index);
        *budget -= tokens;
        true
    }

    pub(super) fn release_output_reservations(&mut self) {
        for &index in &self.batch_slots {
            let sequence = self.slots[index].as_ref().expect("submitted slot");
            self.output.unreserve(sequence.output);
        }
    }
}
