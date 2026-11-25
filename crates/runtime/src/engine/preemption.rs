//! Completion-safe recomputation preemption under all-active-growth or consumer-stall pressure.
//!
//! Preemption runs only when `self.pending.is_none()` (no in-flight submission),
//! ensuring every active sequence is idle before `model.release()` reclaims its
//! physical continuation. Replayed sequences retain their bounded `history`
//! (`prompt + committed generated tokens`) separately from `prompt_tokens` and
//! `generated`, so replay never duplicates emitted tokens or usage counts.

use super::{Engine, WorkState, next_id};
use crate::{EngineError, SequenceId};

impl Engine {
    /// Attempt one completion-safe preemption when no batch could be prepared or
    /// scheduled, returning `Ok(true)` if a sequence was preempted and physical
    /// capacity was reclaimed.
    pub(super) fn try_preempt_for_progress(&mut self) -> Result<bool, EngineError> {
        if !self.capabilities.preemption || self.pending.is_some() || self.active == 0 {
            return Ok(false);
        }

        // Case 1: Stalled-consumer preemption. If an active sequence cannot run
        // because its output mailbox has exhausted credits (< 2) while another
        // output-ready peer (parked, preempted, or waiting) needs capacity,
        // completion-safely preempt the stalled sequence so healthy peers finish.
        if self.has_output_ready_blocked_peer()
            && let Some(stalled) = self.select_stalled_active_victim()
        {
            self.preempt_slot(stalled)?;
            self.reactivate_parked();
            return Ok(true);
        }

        // Case 2: All-active-growth deadlock. Every runnable sequence was parked
        // during preparation (`prefill` and `decode` are empty, `parked` is nonempty).
        // Protect the oldest output-ready survivor and preempt the newest peer.
        // Never self-preempt when only one active sequence remains.
        if self.prefill.is_empty()
            && self.decode.is_empty()
            && !self.parked.is_empty()
            && self.active > 1
        {
            let survivor = self.select_or_keep_survivor();
            if let Some(victim) = self.select_growth_victim(survivor) {
                self.protected_survivor = survivor;
                self.preempt_slot(victim)?;
                // Releasing the victim freed physical blocks; immediately retry
                // parked peers on the next loop iteration of `step()`.
                self.reactivate_parked();
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// True if there is at least one parked, preempted, or fresh-waiting peer whose
    /// output mailbox has enough credits (`>= 2`) to make forward progress.
    fn has_output_ready_blocked_peer(&self) -> bool {
        let ready_in = |queue: &std::collections::VecDeque<usize>| {
            queue.iter().any(|&idx| {
                self.slots[idx]
                    .as_ref()
                    .is_some_and(|seq| self.output.credits(seq.output) >= 2)
            })
        };
        ready_in(&self.parked) || ready_in(&self.preempted) || ready_in(&self.waiting)
    }

    /// Find an active idle sequence whose consumer has stalled (`credits < 2`),
    /// preferring the protected survivor if it stalled so protection is relinquished.
    fn select_stalled_active_victim(&self) -> Option<usize> {
        let is_stalled = |idx: usize| {
            self.slots[idx].as_ref().is_some_and(|seq| {
                seq.admitted
                    && seq.work == WorkState::Idle
                    && seq.terminal.is_none()
                    && self.output.credits(seq.output) < 2
            })
        };
        if let Some(survivor) = self.protected_survivor
            && is_stalled(survivor)
        {
            return Some(survivor);
        }
        // Prefer newest stalled active sequence (highest RequestId).
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| {
                let seq = slot.as_ref()?;
                if is_stalled(idx) {
                    Some((seq.request.get(), idx))
                } else {
                    None
                }
            })
            .max_by_key(|&(req_id, _)| req_id)
            .map(|(_, idx)| idx)
    }

    /// Keep the existing protected survivor if it is still active and output-ready;
    /// otherwise protect the oldest active output-ready sequence (lowest `RequestId`).
    fn select_or_keep_survivor(&self) -> Option<usize> {
        let is_healthy_active = |idx: usize| {
            self.slots[idx].as_ref().is_some_and(|seq| {
                seq.admitted
                    && seq.work == WorkState::Idle
                    && seq.terminal.is_none()
                    && self.output.credits(seq.output) >= 2
            })
        };
        if let Some(survivor) = self.protected_survivor
            && is_healthy_active(survivor)
        {
            return Some(survivor);
        }
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| {
                let seq = slot.as_ref()?;
                if is_healthy_active(idx) {
                    Some((seq.request.get(), idx))
                } else {
                    None
                }
            })
            .min_by_key(|&(req_id, _)| req_id)
            .map(|(_, idx)| idx)
    }

    /// Select a victim distinct from `survivor`: prefer any consumer-stalled peer
    /// first, otherwise choose the newest active peer (LIFO, minimizing recompute).
    fn select_growth_victim(&self, survivor: Option<usize>) -> Option<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| {
                if Some(idx) == survivor {
                    return None;
                }
                let seq = slot.as_ref()?;
                if !seq.admitted || seq.work != WorkState::Idle || seq.terminal.is_some() {
                    return None;
                }
                let stalled = u8::from(self.output.credits(seq.output) < 2);
                Some(((stalled, seq.request.get()), idx))
            })
            .max_by_key(|&(key, _)| key)
            .map(|(_, idx)| idx)
    }

    /// Completion-safely retire `index`'s physical continuation and queue it for
    /// priority replay in `self.preempted`. Failure to release faults the engine
    /// and retains ownership rather than dropping physical charges early.
    fn preempt_slot(&mut self, index: usize) -> Result<(), EngineError> {
        debug_assert!(self.pending.is_none(), "preempt_slot requires idle device");
        self.prefill.retain(|&slot| slot != index);
        self.decode.retain(|&slot| slot != index);
        self.parked.retain(|&slot| slot != index);
        if self.protected_survivor == Some(index) {
            self.protected_survivor = None;
        }
        if let Err(error) = self.release(index) {
            let exec_err = match error {
                EngineError::Execution(e) | EngineError::Faulted(e) => e,
                other => crate::ExecutionError::new(other.to_string()),
            };
            self.fault_all(&exec_err);
            let _ = self.flush_terminals();
            return Err(EngineError::Faulted(exec_err));
        }
        let new_id = SequenceId(next_id()?);
        let sequence = self.slots[index].as_mut().expect("preempted slot exists");
        sequence.id = new_id;
        sequence.replay_tokens = u32::try_from(sequence.history.len()).expect("bounded history");
        sequence.prefix = 0;
        sequence.resource_wait = None;
        sequence.work = WorkState::Idle;
        self.preemptions = self.preemptions.saturating_add(1);
        // Preserve oldest-first readmission order among preempted victims so
        // newer displacements cannot starve earlier victims.
        let req_id = sequence.request.get();
        let pos = self
            .preempted
            .iter()
            .position(|&slot| {
                self.slots[slot]
                    .as_ref()
                    .is_some_and(|other| other.request.get() > req_id)
            })
            .unwrap_or(self.preempted.len());
        self.preempted.insert(pos, index);
        Ok(())
    }
}
