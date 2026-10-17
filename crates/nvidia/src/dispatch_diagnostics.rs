//! Opt-in, bounded observation of the serving dispatcher's enqueue decisions.
use std::sync::{Arc, Mutex};

use engine_core::ExecutionBatch;

use crate::quantized::MAX_BATCH_MEMBERS;

/// First failed whole-batch decode condition, in row-count then scheduler order.
/// A row-wise fallback may still use same-sequence prefill chunks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum DecodeFallback {
    SingleRow,
    RowLimit,
    NonDecodePhase,
    DecodeShape,
    MissingPosition,
    IncompatibleKv,
}

impl DecodeFallback {
    pub const ALL: [Self; 6] = [
        Self::SingleRow,
        Self::RowLimit,
        Self::NonDecodePhase,
        Self::DecodeShape,
        Self::MissingPosition,
        Self::IncompatibleKv,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SingleRow => "single_row",
            Self::RowLimit => "row_limit",
            Self::NonDecodePhase => "non_decode_phase",
            Self::DecodeShape => "decode_shape",
            Self::MissingPosition => "missing_position",
            Self::IncompatibleKv => "incompatible_kv",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DispatchRoute {
    WholeDecode,
    Grouped,
    RowWise(DecodeFallback),
}

/// Fully enqueued model-step work, not kernel launches, output tokens or device
/// completion. Attention inside a decode lane may still visit members serially.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DispatchWork {
    pub prefill_rows: u64,
    pub prefill_tokens: u64,
    pub prefill_chunk_calls: u64,
    pub prefill_chunk_tokens: u64,
    /// Index is decode lane width; index 1 is scalar decode, index 0 is unused.
    pub decode_lane_calls: [u64; MAX_BATCH_MEMBERS + 1],
}

impl DispatchWork {
    pub(crate) fn prefill(tokens: usize, chunked: usize, members: Option<usize>) -> Self {
        Self {
            prefill_rows: 1,
            prefill_tokens: count(tokens),
            prefill_chunk_calls: members.map_or(0, |members| count(chunked / members)),
            prefill_chunk_tokens: count(chunked),
            ..Self::default()
        }
    }

    pub(crate) fn decode(members: usize) -> Self {
        let mut work = Self::default();
        work.decode_lane_calls[members] = 1;
        work
    }

    pub(crate) fn merge(&mut self, other: Self) {
        self.prefill_rows = self.prefill_rows.saturating_add(other.prefill_rows);
        self.prefill_tokens = self.prefill_tokens.saturating_add(other.prefill_tokens);
        self.prefill_chunk_calls = self
            .prefill_chunk_calls
            .saturating_add(other.prefill_chunk_calls);
        self.prefill_chunk_tokens = self
            .prefill_chunk_tokens
            .saturating_add(other.prefill_chunk_tokens);
        for (total, added) in self
            .decode_lane_calls
            .iter_mut()
            .zip(other.decode_lane_calls)
        {
            *total = total.saturating_add(added);
        }
    }
}

/// Lifetime totals since observation was enabled. Counters saturate independently.
/// Work/route totals include only submissions whose entire enqueue and completion
/// event recording succeeded. Failures may have launched uncounted partial work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DispatchSnapshot {
    /// Attempts after health, state-count and submission-identity checks.
    pub attempts: u64,
    pub offered_rows: u64,
    pub offered_tokens: u64,
    pub enqueued_submissions: u64,
    pub enqueue_failures: u64,
    pub whole_decode_submissions: u64,
    /// Experimental grouped policy was selected; actual groups are in `work`.
    pub grouped_policy_submissions: u64,
    /// Successful row-wise submissions, indexed by `DecodeFallback`.
    pub row_wise_fallbacks: [u64; DecodeFallback::ALL.len()],
    pub work: DispatchWork,
    /// Routing, enqueue and event-recording host wall time, including failed
    /// cleanup; excludes initial validation and diagnostic publication. Not
    /// device time, queue latency or summed per-row metrics.
    pub host_submit_nanos: u64,
}

/// Read-only observation; contains no executor, device allocations or callbacks.
/// One dispatcher publishes once per observed submit attempt. Snapshots are coherent
/// fixed-size copies. Enabling observation allocates once; publication allocates nothing.
#[derive(Clone, Debug)]
pub struct DispatchObserver(Arc<Mutex<DispatchSnapshot>>);

impl DispatchObserver {
    pub(crate) fn new() -> Self {
        Self(Arc::new(Mutex::new(DispatchSnapshot::default())))
    }

    #[must_use]
    pub fn snapshot(&self) -> DispatchSnapshot {
        *self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn record(
        &self,
        batch: &ExecutionBatch,
        route: DispatchRoute,
        enqueued: Option<DispatchWork>,
        host_submit_nanos: u64,
    ) {
        let offered_tokens = batch.segments().iter().fold(0_u64, |total, segment| {
            total.saturating_add(u64::from(segment.token_count()))
        });
        let mut snapshot = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        snapshot.attempts = snapshot.attempts.saturating_add(1);
        snapshot.offered_rows = snapshot.offered_rows.saturating_add(count(batch.len()));
        snapshot.offered_tokens = snapshot.offered_tokens.saturating_add(offered_tokens);
        snapshot.host_submit_nanos = snapshot.host_submit_nanos.saturating_add(host_submit_nanos);
        if let Some(work) = enqueued {
            snapshot.enqueued_submissions = snapshot.enqueued_submissions.saturating_add(1);
            snapshot.work.merge(work);
            match route {
                DispatchRoute::WholeDecode => {
                    snapshot.whole_decode_submissions =
                        snapshot.whole_decode_submissions.saturating_add(1);
                }
                DispatchRoute::Grouped => {
                    snapshot.grouped_policy_submissions =
                        snapshot.grouped_policy_submissions.saturating_add(1);
                }
                DispatchRoute::RowWise(reason) => {
                    let total = &mut snapshot.row_wise_fallbacks[reason as usize];
                    *total = total.saturating_add(1);
                }
            }
        } else {
            snapshot.enqueue_failures = snapshot.enqueue_failures.saturating_add(1);
        }
    }
}

fn count(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{ExecutionPhase, ExecutionSegment, RequestId};

    #[test]
    fn snapshots_separate_offers_successful_lanes_and_failed_partial_enqueue() {
        let batch = ExecutionBatch::new(
            (1..=3)
                .map(|id| {
                    ExecutionSegment::new(
                        RequestId::new(id).unwrap(),
                        ExecutionPhase::Decode,
                        1,
                        1,
                        0,
                        vec![],
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap();
        let observer = DispatchObserver::new();
        let reader = observer.clone();
        observer.record(
            &batch,
            DispatchRoute::WholeDecode,
            Some(DispatchWork::decode(3)),
            17,
        );
        observer.record(&batch, DispatchRoute::WholeDecode, None, 23);
        let mut grouped = DispatchWork::decode(2);
        grouped.merge(DispatchWork::decode(1));
        observer.record(&batch, DispatchRoute::Grouped, Some(grouped), 31);
        let snapshot = reader.snapshot();
        assert_eq!(
            (
                snapshot.attempts,
                snapshot.enqueued_submissions,
                snapshot.enqueue_failures
            ),
            (3, 2, 1)
        );
        assert_eq!((snapshot.offered_rows, snapshot.offered_tokens), (9, 9));
        assert_eq!(
            (
                snapshot.whole_decode_submissions,
                snapshot.grouped_policy_submissions
            ),
            (1, 1)
        );
        assert_eq!(snapshot.work.decode_lane_calls[..4], [0, 1, 1, 1]);
        assert_eq!(
            snapshot.host_submit_nanos, 71,
            "count submit duration once, not per row"
        );
        assert_eq!(snapshot.row_wise_fallbacks, [0; DecodeFallback::ALL.len()]);
    }

    #[test]
    fn row_wise_prefill_chunks_are_not_cross_request_decode_batches() {
        let batch = ExecutionBatch::new(vec![
            ExecutionSegment::new(
                RequestId::new(1).unwrap(),
                ExecutionPhase::Prefill,
                1,
                17,
                0,
                vec![],
            )
            .unwrap(),
        ])
        .unwrap();
        let observer = DispatchObserver::new();
        observer.record(
            &batch,
            DispatchRoute::RowWise(DecodeFallback::SingleRow),
            Some(DispatchWork::prefill(17, 16, Some(8))),
            u64::MAX,
        );
        observer.record(
            &batch,
            DispatchRoute::RowWise(DecodeFallback::SingleRow),
            Some(DispatchWork::prefill(17, 16, Some(8))),
            1,
        );
        let snapshot = observer.snapshot();
        assert_eq!(snapshot.work.prefill_rows, 2);
        assert_eq!(snapshot.work.prefill_tokens, 34);
        assert_eq!(snapshot.work.prefill_chunk_calls, 4);
        assert_eq!(snapshot.work.prefill_chunk_tokens, 32);
        assert_eq!(snapshot.work.decode_lane_calls, [0; MAX_BATCH_MEMBERS + 1]);
        assert_eq!(
            snapshot.row_wise_fallbacks[DecodeFallback::SingleRow as usize],
            2
        );
        assert_eq!(snapshot.host_submit_nanos, u64::MAX);
    }
}
