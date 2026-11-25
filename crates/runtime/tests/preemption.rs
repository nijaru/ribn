//! Adversarial host contract tests for completion-safe recomputation preemption
//! (`docs/resource-protocol.md#7`, `docs/roadmap.md#4b`).
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ribn::{
    Admission, BatchItem, BatchPreparation, Engine, EngineConfig, EngineError, Event,
    ExecutionError, ExecutorInfo, FinishReason, GenerationExecutor, GenerationLimits,
    GenerationOptions, PreparedRow, RequestId, SchedulePolicy, SequenceId, StepCompletion,
    SubmissionId, TokenRequest, Usage,
};
use ribn_foundation::{BytePool, PoolLease, ReserveError};

#[derive(Default)]
struct Trace {
    admissions: Vec<(RequestId, Vec<u32>, u32)>,
    releases: Vec<SequenceId>,
    fail_next_release: bool,
}

struct SequenceData {
    _base: PoolLease,
    growth: Vec<PoolLease>,
    tokens: Vec<u32>,
    prefix_hit: u32,
}

struct PreparedRowState {
    item: BatchItem,
    lease: PoolLease,
}

/// Deterministic partial-envelope host executor over a real `BytePool`.
///
/// Each sequence reserves 1 base byte on admission and 1 growth byte per
/// consumed token (beyond any prefix hit). Next token is deterministically
/// `base_tag + position`, so replayed histories produce the exact same token
/// stream regardless of preemption points.
struct PreemptibleModel {
    info: ExecutorInfo,
    pool: Arc<BytePool>,
    trace: Arc<Mutex<Trace>>,
    /// Simulated prefix cache: remembers exact full prefixes of length >= 2.
    prefix_cache_enabled: bool,
    cached_prefixes: Vec<Vec<u32>>,
    sequences: HashMap<SequenceId, SequenceData>,
    prepared: Option<Vec<PreparedRowState>>,
    pending: Option<Vec<PreparedRowState>>,
    next_sub: u64,
}

impl PreemptibleModel {
    fn new(pool: Arc<BytePool>, trace: Arc<Mutex<Trace>>, max_sequences: usize) -> Self {
        Self {
            info: ExecutorInfo {
                name: "host preemptible byte-growth model".into(),
                limits: GenerationLimits {
                    context_tokens: 64,
                    max_sequences,
                    max_batch_tokens: 16,
                    max_decode_tokens: 1,
                },
            },
            pool,
            trace,
            prefix_cache_enabled: false,
            cached_prefixes: Vec::new(),
            sequences: HashMap::new(),
            prepared: None,
            pending: None,
            next_sub: 0,
        }
    }
}

impl GenerationExecutor for PreemptibleModel {
    fn info(&self) -> &ExecutorInfo {
        &self.info
    }

    fn supports_preemption(&self) -> bool {
        true
    }

    fn admitted_prefix(&self, sequence: SequenceId) -> u32 {
        self.sequences.get(&sequence).map_or(0, |s| s.prefix_hit)
    }

    fn admit(
        &mut self,
        request_id: RequestId,
        sequence: SequenceId,
        request: &TokenRequest,
    ) -> Result<Admission, ExecutionError> {
        let wait = self.pool.capacity_wait();
        let base = match self.pool.reserve(1) {
            Ok(lease) => lease,
            Err(ReserveError::Exhausted { .. }) => return Ok(Admission::Deferred(wait)),
            Err(error) => return Err(ExecutionError::new(error.to_string())),
        };
        let mut prefix_hit = 0_u32;
        if self.prefix_cache_enabled && request.tokens.len() > 1 {
            for cached in &self.cached_prefixes {
                let common = cached
                    .iter()
                    .zip(request.tokens.iter())
                    .take_while(|(a, b)| a == b)
                    .count()
                    .min(request.tokens.len() - 1);
                prefix_hit = prefix_hit.max(u32::try_from(common).unwrap());
            }
        }
        self.trace.lock().unwrap().admissions.push((
            request_id,
            request.tokens.to_vec(),
            request.options.max_output_tokens,
        ));
        self.sequences.insert(
            sequence,
            SequenceData {
                _base: base,
                growth: Vec::new(),
                tokens: request.tokens.to_vec(),
                prefix_hit,
            },
        );
        Ok(Admission::Ready)
    }

    fn prepare(&mut self, batch: &[BatchItem]) -> Result<BatchPreparation, ExecutionError> {
        assert!(self.prepared.is_none() && self.pending.is_none());
        let mut rows = Vec::with_capacity(batch.len());
        let mut accepted = Vec::with_capacity(batch.len());

        for &offered in batch {
            let seq = &self.sequences[&offered.sequence];
            // Single-sequence impossible growth rejects locally rather than self-preempting.
            let needed = u64::from(offered.token_budget);
            let total_possible = self.pool.capacity();
            let already_held = 1 + seq.growth.iter().map(PoolLease::bytes).sum::<u64>();
            if already_held + needed > total_possible {
                rows.push(PreparedRow::Rejected(ExecutionError::new(
                    "single sequence exceeds entire pool capacity",
                )));
                continue;
            }
            let wait = self.pool.capacity_wait();
            match self.pool.reserve(needed) {
                Ok(lease) => {
                    accepted.push(PreparedRowState {
                        item: offered,
                        lease,
                    });
                    rows.push(PreparedRow::Ready(offered));
                }
                Err(ReserveError::Exhausted { .. }) => {
                    rows.push(PreparedRow::Deferred(wait));
                }
                Err(error) => {
                    rows.push(PreparedRow::Rejected(ExecutionError::new(
                        error.to_string(),
                    )));
                }
            }
        }

        self.prepared = Some(accepted);
        Ok(BatchPreparation::Selected(rows))
    }

    fn abandon_preparation(&mut self) -> Result<(), ExecutionError> {
        self.prepared = None;
        Ok(())
    }

    fn submit(&mut self, batch: &[BatchItem]) -> Result<SubmissionId, ExecutionError> {
        let prepared = self.prepared.take().expect("prepared batch");
        assert_eq!(prepared.len(), batch.len());
        self.pending = Some(prepared);
        self.next_sub += 1;
        Ok(SubmissionId::new(self.next_sub))
    }

    fn poll(&mut self, _: SubmissionId) -> Result<Option<Vec<StepCompletion>>, ExecutionError> {
        let Some(batch) = self.pending.take() else {
            return Ok(None);
        };
        let mut completions = Vec::with_capacity(batch.len());
        for row in batch {
            let seq = self.sequences.get_mut(&row.item.sequence).unwrap();
            seq.growth.push(row.lease);
            let end = row.item.prefix + row.item.token_budget;
            let base_tag = seq.tokens[0];
            let tokens = if row.item.output_budget > 0 {
                // Deterministic function of prompt tag + logical position!
                let next_tok = base_tag * 100 + end;
                seq.tokens.push(next_tok);
                if self.prefix_cache_enabled {
                    self.cached_prefixes.push(seq.tokens.clone());
                }
                vec![next_tok]
            } else {
                Vec::new()
            };
            completions.push(StepCompletion {
                sequence: row.item.sequence,
                prefix: end,
                tokens,
            });
        }
        Ok(Some(completions))
    }

    fn release(&mut self, sequence: SequenceId) -> Result<(), ExecutionError> {
        // Completion-safety invariant: never release while prepared or in flight!
        assert!(
            !self
                .prepared
                .iter()
                .chain(&self.pending)
                .flatten()
                .any(|row| row.item.sequence == sequence),
            "cannot release active in-flight sequence"
        );
        let mut trace = self.trace.lock().unwrap();
        if trace.fail_next_release {
            trace.fail_next_release = false;
            return Err(ExecutionError::new("injected release failure"));
        }
        if self.sequences.remove(&sequence).is_some() {
            trace.releases.push(sequence);
        }
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), ExecutionError> {
        self.prepared = None;
        self.pending = None;
        Ok(())
    }
}

fn test_policy() -> SchedulePolicy {
    SchedulePolicy {
        max_batch_tokens: 16,
        prefill_chunk_tokens: 8,
        decode_tokens: 1,
        ..SchedulePolicy::default()
    }
}

fn req(tag: u32, prompt_len: usize, max_output_tokens: u32) -> TokenRequest {
    let tokens: Vec<u32> = (0..prompt_len)
        .map(|i| tag + u32::try_from(i).unwrap())
        .collect();
    TokenRequest::new(
        tokens,
        GenerationOptions {
            max_output_tokens,
            ..GenerationOptions::default()
        },
    )
}

#[test]
fn all_active_growth_deadlock_preempts_victim_finishes_survivor_and_replays_without_duplication() {
    // Tight pool: 10 bytes total.
    // Request A (tag=1, prompt=2, out=4) needs 1 base + 2 prefill + 3 decode = 6 bytes alone.
    // Request B (tag=2, prompt=2, out=4) needs 6 bytes alone.
    // Together they admit (2 base) + prefill (4 bytes = 6 total), grow 2 steps each (10 total),
    // and then BOTH deadlock on step 3! Preemption must free B, finish A, then replay B cleanly.
    let pool = BytePool::new(10).shared();
    let trace = Arc::new(Mutex::new(Trace::default()));
    let mut model = PreemptibleModel::new(Arc::clone(&pool), Arc::clone(&trace), 2);
    model.prefix_cache_enabled = true;

    let mut engine = Engine::new(
        model,
        EngineConfig {
            max_active_requests: 2,
            max_queued_requests: 4,
            max_queued_input_tokens: 64,
            max_buffered_events: 32,
            max_events_per_request: 16,
        },
        SchedulePolicy {
            max_batch_tokens: 16,
            prefill_chunk_tokens: 8,
            decode_tokens: 1,
            ..SchedulePolicy::default()
        },
    )
    .unwrap();

    let req_a = engine.enqueue(req(1, 2, 4)).unwrap();
    let req_b = engine.enqueue(req(2, 2, 4)).unwrap();

    let mut tokens_a = Vec::new();
    let mut tokens_b = Vec::new();
    let mut finish_a = None;
    let mut finish_b = None;

    for _ in 0..40 {
        engine.step().unwrap();
        while let Some(ev) = engine.pop_event() {
            match ev {
                Event::Token { request, token } if request == req_a => tokens_a.push(token),
                Event::Token { request, token } if request == req_b => tokens_b.push(token),
                Event::Finished {
                    request,
                    reason,
                    usage,
                } if request == req_a => finish_a = Some((reason, usage)),
                Event::Finished {
                    request,
                    reason,
                    usage,
                } if request == req_b => finish_b = Some((reason, usage)),
                _ => unreachable!(),
            }
        }
        if engine.status().requests == 0 {
            break;
        }
    }

    assert!(
        engine.preemptions() >= 1,
        "expected recomputation preemption"
    );
    // Exact deterministic tokens with ZERO duplicates despite preemption + replay!
    assert_eq!(tokens_a, vec![102, 103, 104, 105]);
    assert_eq!(tokens_b, vec![202, 203, 204, 205]);
    assert_eq!(
        finish_a,
        Some((
            FinishReason::Length,
            Usage {
                prompt_tokens: 2,
                completion_tokens: 4
            }
        ))
    );
    // Replayed victim B must report its ORIGINAL prompt_tokens (2) and exact completion_tokens (4)!
    assert_eq!(
        finish_b,
        Some((
            FinishReason::Length,
            Usage {
                prompt_tokens: 2,
                completion_tokens: 4
            }
        ))
    );
    // Verify B's replay admission carried its accumulated history and remaining output budget!
    let t = trace.lock().unwrap();
    let b_admissions: Vec<_> = t
        .admissions
        .iter()
        .filter(|(id, _, _)| *id == req_b)
        .collect();
    assert!(b_admissions.len() >= 2, "B must have been readmitted");
    let replay = b_admissions[1];
    assert!(
        replay.1.len() > 2,
        "replay must include already-generated tokens: {:?}",
        replay.1
    );
    assert_eq!(
        replay.1.len() + replay.2 as usize,
        6,
        "history + remaining_output must equal original prompt + max_output_tokens"
    );
    assert_eq!(pool.granted(), 0, "all pool bytes released");
}

#[test]
fn preempted_victim_has_priority_over_fresh_arrivals_no_starvation() {
    // Pool holds 10 bytes: A and B collide -> B is preempted.
    // While A is finishing, C arrives in `waiting`. Once A finishes, B MUST be readmitted
    // BEFORE C so fresh arrivals cannot starve preempted victims!
    let pool = BytePool::new(10).shared();
    let trace = Arc::new(Mutex::new(Trace::default()));
    let model = PreemptibleModel::new(Arc::clone(&pool), Arc::clone(&trace), 2);
    let mut engine = Engine::new(
        model,
        EngineConfig {
            max_active_requests: 2,
            max_queued_requests: 4,
            max_queued_input_tokens: 64,
            max_buffered_events: 32,
            max_events_per_request: 16,
        },
        test_policy(),
    )
    .unwrap();

    let req_a = engine.enqueue(req(1, 2, 4)).unwrap();
    let req_b = engine.enqueue(req(2, 2, 4)).unwrap();

    // Run until preemption occurs, then enqueue newcomer C while A is still protected.
    let mut req_c = None;
    let mut completion_order = Vec::new();
    for _ in 0..60 {
        engine.step().unwrap();
        if engine.preemptions() >= 1 && req_c.is_none() {
            req_c = Some(engine.enqueue(req(3, 2, 3)).unwrap());
        }
        while let Some(ev) = engine.pop_event() {
            if let Event::Finished {
                request, reason, ..
            } = ev
            {
                assert_eq!(reason, FinishReason::Length);
                completion_order.push(request);
            }
        }
        if engine.status().requests == 0 {
            break;
        }
    }

    let req_c = req_c.expect("preemption occurred and C was enqueued");
    assert_eq!(completion_order.len(), 3);
    // Protected survivor A finishes first; victim B finishes before newcomer C!
    assert_eq!(completion_order[0], req_a);
    // Verify B was readmitted before C was ever admitted.
    let admissions = &trace.lock().unwrap().admissions;
    let b_replay_pos = admissions
        .iter()
        .enumerate()
        .filter(|(_, (id, _, _))| *id == req_b)
        .nth(1)
        .expect("B replayed")
        .0;
    let c_first_pos = admissions
        .iter()
        .position(|(id, _, _)| *id == req_c)
        .expect("C admitted");
    assert!(
        b_replay_pos < c_first_pos,
        "preempted victim B (pos {b_replay_pos}) must be readmitted before newcomer C (pos {c_first_pos})"
    );
    assert_eq!(pool.granted(), 0);
}

#[test]
fn stalled_survivor_is_preempted_for_healthy_peer_and_resumes_after_drain() {
    // Pool holds 8 bytes; max_events_per_request = 2 (1 token + 1 terminal = full!).
    // A generates 1 token and its consumer STOPS draining (credits < 2).
    // B needs bytes held by A to complete. Engine must preempt stalled A, let B finish,
    // and then once A's consumer drains, readmit and finish A with zero duplicate tokens!
    let pool = BytePool::new(8).shared();
    let trace = Arc::new(Mutex::new(Trace::default()));
    let model = PreemptibleModel::new(Arc::clone(&pool), Arc::clone(&trace), 2);
    let mut engine = Engine::new(
        model,
        EngineConfig {
            max_active_requests: 2,
            max_queued_requests: 2,
            max_queued_input_tokens: 64,
            max_buffered_events: 8,
            max_events_per_request: 2,
        },
        test_policy(),
    )
    .unwrap();

    let req_a = engine.enqueue(req(1, 2, 3)).unwrap();
    // Run A through prefill so it emits 1 token into its 2-slot mailbox -> now A has only 1 credit (< 2)!
    engine.step().unwrap(); // submit A prefill
    engine.step().unwrap(); // complete A prefill (1 token buffered -> A is output-stalled!)
    assert_eq!(engine.status().buffered_events, 1);

    // Enqueue healthy peer B (needs 1 base + 2 prefill + 2 decode = 5 bytes, while A holds 3 bytes -> 8 total;
    // actually B needs 6 bytes so with A holding 3 bytes, B cannot finish unless A is preempted!).
    let req_b = engine.enqueue(req(2, 3, 3)).unwrap();
    let mut tokens_b = Vec::new();
    let mut finished_b = false;

    // Drain ONLY B's events; keep A's consumer completely stalled!
    for _ in 0..30 {
        engine.step().unwrap();
        while let Some(ev) = engine.pop_event_for(req_b) {
            match ev {
                Event::Token { token, .. } => tokens_b.push(token),
                Event::Finished { reason, usage, .. } => {
                    assert_eq!(reason, FinishReason::Length);
                    assert_eq!(usage.prompt_tokens, 3);
                    assert_eq!(usage.completion_tokens, 3);
                    finished_b = true;
                }
            }
        }
        if finished_b {
            break;
        }
    }

    assert!(finished_b, "healthy peer B must finish despite stalled A");
    assert_eq!(tokens_b, vec![203, 204, 205]);
    assert!(
        engine.preemptions() >= 1,
        "stalled A must have been preempted"
    );

    // Now A's consumer wakes up and drains its mailbox -> A resumes and completes with exact tokens!
    let mut tokens_a = Vec::new();
    let mut finished_a = None;
    for _ in 0..30 {
        while let Some(ev) = engine.pop_event_for(req_a) {
            match ev {
                Event::Token { token, .. } => tokens_a.push(token),
                Event::Finished { reason, usage, .. } => finished_a = Some((reason, usage)),
            }
        }
        if finished_a.is_some() {
            break;
        }
        engine.step().unwrap();
    }

    assert_eq!(tokens_a, vec![102, 103, 104]);
    assert_eq!(
        finished_a,
        Some((
            FinishReason::Length,
            Usage {
                prompt_tokens: 2,
                completion_tokens: 3
            }
        ))
    );
    assert_eq!(pool.granted(), 0);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "covers cancelled preempted victim, impossible request rejection, and failed preemption release"
)]
fn cancelled_preempted_victim_failed_preemption_release_and_impossible_request_preserve_ownership()
{
    // 1. Cancelling a preempted victim emits exact prior usage and zero double-release.
    {
        let pool = BytePool::new(10).shared();
        let trace = Arc::new(Mutex::new(Trace::default()));
        let model = PreemptibleModel::new(Arc::clone(&pool), Arc::clone(&trace), 2);
        let mut engine = Engine::new(
            model,
            EngineConfig {
                max_active_requests: 2,
                max_queued_requests: 2,
                max_queued_input_tokens: 64,
                max_buffered_events: 32,
                max_events_per_request: 16,
            },
            test_policy(),
        )
        .unwrap();

        let req_a = engine.enqueue(req(1, 2, 4)).unwrap();
        let req_b = engine.enqueue(req(2, 2, 4)).unwrap();
        let mut b_tokens_before_cancel = 0_u32;
        for _ in 0..20 {
            engine.step().unwrap();
            while let Some(ev) = engine.pop_event() {
                if let Event::Token { request, .. } = ev
                    && request == req_b
                {
                    b_tokens_before_cancel += 1;
                }
            }
            if engine.preemptions() >= 1 {
                engine.cancel(req_b).unwrap();
                break;
            }
        }
        let mut b_cancel = None;
        let mut a_done = false;
        for _ in 0..30 {
            engine.step().unwrap();
            while let Some(ev) = engine.pop_event() {
                match ev {
                    Event::Token { request, .. } if request == req_b => {
                        b_tokens_before_cancel += 1;
                    }
                    Event::Finished {
                        request,
                        reason,
                        usage,
                    } if request == req_b => b_cancel = Some((reason, usage)),
                    Event::Finished {
                        request, reason, ..
                    } if request == req_a => {
                        assert_eq!(reason, FinishReason::Length);
                        a_done = true;
                    }
                    _ => {}
                }
            }
            if engine.status().requests == 0 {
                break;
            }
        }
        assert!(a_done);
        assert_eq!(
            b_cancel,
            Some((
                FinishReason::Cancelled,
                Usage {
                    prompt_tokens: 2,
                    completion_tokens: b_tokens_before_cancel,
                }
            ))
        );
        assert_eq!(pool.granted(), 0);
    }

    // 2. Individually impossible request rejects locally without self-preempting or faulting peer.
    {
        let pool = BytePool::new(5).shared();
        let trace = Arc::new(Mutex::new(Trace::default()));
        let model = PreemptibleModel::new(Arc::clone(&pool), Arc::clone(&trace), 2);
        let mut engine = Engine::new(
            model,
            EngineConfig {
                max_active_requests: 2,
                max_queued_requests: 2,
                max_queued_input_tokens: 64,
                max_buffered_events: 16,
                max_events_per_request: 8,
            },
            test_policy(),
        )
        .unwrap();
        let impossible = engine.enqueue(req(1, 5, 1)).unwrap(); // 1 base + 5 prefill = 6 > 5 pool
        let feasible = engine.enqueue(req(2, 2, 1)).unwrap(); // 1 base + 2 prefill = 3 <= 5
        let mut got_imp = None;
        let mut got_ok = None;
        for _ in 0..20 {
            engine.step().unwrap();
            while let Some(ev) = engine.pop_event() {
                if let Event::Finished {
                    request, reason, ..
                } = ev
                {
                    if request == impossible {
                        got_imp = Some(reason);
                    } else if request == feasible {
                        got_ok = Some(reason);
                    }
                }
            }
            if engine.status().requests == 0 {
                break;
            }
        }
        assert!(matches!(got_imp, Some(FinishReason::Failed(_))));
        assert_eq!(got_ok, Some(FinishReason::Length));
        assert_eq!(engine.preemptions(), 0);
        assert!(!engine.status().faulted);
        assert_eq!(pool.granted(), 0);
    }

    // 3. Failed release during preemption faults the engine and retains ownership until shutdown.
    {
        let pool = BytePool::new(10).shared();
        let trace = Arc::new(Mutex::new(Trace::default()));
        let model = PreemptibleModel::new(Arc::clone(&pool), Arc::clone(&trace), 2);
        let mut engine = Engine::new(
            model,
            EngineConfig {
                max_active_requests: 2,
                max_queued_requests: 2,
                max_queued_input_tokens: 64,
                max_buffered_events: 32,
                max_events_per_request: 16,
            },
            test_policy(),
        )
        .unwrap();
        engine.enqueue(req(1, 2, 4)).unwrap();
        engine.enqueue(req(2, 2, 4)).unwrap();

        let mut faulted = false;
        for step_idx in 0..20 {
            if step_idx == 3 {
                // Inject release failure right before all-active growth deadlock triggers preemption.
                trace.lock().unwrap().fail_next_release = true;
            }
            match engine.step() {
                Ok(_) => while engine.pop_event().is_some() {},
                Err(EngineError::Faulted(_)) => {
                    faulted = true;
                    break;
                }
                Err(other) => panic!("unexpected error: {other}"),
            }
        }
        assert!(faulted, "failed preemption release must fault the engine");
        assert!(engine.status().faulted);
        assert!(
            pool.granted() > 0,
            "charges retained until explicit shutdown"
        );
        engine.shutdown().unwrap();
        assert_eq!(pool.granted(), 0, "shutdown retry releases all charges");
    }
}

#[test]
fn protected_survivor_is_scheduled_ahead_of_older_parked_peers_after_preemption() {
    // Pool holds 9 bytes across 3 concurrent requests (prompt=1, out=4 -> 1 base + 1 prefill + 3 decode = 5 bytes each):
    // - Step 1: all 3 admit (3 base) + prefill (3 bytes -> 6 total).
    // - Step 2: all 3 decode once (3 bytes -> 9 total = FULL!).
    // - Step 3: all 3 park -> Engine protects A, preempts C (frees 3 bytes -> 3 free).
    // - Step 4: A, B wake up -> A gets 1 byte, B gets 1 byte (8 total, 1 free).
    // - Step 5: A gets last 1 byte (9 total, FULL!), B parks first in `self.parked`.
    // - Step 6: A needs 1 more byte -> A parks BEHIND B in `self.parked = [B, A]`!
    //   Engine keeps protected_survivor = Some(A), preempts B (or if 4 requests, preempts C/D).
    // With 4 concurrent requests in a 10-byte pool, after C is preempted while `parked = [B, C, A]`,
    // survivor A MUST be prioritized ahead of B so B cannot steal A's freed bytes!
    let pool = BytePool::new(10).shared();
    let trace = Arc::new(Mutex::new(Trace::default()));
    let model = PreemptibleModel::new(Arc::clone(&pool), Arc::clone(&trace), 4);
    let mut engine = Engine::new(
        model,
        EngineConfig {
            max_active_requests: 4,
            max_queued_requests: 4,
            max_queued_input_tokens: 64,
            max_buffered_events: 64,
            max_events_per_request: 16,
        },
        test_policy(),
    )
    .unwrap();

    let req_a = engine.enqueue(req(1, 1, 5)).unwrap();
    let req_b = engine.enqueue(req(2, 1, 5)).unwrap();
    let req_c = engine.enqueue(req(3, 1, 5)).unwrap();
    let req_d = engine.enqueue(req(4, 1, 5)).unwrap();

    let mut finish_order = Vec::new();
    let mut outputs: HashMap<RequestId, Vec<u32>> = HashMap::new();
    for _ in 0..120 {
        engine.step().unwrap();
        while let Some(ev) = engine.pop_event() {
            match ev {
                Event::Token { request, token } => {
                    outputs.entry(request).or_default().push(token);
                }
                Event::Finished {
                    request, reason, ..
                } => {
                    assert_eq!(reason, FinishReason::Length);
                    finish_order.push(request);
                }
            }
        }
        if engine.status().requests == 0 {
            break;
        }
    }

    assert_eq!(finish_order.len(), 4);
    assert_eq!(
        finish_order[0], req_a,
        "protected survivor A must finish first without peer block stealing"
    );
    assert_eq!(outputs[&req_a], vec![101, 102, 103, 104, 105]);
    assert_eq!(outputs[&req_b], vec![201, 202, 203, 204, 205]);
    assert_eq!(outputs[&req_c], vec![301, 302, 303, 304, 305]);
    assert_eq!(outputs[&req_d], vec![401, 402, 403, 404, 405]);
    assert!(engine.preemptions() >= 2);
    assert_eq!(pool.granted(), 0);
}
