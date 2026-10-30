//! State-only ranges cross the real engine, decoder and retirement envelope.
use super::*;

fn intermediate_engine(probe: Arc<Probe>, mode: Mode) -> Engine {
    let mut engine = engine(probe, mode);
    engine
        .set_policy(ribn::SchedulePolicy {
            prefill_chunk_tokens: 1,
            ..ribn::SchedulePolicy::default()
        })
        .unwrap();
    engine
}

#[test]
fn intermediate_prefill_advances_without_requesting_logits() {
    let probe = Arc::new(Probe::default());
    let mut engine = intermediate_engine(probe.clone(), Mode::Lifecycle);
    engine.step().unwrap();
    engine.step().unwrap();
    assert!(engine.pop_event().is_none());
    assert_eq!(probe.advance_calls.load(Ordering::SeqCst), 2);
    assert_eq!(probe.head_calls.load(Ordering::SeqCst), 0);

    // The next validated batch proves both logical cursors advanced without
    // logits. Only the shorter prompt reaches its end and requests a head.
    engine.step().unwrap();
    assert!(matches!(engine.pop_event(), Some(Event::Token { .. })));
    assert!(engine.pop_event().is_none());
    assert_eq!(probe.advance_calls.load(Ordering::SeqCst), 3);
    assert_eq!(probe.head_calls.load(Ordering::SeqCst), 1);
    engine.shutdown().unwrap();
    assert_eq!(probe.pool.available(), 64);
    assert_eq!(probe.live.load(Ordering::SeqCst), 0);
}

#[test]
fn partial_state_only_failure_retains_every_continuation_until_settlement() {
    let probe = Arc::new(Probe::default());
    let mut engine = intermediate_engine(probe.clone(), Mode::Failure);
    engine.step().unwrap();
    // Boundary::poll checks all logical cursors/budgets remain uncommitted,
    // despite physical progress in the first decoder, and rejects replay/release.
    assert!(engine.step().is_err());
    assert_eq!(probe.advance_calls.load(Ordering::SeqCst), 2);
    assert_eq!(probe.head_calls.load(Ordering::SeqCst), 0);
    *probe.completion.lock().unwrap() = Completion::Waiting;
    assert!(engine.shutdown().is_err());
    assert_eq!(probe.live.load(Ordering::SeqCst), 2);
    assert_eq!(probe.pool.available(), 0);
    *probe.completion.lock().unwrap() = Completion::Ready;
    engine.shutdown().unwrap();
    engine.shutdown().unwrap();
    assert_eq!(probe.live.load(Ordering::SeqCst), 0);
    assert_eq!(probe.pool.available(), 64);
    assert!(engine.step().is_err());
    drop(engine);
    assert!(probe.model.lock().unwrap().upgrade().is_none());
}
