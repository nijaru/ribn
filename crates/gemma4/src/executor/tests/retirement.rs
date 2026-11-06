//! Completed errors may be destroyed; uncertain cleanup retains real pool leases.
use super::*;

#[test]
fn completed_execution_failure_can_teardown_without_resuming() {
    let probe = Arc::new(Probe::default());
    let mut engine = engine(probe.clone(), Mode::Failure);
    engine.step().unwrap();
    assert!(engine.step().is_err());
    while let Some(event) = engine.pop_event() {
        assert!(matches!(
            event,
            Event::Finished {
                reason: ribn::FinishReason::Failed(_),
                ..
            }
        ));
    }
    assert_eq!(probe.live.load(Ordering::SeqCst), 2);
    for _ in 0..2 {
        engine
            .shutdown()
            .expect("CPU error leaves no queued device work");
        assert_eq!(probe.live.load(Ordering::SeqCst), 0);
    }
    assert!(engine.step().is_err(), "teardown must not reopen execution");
    drop(engine);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 2);
    assert_eq!(probe.live.load(Ordering::SeqCst), 0);
    assert_eq!(probe.pool.available(), 64);
    assert!(probe.model.lock().unwrap().upgrade().is_none());
}

#[test]
fn direct_executor_drop_releases_completed_error_and_unwind_states() {
    let mut leaked = Vec::new();
    for mode in [Mode::DropFailure, Mode::DropUnwind] {
        let probe = Arc::new(Probe::default());
        let mut engine = engine(probe.clone(), mode);
        engine.step().unwrap();
        assert!(engine.step().is_err());
        drop(engine);
        let live = probe.live.load(Ordering::SeqCst);
        if live != 0 {
            leaked.push((mode, live));
        }
        assert_eq!(probe.pool.available(), 64);
        assert!(probe.model.lock().unwrap().upgrade().is_none());
    }
    assert!(leaked.is_empty(), "unnecessary retention: {leaked:?}");
}

#[test]
fn failed_completion_retains_charges_until_retry_then_releases() {
    let probe = Arc::new(Probe::default());
    let mut engine = engine(probe.clone(), Mode::Failure);
    engine.step().unwrap();
    assert!(engine.step().is_err());
    *probe.completion.lock().unwrap() = Completion::Waiting;
    for _ in 0..2 {
        assert!(engine.shutdown().is_err());
        assert_eq!(probe.pool.available(), 0);
        assert_eq!(probe.live.load(Ordering::SeqCst), 2);
    }
    *probe.completion.lock().unwrap() = Completion::Ready;
    engine.shutdown().unwrap();
    assert_eq!(probe.pool.available(), 64);
    assert_eq!(probe.live.load(Ordering::SeqCst), 0);
    assert!(engine.step().is_err());
    drop(engine);
    assert!(probe.model.lock().unwrap().upgrade().is_none());
}

#[test]
fn partial_sequence_close_is_retryable_without_double_refund() {
    for mode in [Mode::Failure, Mode::Lifecycle] {
        let probe = Arc::new(Probe::default());
        probe.fail_close_at.store(2, Ordering::SeqCst);
        let mut engine = engine(probe.clone(), mode);
        if mode == Mode::Failure {
            engine.step().unwrap();
            assert!(engine.step().is_err());
            assert!(engine.shutdown().is_err());
            assert_eq!(probe.live.load(Ordering::SeqCst), 2);
        } else {
            let mut failed = false;
            for _ in 0..8 {
                if engine.step().is_err() {
                    failed = true;
                    break;
                }
            }
            assert!(failed, "normal release must expose its close failure");
            assert_eq!(probe.live.load(Ordering::SeqCst), 1);
        }
        assert_eq!(probe.pool.available(), 32);
        engine.shutdown().unwrap();
        engine.shutdown().unwrap();
        assert_eq!(probe.pool.available(), 64);
        assert_eq!(probe.live.load(Ordering::SeqCst), 0);
        drop(engine);
        assert!(probe.model.lock().unwrap().upgrade().is_none());
    }
}

#[test]
fn failed_admission_faults_the_owner_and_retains_partial_construction() {
    let probe = Arc::new(Probe::default());
    *probe.completion.lock().unwrap() = Completion::Waiting;
    let mut engine =
        Engine::with_defaults(counted(probe.clone(), Some(Mode::AdmissionFailure))).unwrap();
    for token in [1, 2] {
        engine
            .enqueue(TokenRequest::new(vec![token], GenerationOptions::default()))
            .unwrap();
    }
    assert!(
        engine.step().is_err(),
        "uncertain construction is owner-fatal"
    );
    assert!(engine.step().is_err(), "failed admission cannot resume");
    let mut failed = 0;
    while let Some(event) = engine.pop_event() {
        match event {
            Event::Finished {
                reason: ribn::FinishReason::Failed(_),
                ..
            } => failed += 1,
            event => panic!("unexpected event after fatal admission: {event:?}"),
        }
    }
    assert_eq!(failed, 2);
    assert_eq!(probe.creations.load(Ordering::SeqCst), 1);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
    assert_eq!(probe.live.load(Ordering::SeqCst), 0);
    assert_eq!(
        probe.pool.available(),
        32,
        "partial construction stays model-owned"
    );
    assert!(engine.shutdown().is_err());
    assert_eq!(probe.pool.available(), 32);
    assert!(probe.model.lock().unwrap().upgrade().is_some());
    *probe.completion.lock().unwrap() = Completion::Ready;
    engine.shutdown().unwrap();
    assert_eq!(probe.pool.available(), 64);
    drop(engine);
    assert!(probe.model.lock().unwrap().upgrade().is_none());
}

#[test]
fn fatal_construction_retains_existing_peers_until_retryable_retirement() {
    let probe = Arc::new(Probe::default());
    probe.fail_creation_at.store(2, Ordering::SeqCst);
    *probe.completion.lock().unwrap() = Completion::Waiting;
    let mut engine =
        Engine::with_defaults(counted(probe.clone(), Some(Mode::AdmissionFailure))).unwrap();
    for token in [1, 2, 3] {
        engine
            .enqueue(TokenRequest::new(vec![token], GenerationOptions::default()))
            .unwrap();
    }
    assert!(engine.step().is_err());
    assert_eq!(probe.creations.load(Ordering::SeqCst), 2);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
    assert_eq!(probe.live.load(Ordering::SeqCst), 1);
    assert_eq!(probe.pool.available(), 0);
    assert!(engine.shutdown().is_err());
    assert!(engine.shutdown().is_err());
    assert_eq!(probe.close_calls.load(Ordering::SeqCst), 0);
    assert_eq!(probe.pool.available(), 0);
    *probe.completion.lock().unwrap() = Completion::Ready;
    engine.shutdown().unwrap();
    engine.shutdown().unwrap();
    assert_eq!(probe.close_calls.load(Ordering::SeqCst), 1);
    assert_eq!(probe.pool.available(), 64);
    assert_eq!(probe.live.load(Ordering::SeqCst), 0);
}

#[test]
fn clean_construction_refusal_and_invalid_input_leave_a_peer_usable() {
    let probe = Arc::new(Probe::default());
    let mut engine =
        Engine::with_defaults(counted(probe.clone(), Some(Mode::AdmissionFailure))).unwrap();
    let mut requests = Vec::new();
    for token in [u32::MAX, 1, 2] {
        requests.push(
            engine
                .enqueue(TokenRequest::new(
                    vec![token],
                    GenerationOptions {
                        max_output_tokens: 1,
                        ..GenerationOptions::default()
                    },
                ))
                .unwrap(),
        );
    }
    for _ in 0..8 {
        engine.step().unwrap();
        if engine.status().requests == 0 {
            break;
        }
    }
    assert_eq!(engine.status().requests, 0);
    let mut finished = Vec::new();
    while let Some(event) = engine.pop_event() {
        if let Event::Finished {
            request, reason, ..
        } = event
        {
            finished.push((request, reason));
        }
    }
    for request in &requests[..2] {
        assert!(
            finished.iter().any(
                |(id, reason)| id == request && matches!(reason, ribn::FinishReason::Failed(_))
            )
        );
    }
    assert!(finished.contains(&(requests[2], ribn::FinishReason::Length)));
    assert_eq!(probe.creations.load(Ordering::SeqCst), 2);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
    assert_eq!(probe.pool.available(), 64);
    engine.shutdown().unwrap();
}

#[test]
fn sequence_close_unwind_cannot_leave_execution_runnable() {
    let probe = Arc::new(Probe::default());
    probe.close_unwind.store(true, Ordering::SeqCst);
    let mut engine = engine(probe.clone(), Mode::ReleaseUnwind);
    let mut failed = false;
    for _ in 0..8 {
        if engine.step().is_err() {
            failed = true;
            break;
        }
    }
    assert!(failed);
    assert_eq!(probe.pool.available(), 0);
    engine.shutdown().unwrap();
    assert_eq!(probe.pool.available(), 64);
    drop(engine);
    assert!(probe.model.lock().unwrap().upgrade().is_none());
}

#[test]
fn defensive_drop_retains_only_unresolved_cleanup() {
    for completion in [Completion::Waiting, Completion::Unwind] {
        let probe = Arc::new(Probe::default());
        *probe.completion.lock().unwrap() = completion;
        let mut engine = engine(probe.clone(), Mode::DropFailure);
        engine.step().unwrap();
        assert!(engine.step().is_err());
        drop(engine);
        assert_eq!(probe.pool.available(), 0);
        assert_eq!(probe.live.load(Ordering::SeqCst), 2);
        assert!(probe.model.lock().unwrap().upgrade().is_some());
    }
}
