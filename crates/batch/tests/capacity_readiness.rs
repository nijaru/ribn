//! Real byte-pool/batch-runtime readiness, with a bounded caller-owned wake channel.
use std::convert::Infallible;
use std::sync::{Arc, mpsc};
use std::task::{Wake, Waker};
use std::time::Duration;

use ribn_batch::{
    BatchConfig, BatchExecutor, BatchRuntime, BatchSelection, BlockReason, CancelOutcome,
    EnqueueError, Job, JobOutput, RuntimeError, StepOutcome,
};
use ribn_foundation::{BytePool, ParameterVersion};

struct Notify(mpsc::SyncSender<()>);
impl Wake for Notify {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.0.try_send(());
    }
}
fn notification() -> (Waker, mpsc::Receiver<()>) {
    let (sender, receiver) = mpsc::sync_channel(1);
    (Waker::from(Arc::new(Notify(sender))), receiver)
}
fn notified(receiver: &mpsc::Receiver<()>) {
    // A test deadline, not periodic capacity polling: one notification must arrive.
    receiver
        .recv_timeout(Duration::from_secs(3))
        .expect("resource notification");
}

/// The fixture executor's retained charge is exactly its input envelope.
struct Envelope;
impl BatchExecutor for Envelope {
    type Input = u64;
    type Output = u64;
    type Error = Infallible;
    type Constraint = Infallible;
    fn parameter_version(&self) -> ParameterVersion {
        ParameterVersion::new(1)
    }
    fn max_batch_items(&self) -> usize {
        4
    }
    fn select_batch(&self, candidates: &[&Self::Input]) -> BatchSelection<Self::Constraint> {
        BatchSelection::Ready {
            items: candidates.len(),
        }
    }
    fn waiting_input_bytes(&self, _input: &Self::Input) -> usize {
        0
    }

    fn retained_bytes(&self, input: &Self::Input) -> u64 {
        *input
    }
    fn execute(
        &mut self,
        batch: Vec<Job<Self::Input>>,
    ) -> Result<Vec<JobOutput<Self::Output>>, EnqueueError<Self::Error>> {
        Ok(batch
            .into_iter()
            .map(|job| {
                let (request, input, lease) = job.into_parts();
                JobOutput::new(request, input, lease)
            })
            .collect())
    }
    fn retire(&mut self, _outputs: Vec<JobOutput<Self::Output>>) {
        // No device storage: ordinary drop is safe retirement for this fixture.
    }
}
fn runtime(pool: &Arc<BytePool>) -> BatchRuntime<Envelope> {
    BatchRuntime::new(
        Envelope,
        Arc::clone(pool),
        BatchConfig {
            max_waiting_requests: 8,
            max_retained_results: 4,
            ..BatchConfig::default()
        },
    )
    .expect("runtime")
}
fn blocked(runtime: &mut BatchRuntime<Envelope>) {
    assert_eq!(
        runtime.step().unwrap(),
        StepOutcome::Blocked(BlockReason::Pool {
            requested: 64,
            available: 0,
        })
    );
}
fn executed(runtime: &mut BatchRuntime<Envelope>) {
    assert_eq!(
        runtime.step().unwrap(),
        StepOutcome::Executed { results: 1 }
    );
}

#[test]
fn submit_after_idle_notifies_only_this_runtime() {
    let pool = BytePool::new(64).shared();
    let mut owner = runtime(&pool);
    let mut sibling = runtime(&pool);
    let mut wait = owner.local_wait();
    let sibling_wait = sibling.local_wait();
    assert_eq!(owner.step().unwrap(), StepOutcome::Idle);
    sibling.submit(1).unwrap();
    assert!(sibling_wait.changed());
    assert!(!wait.changed());
    assert!(owner.pop_completed().is_none());
    assert!(!wait.changed());
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    owner.submit(64).unwrap();
    notified(&receiver);
    executed(&mut owner);
}

#[test]
fn queued_cancellation_before_arming_wakes_a_pool_block_without_a_release() {
    let pool = BytePool::new(64).shared();
    let _held = pool.reserve(64).unwrap();
    let mut owner = runtime(&pool);
    let request = owner.submit(64).unwrap();
    let mut local = owner.local_wait();
    let pool_wait = owner.capacity_wait();
    blocked(&mut owner);
    assert_eq!(owner.cancel(request), CancelOutcome::Queued);
    let unchanged = owner.local_wait();
    assert_eq!(owner.cancel(request), CancelOutcome::Queued);
    assert!(!unchanged.changed(), "duplicate intent changes no state");
    assert!(!pool_wait.changed());
    let (waker, receiver) = notification();
    local.wake_on_change(&waker);
    notified(&receiver);
    assert_eq!(owner.step().unwrap(), StepOutcome::Cancelled { request });
    assert_eq!(pool.granted(), 64);
}

#[test]
fn dequeuing_notifies_local_delivery_without_refunding_the_live_output() {
    let pool = BytePool::new(128).shared();
    let mut owner = BatchRuntime::new(
        Envelope,
        Arc::clone(&pool),
        BatchConfig {
            max_retained_results: 1,
            ..BatchConfig::default()
        },
    )
    .unwrap();
    owner.submit(64).unwrap();
    executed(&mut owner);
    owner.submit(64).unwrap();
    let mut local = owner.local_wait();
    let capacity = owner.capacity_wait();
    assert_eq!(
        owner.step().unwrap(),
        StepOutcome::Blocked(BlockReason::RetainedResults)
    );
    let (waker, receiver) = notification();
    local.wake_on_change(&waker);
    let output = owner.pop_completed().unwrap();
    notified(&receiver);
    assert!(!capacity.changed());
    assert_eq!(pool.granted(), 64);
    executed(&mut owner);
    assert_eq!(pool.granted(), 128);
    drop(output);
    assert!(capacity.changed());
}

#[test]
fn broken_local_waker_cannot_hide_an_accepted_request_or_drop_a_dequeued_lease() {
    struct Panicker;
    impl Wake for Panicker {
        fn wake(self: Arc<Self>) {
            panic!("injected notification failure");
        }
    }
    let pool = BytePool::new(64).shared();
    let mut owner = runtime(&pool);
    let panic_waker = Waker::from(Arc::new(Panicker));
    let (waker, receiver) = notification();
    let mut broken = owner.local_wait();
    let mut healthy = owner.local_wait();
    broken.wake_on_change(&panic_waker);
    healthy.wake_on_change(&waker);
    let request = owner.submit(64).unwrap();
    notified(&receiver);
    executed(&mut owner);
    let mut broken = owner.local_wait();
    broken.wake_on_change(&panic_waker);
    let output = owner.pop_completed().unwrap();
    assert_eq!(output.request(), request);
    assert_eq!(pool.granted(), 64);
    drop(output);
    assert_eq!(pool.granted(), 0);
}

#[test]
fn a_release_before_arming_is_not_lost() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).unwrap();
    let mut runtime = runtime(&pool);
    runtime.submit(64).unwrap();
    let mut wait = runtime.capacity_wait();
    blocked(&mut runtime);
    assert!(!wait.changed());
    drop(held);
    assert!(wait.changed());
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    notified(&receiver);
    executed(&mut runtime);
}

#[test]
fn a_registration_is_taken_from_this_runtimes_own_pool() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).unwrap();
    let mut runtime = runtime(&pool);
    runtime.submit(64).unwrap();
    let mut wait = runtime.capacity_wait();
    blocked(&mut runtime);
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    let other = BytePool::new(64).shared();
    drop(other.reserve(64).unwrap());
    assert!(!wait.changed());
    assert!(receiver.try_recv().is_err());
    drop(held);
    notified(&receiver);
    assert!(wait.changed());
}

#[test]
fn a_release_from_another_thread_wakes_without_polling() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).unwrap();
    let mut runtime = runtime(&pool);
    runtime.submit(64).unwrap();
    let mut wait = runtime.capacity_wait();
    blocked(&mut runtime);
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    let releaser = std::thread::spawn(move || drop(held));
    notified(&receiver);
    releaser.join().unwrap();
    executed(&mut runtime);
}

#[test]
fn pool_closure_reactivates_a_wait_without_releasing_live_charges() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).unwrap();
    let mut runtime = runtime(&pool);
    runtime.submit(64).unwrap();
    let mut wait = runtime.capacity_wait();
    blocked(&mut runtime);
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    pool.close();
    notified(&receiver);
    assert!(wait.changed());
    assert_eq!(pool.granted(), 64);
    assert!(matches!(runtime.step(), Err(RuntimeError::PoolClosed)));
    let mut already_closed = pool.capacity_wait();
    already_closed.wake_on_change(&waker);
    pool.close();
    assert!(
        !already_closed.changed(),
        "repeated closure changes no condition"
    );
    assert!(receiver.try_recv().is_err());
    drop(held);
    assert_eq!(pool.granted(), 0);
}

#[test]
fn cancelling_and_dropping_a_registration_detaches_its_notification() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).unwrap();
    let mut runtime = runtime(&pool);
    let request = runtime.submit(64).unwrap();
    let mut wait = runtime.capacity_wait();
    blocked(&mut runtime);
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    assert_eq!(runtime.cancel(request), ribn_batch::CancelOutcome::Queued);
    drop(wait);
    drop(held);
    assert!(receiver.try_recv().is_err());
    assert_eq!(pool.granted(), 0);
}

#[test]
fn dequeued_output_keeps_its_charge_and_only_retirement_wakes_a_sibling() {
    let pool = BytePool::new(64).shared();
    let mut first = runtime(&pool);
    let mut second = runtime(&pool);
    first.submit(64).unwrap();
    executed(&mut first);
    second.submit(64).unwrap();
    let mut wait = second.capacity_wait();
    blocked(&mut second);
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    let output = first.pop_completed().unwrap();
    assert_eq!(pool.granted(), 64);
    assert!(!wait.changed());
    assert!(receiver.try_recv().is_err());
    drop(output);
    notified(&receiver);
    executed(&mut second);
}

#[test]
fn stolen_capacity_requires_a_fresh_registration_and_second_publication() {
    let pool = BytePool::new(64).shared();
    let held = pool.reserve(64).unwrap();
    let mut runtime = runtime(&pool);
    runtime.submit(64).unwrap();
    let mut wait = runtime.capacity_wait();
    blocked(&mut runtime);
    let (waker, receiver) = notification();
    wait.wake_on_change(&waker);
    drop(held);
    notified(&receiver);
    let competitor = pool.reserve(64).unwrap();
    let mut retry = runtime.capacity_wait();
    blocked(&mut runtime);
    retry.wake_on_change(&waker);
    assert!(!retry.changed());
    assert!(receiver.try_recv().is_err());
    drop(competitor);
    notified(&receiver);
    executed(&mut runtime);
}
