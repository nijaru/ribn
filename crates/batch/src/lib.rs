//! Non-autoregressive dynamic-batch runtime used to pressure-test Ribn's common boundaries.
//!
//! Inputs and outputs are executor-defined. The runtime has no token, prefix, KV,
//! decode, or chat semantics. Its batching policy is intentionally minimal until
//! real encoder/pooling models provide shape and memory evidence.
//!
//! Waiting work is bounded by [`BatchConfig::max_waiting_requests`] and
//! [`BatchConfig::max_waiting_input_bytes`]. Terminal results the caller has not
//! consumed are bounded by [`BatchConfig::max_retained_results`] and by a shared
//! [`BytePool`], because a caller that stops draining results would otherwise grow
//! memory without limit, and because encoder or media outputs can own substantial
//! host and device allocations.
//!
//! The pool is the shared byte authority for retained results, so one constraint
//! binds siblings independently of their local waiting input budgets. Reserving a request's retained output yields an owning [`PoolLease`]
//! that travels with the entry, so a completed result keeps its charge until the
//! caller takes responsibility for the storage it covers.
//!
//! Admission has three outcomes rather than an integer. The executor reports that
//! the oldest candidates are [`BatchSelection::Ready`], temporarily
//! [`BatchSelection::Blocked`], or permanently [`BatchSelection::Rejected`] under
//! its own limits. Temporary contention is not a runtime invariant failure, and
//! every blocking outcome names the condition that must change before
//! [`BatchRuntime::step`] can progress, so a caller never has to retry blindly.
//!
//! Retained bytes are reserved before a batch executes and remain charged while
//! the result owns its lease, so an executor cannot overshoot the budget by
//! running first and accounting afterwards. When the pool cannot cover the whole
//! selected batch, the runtime runs the largest prefix it can reserve instead of
//! blocking. On temporary head contention, a bounded scan can offer a later
//! candidate alone. Each head can be bypassed only a bounded number of times.
//! A blocked scan returns readiness for the head and its eligible peers.
//!
//! Take a [`BatchRuntime::local_wait`] and a [`BatchRuntime::capacity_wait`] before
//! the step. When idle or blocked, arm the local registration before parking. For
//! a blocked step, also call [`BlockReason::wake_on_change`] with the pre-step pool
//! registration. Publication before arming is not lost. The caller owns
//! parking and retry; this runtime adds no worker.
//!
//! Output is executor-defined and may be device-resident. A result that completes
//! asynchronously carries its own producer completion dependency
//! (`is_complete`/`synchronize`/`read` in the executor's own type) and its
//! [`PoolLease`], so a consumer that stalls keeps the bytes charged and a consumer
//! that takes the result takes the charge with it.
//!
//! A granted reservation travels into the submission with the request it covers, and
//! the executor owns it from there. That is what makes a failure unambiguous: the
//! executor either released a reservation nothing was ever built for, or it still
//! holds that reservation together with the storage the device may be writing. The
//! runtime reports which happened ([`EnqueueError`]) and never hands a reservation
//! back to a caller that did not create the storage underneath it. Output the runtime
//! cannot commit is returned through [`BatchExecutor::retire`] for the same reason.

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ribn_foundation::{
    AllocationId, BytePool, ParameterVersion, PoolLease, Readiness, ReadinessWait, ReserveError,
};

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RequestId(u64);

impl RequestId {
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Bounds for waiting work and for terminal results that are still retained.
///
/// Queued input bytes have a local bound. Output bytes are instead charged to the
/// shared [`BytePool`], so sibling runtimes cannot each spend the same capacity.
/// The count bound applies to every terminal entry. Rejections hold no bulk payload,
/// so they are bounded by [`Self::max_retained_results`] alone and stay
/// deliverable while the pool is exhausted.
///
/// Defaults are placeholders. A deployment sizes the limits from
/// its own admission policy, not from the physical pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchConfig {
    /// Requests that may wait for execution.
    pub max_waiting_requests: usize,
    /// Heap bytes retained by accepted inputs still in the waiting queue.
    pub max_waiting_input_bytes: usize,
    /// Terminal entries that may be retained before the caller consumes them.
    pub max_retained_results: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_waiting_requests: 1024,
            max_waiting_input_bytes: 16 * 1024 * 1024,
            max_retained_results: 1024,
        }
    }
}

/// One request handed to an executor, with the reservation covering its storage.
///
/// The reservation is granted during preparation and moves to the executor at
/// submission. From then on the executor owns it: it attaches the charge to whatever
/// storage it materializes for that request, and releases it only when nothing can
/// have reached the device. A reservation that outlives an enqueue failure keeps
/// covering the storage the device may still access.
pub struct Job<I> {
    request: RequestId,
    input: I,
    lease: PoolLease,
}

impl<I> Job<I> {
    #[must_use]
    pub const fn request(&self) -> RequestId {
        self.request
    }

    #[must_use]
    pub fn input(&self) -> &I {
        &self.input
    }

    /// The reservation this request was granted.
    #[must_use]
    pub const fn lease(&self) -> &PoolLease {
        &self.lease
    }

    /// Split into the request identity, its input and the reservation the executor
    /// now owns.
    #[must_use]
    pub fn into_parts(self) -> (RequestId, I, PoolLease) {
        (self.request, self.input, self.lease)
    }
}

/// One result and the reservation covering its storage, from execution to retirement.
///
/// Callers may borrow the output but cannot extract its lease independently. An
/// executor must provide a retirement callback for storage whose destructor does
/// not prove reuse. Undeliverable outputs still go through [`BatchExecutor::retire`].
#[derive(Debug)]
pub struct JobOutput<O> {
    request: RequestId,
    retained: Option<(O, PoolLease)>,
    retire: fn(O, PoolLease),
}

impl<O> JobOutput<O> {
    /// For output whose destructor releases its storage synchronously.
    #[must_use]
    pub fn new(request: RequestId, output: O, lease: PoolLease) -> Self {
        Self::with_retirement(request, output, lease, |output, lease| {
            drop(output);
            drop(lease);
        })
    }

    /// Register the owner that retires both storage and lease when discarded.
    /// The callback must keep the lease until reuse is proven, including on failure.
    #[must_use]
    pub const fn with_retirement(
        request: RequestId,
        output: O,
        lease: PoolLease,
        retire: fn(O, PoolLease),
    ) -> Self {
        Self {
            request,
            retained: Some((output, lease)),
            retire,
        }
    }

    #[must_use]
    pub const fn request(&self) -> RequestId {
        self.request
    }

    #[must_use]
    pub fn output(&self) -> &O {
        &self.retained().0
    }

    /// The reservation covering this result's storage.
    #[must_use]
    pub fn lease(&self) -> &PoolLease {
        &self.retained().1
    }

    fn retained(&self) -> &(O, PoolLease) {
        self.retained.as_ref().expect("taken only in Drop")
    }
}

impl<O> Drop for JobOutput<O> {
    fn drop(&mut self) {
        if let Some((output, lease)) = self.retained.take() {
            (self.retire)(output, lease);
        }
    }
}

/// What the oldest compatible candidates should do next.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BatchSelection<C> {
    /// Execute the oldest `items` candidates.
    ///
    /// The runtime reserves [`BatchExecutor::retained_bytes`] from the shared pool
    /// for every selected input before it runs them, so an executor that
    /// under-reports its retention is what breaks the bound, not the runtime.
    Ready { items: usize },
    /// The oldest candidate is valid but cannot run yet.
    ///
    /// The queue is left untouched. This is temporary contention, not an
    /// executor invariant failure, and it must not be reported as an error.
    Blocked {
        reason: C,
        /// Registration taken before checking the condition reported in `reason`.
        wait: ReadinessWait,
    },
    /// The oldest candidate cannot execute under this executor's limits, and no
    /// amount of waiting changes that.
    ///
    /// Only the queue head can be rejected. During a bypass search the runtime
    /// skips rejected candidates instead of settling them out of order.
    Rejected(C),
}

/// Why the runtime stopped short of executing work.
///
/// Every variant names the condition that must change before
/// [`BatchRuntime::step`] can make progress.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BlockReason<C> {
    /// Not even the head request fits the retained-result bound, so the caller
    /// must consume retained results before more work can run.
    RetainedResults,
    /// The shared pool could not cover the head, and no eligible peer obtained a
    /// reservation. A pool capacity change or local change gives a reason to retry.
    ///
    /// Register with [`BatchRuntime::capacity_wait`] *before* attempting, then arm
    /// [`ReadinessWait::wake_on_change`] before parking. Retry when the registration
    /// changes; another waiter may have taken the released capacity first.
    Pool {
        /// Bytes the head request needs reserved before it can run.
        requested: u64,
        /// Bytes the pool had available when the reservation failed.
        available: u64,
    },
    /// The executor reports temporary contention. Use [`Self::wake_on_change`]
    /// to arm both its wait and pool readiness, including pool closure.
    Executor { reason: C, wait: ReadinessWait },
    /// The head was blocked and the bounded scan found additional conditions
    /// that may make later work runnable. Arm every wait before parking.
    Peers {
        head: Box<Self>,
        executor_waits: Vec<ReadinessWait>,
    },
}

impl<C> BlockReason<C> {
    /// Arm all resource waits reported by this step. `pool_wait` must have been
    /// registered through [`BatchRuntime::capacity_wait`] before stepping. The
    /// driver also arms its pre-step local wait independently.
    pub fn wake_on_change(&mut self, pool_wait: &mut ReadinessWait, waker: &std::task::Waker) {
        match self {
            Self::RetainedResults => {}
            Self::Pool { .. } => pool_wait.wake_on_change(waker),
            Self::Executor { wait, .. } => {
                pool_wait.wake_on_change(waker);
                wait.wake_on_change(waker);
            }
            Self::Peers {
                head,
                executor_waits,
            } => {
                head.wake_on_change(pool_wait, waker);
                for wait in executor_waits {
                    wait.wake_on_change(waker);
                }
            }
        }
    }
}

/// Why a request was refused before any work was submitted for it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Rejection<C> {
    /// The executor can never run this request under its own limits.
    Executor(C),
    /// This request's own retained output can never fit the shared pool, so
    /// releasing capacity can never admit it. Permanent request-local
    /// infeasibility, not backpressure.
    RetainedOutputTooLarge {
        /// Bytes the request needs reserved.
        requested: u64,
        /// Total bytes the pool can ever grant.
        capacity: u64,
    },
}

/// How one request ended.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Terminal<O, C> {
    /// The request executed. Its output stays retained until consumed.
    Output(O),
    /// The request could not execute.
    Rejected(Rejection<C>),
    /// The request was cancelled, either while it was still waiting or after it had
    /// produced output the caller no longer wants. Nothing is retained: a result that
    /// already existed went back to the executor with the charge covering it.
    Cancelled,
}

/// One terminal entry awaiting consumption, in completion order.
pub struct Completed<O, C> {
    request: RequestId,
    parameter_version: ParameterVersion,
    outcome: Terminal<JobOutput<O>, C>,
}

impl<O, C> Completed<O, C> {
    #[must_use]
    pub const fn request(&self) -> RequestId {
        self.request
    }

    #[must_use]
    pub const fn parameter_version(&self) -> ParameterVersion {
        self.parameter_version
    }

    #[must_use]
    pub const fn outcome(&self) -> &Terminal<JobOutput<O>, C> {
        &self.outcome
    }

    /// Take the terminal outcome. Any output still owns its reservation.
    #[must_use]
    pub fn into_outcome(self) -> Terminal<JobOutput<O>, C> {
        self.outcome
    }

    /// The reservation covering this entry's storage until it is consumed.
    #[must_use]
    pub fn lease(&self) -> Option<&PoolLease> {
        match &self.outcome {
            Terminal::Output(output) => Some(output.lease()),
            Terminal::Rejected(_) | Terminal::Cancelled => None,
        }
    }

    /// Identity of the allocation this entry holds, when it produced output.
    #[must_use]
    pub fn allocation(&self) -> Option<AllocationId> {
        self.lease().map(PoolLease::allocation)
    }

    /// Bytes this entry retains until the caller consumes it.
    ///
    /// Rejections and cancellations retain nothing and report zero.
    #[must_use]
    pub fn retained_bytes(&self) -> u64 {
        self.lease().map_or(0, PoolLease::bytes)
    }

    /// Whether this request was cancelled instead of finishing.
    #[must_use]
    pub const fn is_cancelled(&self) -> bool {
        matches!(self.outcome, Terminal::Cancelled)
    }

    /// The completed output, when this request executed rather than failing.
    #[must_use]
    pub fn output(&self) -> Option<&O> {
        match &self.outcome {
            Terminal::Output(output) => Some(output.output()),
            Terminal::Rejected(_) | Terminal::Cancelled => None,
        }
    }
}

/// What one [`BatchRuntime::step`] call did.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StepOutcome<C> {
    /// Admission is open, but there is no queued work.
    Idle,
    /// Admission is closed and the waiting queue is empty. Terminal entries
    /// are still deliverable; this does not establish device completion.
    AdmissionDrained,
    /// A batch executed and its results are retained until consumed.
    Executed { results: usize },
    /// Nothing executed; the reason names the wakeup condition.
    Blocked(BlockReason<C>),
    /// The head request was rejected, and its terminal entry was delivered.
    Rejected { request: RequestId },
    /// A queued request was cancelled before it ran, and its terminal entry was
    /// delivered.
    Cancelled { request: RequestId },
}

/// Coarse execution boundary for non-autoregressive request batching.
/// Concrete executors own tensor shapes, padding/ragged layout, device buffers,
/// and any model-specific batch preparation.
pub trait BatchExecutor {
    type Input;
    type Output;
    type Error: Error + Send + Sync + 'static;

    /// Why this executor cannot run a request, either now ([`BatchSelection::Blocked`])
    /// or under its own limits ([`BatchSelection::Rejected`]).
    type Constraint: Error + Send + Sync + 'static;

    fn parameter_version(&self) -> ParameterVersion;

    fn max_batch_items(&self) -> usize;

    /// Choose what the oldest compatible candidates should do next.
    ///
    /// `candidates` contains at most [`Self::max_batch_items`] requests, all
    /// pinned to the executor's current parameter version. Returning the full
    /// slice is the default. A model can select a shorter nonzero prefix when
    /// shapes, padding, workspace, or another concrete execution constraint makes
    /// the whole candidate set unsuitable.
    ///
    /// The runtime normally offers the FIFO head and its peers. If the head
    /// is temporarily blocked, it may instead offer a later candidate alone.
    fn select_batch(&self, candidates: &[&Self::Input]) -> BatchSelection<Self::Constraint>;

    /// Heap bytes the input keeps alive while queued, including allocation spare
    /// capacity and nested allocations. The request count bounds inline storage.
    /// Report a conservative upper bound; inputs must not grow after acceptance.
    fn waiting_input_bytes(&self, input: &Self::Input) -> usize;

    /// Bytes the completed output retains until its storage and lease retire.
    ///
    /// Include every host allocation and device buffer the retained output keeps
    /// alive, not just the bytes a caller is likely to read. The runtime reserves
    /// this amount before execution. Under-reporting can exceed the retained
    /// budget; measuring after execution is too late.
    fn retained_bytes(&self, input: &Self::Input) -> u64;

    /// Enqueue one prepared batch and return its results.
    ///
    /// Each job carries the reservation covering that request's storage. The
    /// executor owns those reservations from this call on: it attaches each charge to
    /// the storage it materializes and returns the two together in
    /// [`JobOutput`].
    ///
    /// # Errors
    /// The returned [`EnqueueError`] must distinguish a refusal that never touched the
    /// device from work that may already be in flight. After a device-visible
    /// mutation the executor keeps ownership of everything it may have enqueued —
    /// storage and charge together — and releases only what provably never reached
    /// the device. Returning an error is not permission to release uncertain storage.
    fn execute(
        &mut self,
        batch: Vec<Job<Self::Input>>,
    ) -> Result<Vec<JobOutput<Self::Output>>, EnqueueError<Self::Error>>;

    /// Take back output the runtime cannot deliver or that the caller abandons.
    ///
    /// The executor owns the
    /// storage and the charge carried by every returned value again; it must keep
    /// both until completion is established, exactly as after a failed enqueue.
    /// Dropping them here would release device memory the device may still write.
    fn retire(&mut self, outputs: Vec<JobOutput<Self::Output>>);
}

/// Opt-in shutdown for an execution owner. Already delivered outputs are
/// outside the executor's shutdown obligation.
pub trait BatchExecutorShutdown: BatchExecutor {
    /// Permanently close the executor and retire work it still owns.
    ///
    /// # Errors
    /// Failure retains uncertain work and reservations for retry. Success
    /// acknowledges completion of executor-owned retirement, not of work
    /// already handed to a consumer.
    fn shutdown(&mut self) -> Result<(), Self::Error>;
}

/// Why a batch could not be enqueued, and what the executor still owns.
///
/// The distinction is the point: it tells the caller whether device work may exist for
/// a batch whose results will never be delivered.
#[derive(Debug)]
pub enum EnqueueError<E> {
    /// Nothing reached the device. Every reservation the executor was handed is
    /// released and nothing needs retiring.
    Refused(E),
    /// A device-visible mutation happened before the failure, so some work may be in
    /// flight. The executor retains the storage and the reservations for everything
    /// it may have enqueued; only a proven completion may release them.
    Uncertain(E),
}

impl<E> EnqueueError<E> {
    /// The executor's own error, whichever ownership state it left behind.
    #[must_use]
    pub fn into_inner(self) -> E {
        match self {
            Self::Refused(error) | Self::Uncertain(error) => error,
        }
    }
}

// Local policy, not a cross-runtime fairness or wall-clock guarantee.
const MAX_BYPASS_SCAN: usize = 8;
const MAX_BYPASSES: u8 = 8;

struct Queued<I> {
    request: RequestId,
    input: I,
    waiting_bytes: usize,
    parameter_version: ParameterVersion,
    /// Cancellation intent, settled before runnable selection when delivery has room.
    cancelled: bool,
    bypasses: u8,
}

/// What the queue resolved to before any execution or reservation happened.
///
/// `Ready` stays separate from `Blocked` so the capacity checks that follow can
/// block on the runtime's own retained budget without confusing it with an
/// executor-reported constraint.
enum Resolved<C> {
    Idle,
    Blocked(BlockReason<C>),
    Rejected {
        request: RequestId,
    },
    Cancelled {
        request: RequestId,
    },
    /// Per-request retention reservations for the executor's selected prefix.
    Ready {
        items: usize,
    },
}

pub struct BatchRuntime<E: BatchExecutor> {
    executor: E,
    pool: Arc<BytePool>,
    config: BatchConfig,
    queue: VecDeque<Queued<E::Input>>,
    waiting_input_bytes: usize,
    terminal: VecDeque<Completed<E::Output, E::Constraint>>,
    local_readiness: Readiness,
    admission_closed: bool,
    shutdown_complete: bool,
}

impl<E: BatchExecutor> BatchRuntime<E> {
    /// Create a runtime that reserves retained output from `pool`.
    ///
    /// The pool is shared, not owned: sibling runtimes drawing on one physical
    /// pool pass the same authority, which is what makes a single constraint bind
    /// all of them.
    ///
    /// # Errors
    /// Rejects a zero bound or an executor that cannot run one item.
    pub fn new(
        executor: E,
        pool: Arc<BytePool>,
        config: BatchConfig,
    ) -> Result<Self, RuntimeError<E::Error>> {
        if config.max_waiting_requests == 0
            || config.max_waiting_input_bytes == 0
            || config.max_retained_results == 0
            || executor.max_batch_items() == 0
        {
            return Err(RuntimeError::InvalidConfig);
        }
        Ok(Self {
            executor,
            pool,
            config,
            queue: VecDeque::with_capacity(config.max_waiting_requests),
            waiting_input_bytes: 0,
            terminal: VecDeque::new(),
            local_readiness: Readiness::default(),
            admission_closed: false,
            shutdown_complete: false,
        })
    }

    /// Permanently stop accepting new inputs on this runtime. Requests already
    /// accepted can still run or settle cancellation, and retained results stay
    /// available. This does not close the shared pool or synchronize the executor.
    pub fn close_admission(&mut self) {
        if !self.admission_closed {
            self.admission_closed = true;
            self.publish_local_change();
        }
    }

    /// Close admission and discard all work still owned by this runtime. A
    /// caller that needs terminal outcomes should use [`Self::close_admission`]
    /// and continue driving instead. Results already handed out are untouched.
    /// Undelivered outputs go back to the executor with their leases. This does
    /// not close the shared pool or establish executor shutdown.
    pub fn abandon(&mut self) {
        let changed = !self.admission_closed || !self.queue.is_empty() || !self.terminal.is_empty();
        if !changed {
            return;
        }
        self.admission_closed = true;
        while let Some(input) = self.remove_queued(0) {
            drop(input);
        }
        let outputs: Vec<_> = self
            .terminal
            .drain(..)
            .filter_map(|entry| match entry.into_outcome() {
                Terminal::Output(output) => Some(output),
                Terminal::Cancelled | Terminal::Rejected(_) => None,
            })
            .collect();
        if !outputs.is_empty() {
            self.executor.retire(outputs);
        }
        self.publish_local_change();
    }

    /// # Errors
    /// Rejects closed admission or pool, an over-budget input, a full waiting
    /// queue or identity exhaustion.
    pub fn submit(&mut self, input: E::Input) -> Result<RequestId, RuntimeError<E::Error>> {
        if self.admission_closed {
            return Err(RuntimeError::AdmissionClosed);
        }
        if self.pool.is_closed() {
            return Err(RuntimeError::PoolClosed);
        }
        if self.queue.len() >= self.config.max_waiting_requests {
            return Err(RuntimeError::WaitingCapacityExhausted);
        }
        let waiting_bytes = self.executor.waiting_input_bytes(&input);
        let limit = self.config.max_waiting_input_bytes;
        if waiting_bytes > limit {
            return Err(RuntimeError::WaitingInputTooLarge {
                requested: waiting_bytes,
                limit,
            });
        }
        let available = limit - self.waiting_input_bytes;
        if waiting_bytes > available {
            return Err(RuntimeError::WaitingInputCapacityExhausted {
                requested: waiting_bytes,
                available,
            });
        }
        let request = RequestId(
            NEXT_REQUEST_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| RuntimeError::IdentityExhausted)?,
        );
        self.queue.push_back(Queued {
            request,
            input,
            waiting_bytes,
            parameter_version: self.executor.parameter_version(),
            cancelled: false,
            bypasses: 0,
        });
        self.waiting_input_bytes += waiting_bytes;
        self.publish_local_change();
        Ok(request)
    }

    fn publish_local_change(&self) {
        // A notifier must not prevent returning an accepted identity or a
        // dequeued output and its lease. The publisher notifies other waiters
        // before rethrowing, so a broken transport cannot strand them either.
        let readiness = &self.local_readiness;
        let _ = std::panic::catch_unwind(|| readiness.publish());
    }

    fn remove_queued(&mut self, index: usize) -> Option<Queued<E::Input>> {
        let entry = self.queue.remove(index)?;
        self.waiting_input_bytes -= entry.waiting_bytes;
        Some(entry)
    }

    /// Cancel one live request.
    ///
    /// Cancellation is intent, not device completion. A request that is still waiting
    /// reports cancellation instead of running, which releases nothing because nothing
    /// was reserved for it. A request whose result is already retained gives that
    /// result back to the executor, together with the charge covering it, and reports
    /// cancellation instead; the runtime never releases device storage or its charge on
    /// its own. Either outcome is delivered through the ordinary terminal queue, so a
    /// caller does not need a second path to learn a cancellation happened.
    pub fn cancel(&mut self, request: RequestId) -> CancelOutcome {
        if let Some(queued) = self
            .queue
            .iter_mut()
            .find(|queued| queued.request == request)
        {
            if !queued.cancelled {
                queued.cancelled = true;
                self.publish_local_change();
            }
            return CancelOutcome::Queued;
        }
        let Some(index) = self
            .terminal
            .iter()
            .position(|entry| entry.request() == request)
        else {
            return CancelOutcome::Unknown;
        };
        let Some(entry) = self.terminal.remove(index) else {
            return CancelOutcome::Unknown;
        };
        let parameter_version = entry.parameter_version();
        if let Terminal::Output(output) = entry.into_outcome() {
            self.executor.retire(vec![output]);
        }
        self.terminal.push_back(Completed {
            request,
            parameter_version,
            outcome: Terminal::Cancelled,
        });
        CancelOutcome::Discarded
    }

    /// Settle the first queued cancellation, or select and execute at most one batch.
    ///
    /// A [`StepOutcome::Blocked`] result is a wakeup contract rather than a
    /// failure. Arm the pre-step [`Self::local_wait`] when idle or blocked.
    /// For a block, also arm [`BlockReason::wake_on_change`] with the pre-step
    /// [`Self::capacity_wait`]. A bypass scan considers at most eight following
    /// requests, and at most eight submissions may overtake the same head.
    ///
    /// # Errors
    /// Rejects a parameter-version change while queued work still targets the
    /// previous version, a malformed executor batch selection, malformed executor
    /// output, an internal queue invariant failure, or an executor failure.
    pub fn step(&mut self) -> Result<StepOutcome<E::Constraint>, RuntimeError<E::Error>> {
        // Terminal entries all occupy delivery capacity until consumed.
        let free_slots = self
            .config
            .max_retained_results
            .saturating_sub(self.terminal.len());
        let accepted = match self.resolve_queue()? {
            Resolved::Idle if self.admission_closed => return Ok(StepOutcome::AdmissionDrained),
            Resolved::Idle => return Ok(StepOutcome::Idle),
            Resolved::Rejected { request } => return Ok(StepOutcome::Rejected { request }),
            Resolved::Cancelled { request } => return Ok(StepOutcome::Cancelled { request }),
            Resolved::Blocked(_) | Resolved::Ready { .. } if free_slots == 0 => {
                return Ok(StepOutcome::Blocked(BlockReason::RetainedResults));
            }
            Resolved::Blocked(reason @ BlockReason::Executor { .. }) => {
                if self.pool.is_closed() {
                    return Err(RuntimeError::PoolClosed);
                }
                return self.try_bypass(
                    reason,
                    self.pool.available(),
                    self.executor.parameter_version(),
                );
            }
            Resolved::Blocked(reason) => return Ok(StepOutcome::Blocked(reason)),
            Resolved::Ready { items } => items.min(free_slots),
        };
        let version = self.executor.parameter_version();

        // Reserve every retained output before submitting work, so a full pool
        // stops execution instead of being discovered after the fact. Batching is
        // an optimization, so the accepted range shrinks to the largest prefix the
        // pool grants rather than blocking on the whole selection.
        let mut leases = Vec::with_capacity(accepted);
        for index in 0..accepted {
            let bytes = self.executor.retained_bytes(&self.queue[index].input);
            match self.pool.reserve(bytes) {
                Ok(lease) => leases.push(lease),
                Err(ReserveError::Exhausted {
                    requested,
                    available,
                }) if leases.is_empty() => {
                    return self.try_bypass(
                        BlockReason::Pool {
                            requested,
                            available,
                        },
                        available,
                        version,
                    );
                }
                // A later oversized item stops the prefix here and becomes the
                // queue head's own rejection once earlier work drains.
                Err(ReserveError::Exhausted { .. } | ReserveError::TooLarge { .. }) => break,
                Err(ReserveError::Closed) => return Err(RuntimeError::PoolClosed),
            }
        }
        if leases.is_empty() {
            return Err(RuntimeError::QueueInvariant);
        }

        self.execute_granted(0, leases, version)
    }

    fn execute_granted(
        &mut self,
        start: usize,
        leases: Vec<PoolLease>,
        version: ParameterVersion,
    ) -> Result<StepOutcome<E::Constraint>, RuntimeError<E::Error>> {
        let items = leases.len();
        // The reservation granted for each accepted request travels into its job, so
        // an executor that fails after reaching the device owns that charge along with
        // the storage it covers instead of handing it back inside an error.
        let executed = self.execute_selected(start, items, leases)?;
        let results = executed.len();
        self.terminal
            .extend(executed.into_iter().map(|output| Completed {
                request: output.request(),
                parameter_version: version,
                outcome: Terminal::Output(output),
            }));
        Ok(StepOutcome::Executed { results })
    }

    /// Submit at most one feasible peer with its own lease. On no progress,
    /// carry each relevant pre-check wait back to the driver.
    fn try_bypass(
        &mut self,
        head: BlockReason<E::Constraint>,
        available: u64,
        version: ParameterVersion,
    ) -> Result<StepOutcome<E::Constraint>, RuntimeError<E::Error>> {
        if self.queue[0].bypasses >= MAX_BYPASSES {
            return Ok(StepOutcome::Blocked(head));
        }
        let mut executor_waits = Vec::new();
        for (index, queued) in self.queue.iter().enumerate().skip(1).take(MAX_BYPASS_SCAN) {
            if queued.parameter_version != version {
                break;
            }
            let bytes = self.executor.retained_bytes(&queued.input);
            if bytes > self.pool.capacity() {
                continue;
            }
            match self.executor.select_batch(&[&queued.input]) {
                BatchSelection::Ready { items: 1 } => {}
                BatchSelection::Ready { items } => {
                    return Err(RuntimeError::MalformedBatchSelection {
                        selected: items,
                        candidates: 1,
                    });
                }
                BatchSelection::Blocked { wait, .. } => {
                    let head_watches_it = matches!(&head, BlockReason::Executor { wait: current, .. } if *current == wait);
                    if !head_watches_it && !executor_waits.contains(&wait) {
                        executor_waits.push(wait);
                    }
                    continue;
                }
                BatchSelection::Rejected(_) => continue,
            }
            if bytes > available {
                continue;
            }
            match self.pool.reserve(bytes) {
                Ok(lease) => {
                    self.queue[0].bypasses += 1;
                    return self.execute_granted(index, vec![lease], version);
                }
                Err(ReserveError::Exhausted { .. }) => {}
                Err(ReserveError::TooLarge { .. }) => return Err(RuntimeError::QueueInvariant),
                Err(ReserveError::Closed) => return Err(RuntimeError::PoolClosed),
            }
        }
        let reason = if executor_waits.is_empty() {
            head
        } else {
            BlockReason::Peers {
                head: Box::new(head),
                executor_waits,
            }
        };
        Ok(StepOutcome::Blocked(reason))
    }

    /// Resolve cancellation or the runnable head without executing anything.
    fn resolve_queue(&mut self) -> Result<Resolved<E::Constraint>, RuntimeError<E::Error>> {
        // Cancellation is independent of runnable head pressure or executor state.
        if let Some(index) = self.queue.iter().position(|queued| queued.cancelled) {
            if self.terminal.len() >= self.config.max_retained_results {
                // The terminal count bound applies to cancellations exactly as it does
                // to rejections: the intent is already recorded, so consuming one entry
                // delivers it.
                return Ok(Resolved::Blocked(BlockReason::RetainedResults));
            }
            let Some(cancelled) = self.remove_queued(index) else {
                return Err(RuntimeError::QueueInvariant);
            };
            self.terminal.push_back(Completed {
                request: cancelled.request,
                parameter_version: cancelled.parameter_version,
                outcome: Terminal::Cancelled,
            });
            return Ok(Resolved::Cancelled {
                request: cancelled.request,
            });
        }
        let version = self.executor.parameter_version();
        let head_retention = match self.queue.front() {
            None => return Ok(Resolved::Idle),
            Some(front) => {
                if front.parameter_version != version {
                    return Err(RuntimeError::ParameterVersionChanged {
                        queued: front.parameter_version,
                        current: version,
                    });
                }
                self.executor.retained_bytes(&front.input)
            }
        };
        // Releasing capacity cannot change the pool's total capacity, so a head
        // whose own output can never fit is rejected rather than parked.
        if head_retention > self.pool.capacity() {
            return self.reject_head(Rejection::RetainedOutputTooLarge {
                requested: head_retention,
                capacity: self.pool.capacity(),
            });
        }

        let candidates = self
            .queue
            .iter()
            .take(self.executor.max_batch_items())
            .take_while(|queued| queued.parameter_version == version)
            .map(|queued| &queued.input)
            .collect::<Vec<_>>();
        match self.executor.select_batch(&candidates) {
            BatchSelection::Blocked { reason, wait } => {
                Ok(Resolved::Blocked(BlockReason::Executor { reason, wait }))
            }
            BatchSelection::Rejected(constraint) => {
                self.reject_head(Rejection::Executor(constraint))
            }
            BatchSelection::Ready { items } => {
                if items == 0 || items > candidates.len() {
                    return Err(RuntimeError::MalformedBatchSelection {
                        selected: items,
                        candidates: candidates.len(),
                    });
                }
                Ok(Resolved::Ready { items })
            }
        }
    }

    /// Deliver the head's rejection now, or block while no terminal slot is free.
    fn reject_head(
        &mut self,
        rejection: Rejection<E::Constraint>,
    ) -> Result<Resolved<E::Constraint>, RuntimeError<E::Error>> {
        if self.terminal.len() >= self.config.max_retained_results {
            return Ok(Resolved::Blocked(BlockReason::RetainedResults));
        }
        let Some(rejected) = self.remove_queued(0) else {
            return Err(RuntimeError::QueueInvariant);
        };
        self.terminal.push_back(Completed {
            request: rejected.request,
            parameter_version: rejected.parameter_version,
            outcome: Terminal::Rejected(rejection),
        });
        Ok(Resolved::Rejected {
            request: rejected.request,
        })
    }

    /// Remove a FIFO prefix or a single bypassed request starting at `start`.
    ///
    /// `leases` are the reservations made for those items, one per queued request in
    /// the same order. They travel into the jobs, so the executor owns every
    /// reservation it was handed from the moment it is called: a failure after a
    /// device-visible mutation keeps the charge with the storage rather than
    /// returning it here.
    fn execute_selected(
        &mut self,
        start: usize,
        items: usize,
        leases: Vec<PoolLease>,
    ) -> Result<Vec<JobOutput<E::Output>>, RuntimeError<E::Error>> {
        let mut queued = Vec::with_capacity(items);
        while queued.len() < items {
            let Some(next) = self.remove_queued(start) else {
                return Err(RuntimeError::QueueInvariant);
            };
            queued.push(next);
        }
        let expected = queued.iter().map(|item| item.request).collect::<Vec<_>>();
        let jobs = queued
            .into_iter()
            .zip(leases)
            .map(|(item, lease)| Job {
                request: item.request,
                input: item.input,
                lease,
            })
            .collect();
        let outputs = match self.executor.execute(jobs) {
            Ok(outputs) => outputs,
            Err(EnqueueError::Refused(source)) => {
                return Err(RuntimeError::Executor {
                    requests: expected,
                    source,
                });
            }
            // The executor keeps the storage and the charge; the runtime reports
            // which requests it will never deliver, and retains nothing itself.
            Err(EnqueueError::Uncertain(source)) => {
                return Err(RuntimeError::Uncertain {
                    requests: expected,
                    source,
                });
            }
        };
        if outputs.len() != expected.len()
            || outputs
                .iter()
                .zip(expected.iter())
                .any(|(output, request)| output.request() != *request)
        {
            let returned = outputs.len();
            // Nothing here can be committed, and the storage behind each output may
            // still be live: hand it back to the owner that can wait for completion
            // instead of dropping it into a free list.
            self.executor.retire(outputs);
            return Err(RuntimeError::MalformedCompletion {
                requests: expected,
                returned,
            });
        }
        Ok(outputs)
    }

    /// Consume the oldest terminal entry and hand over its reservation.
    ///
    /// Dropping the returned entry after popping discards the output and retires
    /// its charge. Handing the entry to another owner transfers the charge with it,
    /// because the storage it covers travels too.
    #[must_use]
    pub fn pop_completed(&mut self) -> Option<Completed<E::Output, E::Constraint>> {
        let entry = self.terminal.pop_front()?;
        self.publish_local_change();
        Some(entry)
    }

    /// Register before stepping. Successful submission, new queued cancellation,
    /// terminal dequeue, admission closure and abandonment publish after updating state.
    /// This does not watch the byte pool or executor; arm those waits too when
    /// their resources block the step. Use a waker that schedules the driver
    /// without synchronously reentering the caller's lock. An unwinding callback
    /// does not undo the mutation or prevent returning the request or output.
    #[must_use]
    pub fn local_wait(&self) -> ReadinessWait {
        self.local_readiness.register()
    }

    #[must_use]
    pub fn queued(&self) -> usize {
        self.queue.len()
    }

    /// Terminal entries the caller has not consumed, which is what
    /// `max_retained_results` bounds: a rejection occupies the budget until it is
    /// consumed exactly like an output does.
    #[must_use]
    pub fn retained_results(&self) -> usize {
        self.terminal.len()
    }

    /// Register before stepping and pass this registration to
    /// [`BlockReason::wake_on_change`] if blocked. A release or closure notifies
    /// the caller even if it preceded arming.
    /// A changed registration permits retrying [`Self::step`], not a reservation.
    #[must_use]
    pub fn capacity_wait(&self) -> ReadinessWait {
        self.pool.capacity_wait()
    }

    /// Bytes this runtime's retained entries currently hold in the shared pool.
    ///
    /// A sibling runtime's leases are not counted here; read [`Self::pool`] for the
    /// authority's own totals.
    #[must_use]
    pub fn retained_output_bytes(&self) -> u64 {
        self.terminal
            .iter()
            .filter_map(Completed::lease)
            .fold(0_u64, |total, lease| total.saturating_add(lease.bytes()))
    }

    /// The shared byte authority this runtime reserves from.
    #[must_use]
    pub fn pool(&self) -> &Arc<BytePool> {
        &self.pool
    }

    #[must_use]
    pub const fn config(&self) -> BatchConfig {
        self.config
    }

    #[must_use]
    pub fn executor(&self) -> &E {
        &self.executor
    }

    #[must_use]
    pub fn executor_mut(&mut self) -> &mut E {
        &mut self.executor
    }
}

/// How one [`BatchRuntime::cancel`] ended.
impl<E: BatchExecutorShutdown> BatchRuntime<E> {
    /// Abandon local delivery and shut down the same executor. Once successful
    /// subsequent calls are no-ops. Does not close the shared pool.
    ///
    /// # Errors
    /// Returns the executor's failure. Retry on this same runtime; neither
    /// admission nor delivery reopens.
    pub fn shutdown(&mut self) -> Result<(), E::Error> {
        if !self.shutdown_complete {
            self.abandon();
            self.executor.shutdown()?;
            self.shutdown_complete = true;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancelOutcome {
    /// The request was still waiting; it will report cancellation rather than run.
    /// No reservation existed for it, so nothing was released.
    Queued,
    /// The request had a retained result. That result and the charge covering it went
    /// back to the executor, and the entry now reports cancellation.
    Discarded,
    /// No live request has this identity: it was never submitted, was already
    /// consumed, or was cancelled earlier.
    Unknown,
}

#[derive(Debug)]
pub enum RuntimeError<E> {
    InvalidConfig,
    WaitingCapacityExhausted,
    WaitingInputTooLarge {
        requested: usize,
        limit: usize,
    },
    WaitingInputCapacityExhausted {
        requested: usize,
        available: usize,
    },
    IdentityExhausted,
    ParameterVersionChanged {
        queued: ParameterVersion,
        current: ParameterVersion,
    },
    MalformedBatchSelection {
        selected: usize,
        candidates: usize,
    },
    QueueInvariant,
    /// The executor returned output that does not match the work it was given, so
    /// nothing can be committed. Its storage and charge were handed back through
    /// [`BatchExecutor::retire`].
    MalformedCompletion {
        /// Requests this call submitted.
        requests: Vec<RequestId>,
        /// Results the executor returned for them.
        returned: usize,
    },
    /// The executor refused the batch before any device access, so no result will be
    /// delivered and nothing is retained.
    Executor {
        /// Requests this call submitted.
        requests: Vec<RequestId>,
        source: E,
    },
    /// The executor failed after a device-visible mutation, so some work may be in
    /// flight. The executor retains ownership of everything it may have enqueued,
    /// together with those reservations; the runtime delivers nothing for these
    /// requests. Release waits on a completion only the executor can establish.
    Uncertain {
        /// Requests this call submitted.
        requests: Vec<RequestId>,
        source: E,
    },
    /// This runtime no longer accepts new submissions.
    AdmissionClosed,
    /// The shared byte pool is closed, so no new submission is accepted.
    PoolClosed,
}

impl<E: fmt::Display> fmt::Display for RuntimeError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => f.write_str("batch runtime configuration is invalid"),
            Self::WaitingCapacityExhausted => {
                f.write_str("batch runtime waiting capacity is exhausted")
            }
            Self::WaitingInputTooLarge { requested, limit } => write!(
                f,
                "batch input retains {requested} bytes, exceeding the waiting input limit of {limit}"
            ),
            Self::WaitingInputCapacityExhausted {
                requested,
                available,
            } => write!(
                f,
                "batch input retains {requested} bytes, but only {available} waiting input bytes are available"
            ),
            Self::IdentityExhausted => {
                f.write_str("batch runtime request identity space is exhausted")
            }
            Self::ParameterVersionChanged { queued, current } => write!(
                f,
                "queued request targets parameter version {}, but executor now exposes version {}",
                queued.get(),
                current.get()
            ),
            Self::MalformedBatchSelection {
                selected,
                candidates,
            } => write!(
                f,
                "batch executor selected {selected} requests from {candidates} candidates"
            ),
            Self::QueueInvariant => {
                f.write_str("batch runtime queue changed while draining a selected batch")
            }
            Self::MalformedCompletion { requests, returned } => write!(
                f,
                "batch executor returned {returned} results for {} submitted requests",
                requests.len()
            ),
            Self::Executor { source, .. } => source.fmt(f),
            Self::Uncertain { source, .. } => write!(
                f,
                "batch executor failed after reaching the device and retains the work: {source}"
            ),
            Self::AdmissionClosed => f.write_str("batch runtime admission is closed"),
            Self::PoolClosed => f.write_str("shared byte pool is closed"),
        }
    }
}

impl<E: Error + 'static> Error for RuntimeError<E> {}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::mpsc;
    use std::task::{Wake, Waker};

    use super::*;

    fn wake_channel() -> (Waker, mpsc::Receiver<()>) {
        struct Notify(mpsc::Sender<()>);
        impl Wake for Notify {
            fn wake(self: Arc<Self>) {
                self.0.send(()).unwrap();
            }
        }
        let (tx, rx) = mpsc::channel();
        (Waker::from(Arc::new(Notify(tx))), rx)
    }

    #[derive(Debug, Eq, PartialEq)]
    enum Constraint {
        WorkspaceBusy,
        ExceedsTokenBudget { tokens: usize },
    }

    impl fmt::Display for Constraint {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::WorkspaceBusy => f.write_str("encoder workspace is busy"),
                Self::ExceedsTokenBudget { tokens } => {
                    write!(f, "request of {tokens} tokens exceeds the encoder budget")
                }
            }
        }
    }

    impl Error for Constraint {}

    struct Encoder {
        version: ParameterVersion,
        batches: Vec<Vec<usize>>,
        token_budget: Option<usize>,
        busy: bool,
        workspace_readiness: ribn_foundation::Readiness,
        bytes_per_token: u64,
        over_select: bool,
        /// Return fewer results than the runtime submitted.
        short_return: bool,
        /// Outputs handed back by the runtime after a malformed completion.
        retired_outputs: usize,
        max_batch_items: usize,
        rejected_marker: Option<u32>,
        blocked_heads: [Option<u32>; 2],
        head_readiness: [Readiness; 2],
    }

    impl Encoder {
        fn new() -> Self {
            Self {
                version: ParameterVersion::new(7),
                batches: Vec::new(),
                token_budget: None,
                busy: false,
                workspace_readiness: ribn_foundation::Readiness::default(),
                bytes_per_token: 1,
                over_select: false,
                short_return: false,
                retired_outputs: 0,
                max_batch_items: 2,
                rejected_marker: None,
                blocked_heads: [None; 2],
                head_readiness: std::array::from_fn(|_| Readiness::default()),
            }
        }
    }

    impl BatchExecutor for Encoder {
        type Input = Vec<u32>;
        type Output = u32;
        type Error = Infallible;
        type Constraint = Constraint;

        fn parameter_version(&self) -> ParameterVersion {
            self.version
        }

        fn max_batch_items(&self) -> usize {
            self.max_batch_items
        }

        fn select_batch(&self, candidates: &[&Self::Input]) -> BatchSelection<Self::Constraint> {
            let head = candidates
                .first()
                .expect("the runtime never selects from an empty candidate set");
            if self.over_select {
                return BatchSelection::Ready {
                    items: candidates.len() + 1,
                };
            }
            if self.token_budget.is_some_and(|budget| head.len() > budget)
                || self
                    .rejected_marker
                    .is_some_and(|marker| head.first() == Some(&marker))
            {
                return BatchSelection::Rejected(Constraint::ExceedsTokenBudget {
                    tokens: head.len(),
                });
            }
            let wait = self.workspace_readiness.register();
            if self.busy {
                return BatchSelection::Blocked {
                    reason: Constraint::WorkspaceBusy,
                    wait,
                };
            }
            for (marker, readiness) in self.blocked_heads.iter().zip(&self.head_readiness) {
                if marker.as_ref().is_some_and(|m| head.first() == Some(m)) {
                    return BatchSelection::Blocked {
                        reason: Constraint::WorkspaceBusy,
                        wait: readiness.register(),
                    };
                }
            }
            let mut items = 0_usize;
            let mut tokens = 0_usize;
            for candidate in candidates {
                if candidate
                    .first()
                    .is_some_and(|m| self.blocked_heads.contains(&Some(*m)))
                {
                    break;
                }
                let next = tokens.saturating_add(candidate.len());
                if items > 0 && self.token_budget.is_some_and(|budget| next > budget) {
                    break;
                }
                tokens = next;
                items += 1;
            }
            BatchSelection::Ready { items }
        }

        fn waiting_input_bytes(&self, input: &Self::Input) -> usize {
            input.capacity() * size_of::<u32>()
        }

        fn retained_bytes(&self, input: &Self::Input) -> u64 {
            u64::try_from(input.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(self.bytes_per_token)
        }

        fn execute(
            &mut self,
            batch: Vec<Job<Self::Input>>,
        ) -> Result<Vec<JobOutput<Self::Output>>, EnqueueError<Self::Error>> {
            self.batches
                .push(batch.iter().map(|job| job.input().len()).collect());
            let mut outputs = batch
                .into_iter()
                .map(|job| {
                    let (request, input, lease) = job.into_parts();
                    JobOutput::new(request, input.into_iter().sum(), lease)
                })
                .collect::<Vec<_>>();
            if self.short_return {
                // A contract violation: one submitted request gets no result.
                outputs.pop();
            }
            Ok(outputs)
        }

        fn retire(&mut self, outputs: Vec<JobOutput<Self::Output>>) {
            // This fixture's output owns no storage; the hand-back is what the test
            // observes.
            self.retired_outputs += outputs.len();
        }
    }

    fn pool() -> Arc<BytePool> {
        BytePool::new(1 << 20).shared()
    }

    /// An executor that fails after the runtime has reserved for it.
    ///
    /// Its output owns no storage, so the reservations it takes back are the only
    /// resource at stake: the test watches when they return to the pool.
    struct FailingEncoder {
        /// Whether the failure happened after device-visible work.
        after_device: bool,
        /// Reservations the executor still owns after its failure.
        quarantined: Vec<PoolLease>,
        retired_outputs: Vec<JobOutput<u32>>,
    }

    impl FailingEncoder {
        fn new(after_device: bool) -> Self {
            Self {
                after_device,
                quarantined: Vec::new(),
                retired_outputs: Vec::new(),
            }
        }

        fn release_quarantined(&mut self) {
            self.quarantined.clear();
            self.retired_outputs.clear();
        }
    }

    impl BatchExecutor for FailingEncoder {
        type Input = Vec<u32>;
        type Output = u32;
        type Error = Constraint;
        type Constraint = Constraint;

        fn parameter_version(&self) -> ParameterVersion {
            ParameterVersion::new(1)
        }

        fn max_batch_items(&self) -> usize {
            1
        }

        fn select_batch(&self, _candidates: &[&Self::Input]) -> BatchSelection<Self::Constraint> {
            BatchSelection::Ready { items: 1 }
        }

        fn waiting_input_bytes(&self, input: &Self::Input) -> usize {
            input.capacity() * size_of::<u32>()
        }

        fn retained_bytes(&self, _input: &Self::Input) -> u64 {
            1
        }

        fn execute(
            &mut self,
            batch: Vec<Job<Self::Input>>,
        ) -> Result<Vec<JobOutput<Self::Output>>, EnqueueError<Self::Error>> {
            if self.after_device {
                // The reservation stays here: the device may still hold storage this
                // failure created.
                self.quarantined
                    .extend(batch.into_iter().map(|job| job.into_parts().2));
                Err(EnqueueError::Uncertain(Constraint::WorkspaceBusy))
            } else {
                // Nothing reached the device, so every reservation is released.
                drop(batch);
                Err(EnqueueError::Refused(Constraint::WorkspaceBusy))
            }
        }

        fn retire(&mut self, outputs: Vec<JobOutput<Self::Output>>) {
            self.retired_outputs.extend(outputs);
        }
    }

    fn config() -> BatchConfig {
        BatchConfig {
            max_waiting_requests: 8,
            max_retained_results: 4,
            ..BatchConfig::default()
        }
    }

    fn runtime() -> BatchRuntime<Encoder> {
        BatchRuntime::new(Encoder::new(), pool(), config()).expect("runtime")
    }

    #[test]
    fn encoder_batches_without_token_or_sequence_semantics() {
        let mut runtime = runtime();
        let first = runtime.submit(vec![1, 2]).expect("first");
        let second = runtime.submit(vec![3, 4, 5]).expect("second");
        let third = runtime.submit(vec![6, 7]).expect("third");
        assert_eq!(
            runtime.step().expect("step"),
            StepOutcome::Executed { results: 2 }
        );
        assert_eq!(runtime.queued(), 1);
        assert_eq!(runtime.executor().batches, vec![vec![2, 3]]);
        let first_output = runtime.pop_completed().expect("first output");
        let second_output = runtime.pop_completed().expect("second output");
        assert_eq!(first_output.request(), first);
        assert_eq!(*first_output.output().expect("first output"), 3);
        assert_eq!(first_output.parameter_version(), ParameterVersion::new(7));
        assert_eq!(second_output.request(), second);
        assert_eq!(*second_output.output().expect("second output"), 12);
        assert_eq!(
            runtime.step().expect("second step"),
            StepOutcome::Executed { results: 1 }
        );
        assert_eq!(
            runtime.pop_completed().expect("third output").request(),
            third
        );
    }

    #[test]
    fn queued_work_is_pinned_to_a_parameter_version() {
        let mut runtime = runtime();
        runtime.submit(vec![1]).expect("request");
        runtime.executor_mut().version = ParameterVersion::new(8);
        assert!(matches!(
            runtime.step(),
            Err(RuntimeError::ParameterVersionChanged {
                queued,
                current
            }) if queued == ParameterVersion::new(7) && current == ParameterVersion::new(8)
        ));
    }

    #[test]
    fn retained_results_are_bounded_when_the_caller_stops_draining() {
        let mut runtime = BatchRuntime::new(
            Encoder::new(),
            pool(),
            BatchConfig {
                max_retained_results: 2,
                ..config()
            },
        )
        .expect("runtime");
        for index in 0..4_u32 {
            runtime.submit(vec![index]).expect("request");
        }
        assert_eq!(
            runtime.step().expect("first step"),
            StepOutcome::Executed { results: 2 }
        );
        assert_eq!(runtime.retained_results(), 2);
        assert_eq!(runtime.queued(), 2);

        assert_eq!(
            runtime.step().expect("blocked step"),
            StepOutcome::Blocked(BlockReason::RetainedResults)
        );
        assert_eq!(runtime.queued(), 2, "blocked work stays queued");
        assert_eq!(
            runtime.executor().batches.len(),
            1,
            "blocked admission must not execute"
        );

        runtime.pop_completed().expect("release one result");
        assert_eq!(runtime.retained_results(), 1);
        assert_eq!(
            runtime.step().expect("partially drained step"),
            StepOutcome::Executed { results: 1 },
            "the selected batch shrinks to the capacity that is free"
        );
        assert_eq!(runtime.retained_results(), 2);
        assert_eq!(runtime.queued(), 1);
        assert_eq!(
            runtime.step().expect("fully retained step"),
            StepOutcome::Blocked(BlockReason::RetainedResults)
        );
        runtime.pop_completed().expect("release one result");
        runtime.pop_completed().expect("release the other result");
        assert_eq!(runtime.retained_results(), 0);
        assert_eq!(
            runtime.step().expect("step after release"),
            StepOutcome::Executed { results: 1 }
        );
    }

    #[test]
    fn retained_output_bytes_are_reserved_before_execution() {
        let mut executor = Encoder::new();
        executor.bytes_per_token = 1 << 19;
        let pool = BytePool::new(1 << 19).shared();
        let mut runtime =
            BatchRuntime::new(executor, Arc::clone(&pool), config()).expect("runtime");
        runtime.submit(vec![1]).expect("first request");
        runtime.submit(vec![2]).expect("second request");

        assert_eq!(
            runtime.step().expect("first step"),
            StepOutcome::Executed { results: 1 },
            "the executor wanted both requests, but only one reservation fits"
        );
        assert_eq!(runtime.retained_output_bytes(), 1 << 19);
        assert_eq!(pool.granted(), 1 << 19);
        assert_eq!(
            runtime.step().expect("pool-blocked step"),
            StepOutcome::Blocked(BlockReason::Pool {
                requested: 1 << 19,
                available: 0
            })
        );
        assert_eq!(runtime.executor().batches.len(), 1);

        // Consuming the entry hands the charge over rather than releasing it:
        // the caller's storage still occupies those bytes.
        let Terminal::Output(output) = runtime
            .pop_completed()
            .expect("released entry")
            .into_outcome()
        else {
            panic!("request should have executed");
        };
        assert_eq!(output.lease().bytes(), 1 << 19);
        assert_eq!(pool.granted(), 1 << 19);
        assert_eq!(runtime.retained_output_bytes(), 0);
        assert_eq!(
            runtime
                .step()
                .expect("step while the caller holds the charge"),
            StepOutcome::Blocked(BlockReason::Pool {
                requested: 1 << 19,
                available: 0
            }),
            "a popped result keeps its bytes reserved until its owner frees them"
        );
        drop(output);
        assert_eq!(pool.granted(), 0);
        assert_eq!(
            runtime.step().expect("step after release"),
            StepOutcome::Executed { results: 1 }
        );
    }

    #[test]
    fn one_pool_binds_every_runtime_drawing_on_it() {
        let pool = BytePool::new(1 << 19).shared();
        let mut first_executor = Encoder::new();
        first_executor.bytes_per_token = 1 << 19;
        let mut second_executor = Encoder::new();
        second_executor.bytes_per_token = 1 << 19;
        let mut first =
            BatchRuntime::new(first_executor, Arc::clone(&pool), config()).expect("first runtime");
        let mut second = BatchRuntime::new(second_executor, Arc::clone(&pool), config())
            .expect("second runtime");
        first.submit(vec![1]).expect("first request");
        second.submit(vec![1]).expect("second request");
        assert_eq!(
            first.step().expect("first step"),
            StepOutcome::Executed { results: 1 }
        );
        assert_eq!(
            second.step().expect("sibling step"),
            StepOutcome::Blocked(BlockReason::Pool {
                requested: 1 << 19,
                available: 0
            }),
            "a private per-runtime budget would have admitted this work"
        );
        let wait = pool.capacity_wait();
        drop(first.pop_completed().expect("first entry"));
        assert!(wait.changed(), "release publishes capacity readiness");
        assert_eq!(
            second.step().expect("sibling step after release"),
            StepOutcome::Executed { results: 1 }
        );
    }

    #[test]
    fn pool_bypass_is_bounded_and_an_aged_head_gets_the_next_grant() {
        let pool = BytePool::new(8).shared();
        let sibling_lease = pool.reserve(4).unwrap();
        let mut runtime = BatchRuntime::new(Encoder::new(), Arc::clone(&pool), config()).unwrap();
        let head = runtime.submit(vec![1; 8]).unwrap();
        for peer in 0..8 {
            let request = runtime.submit(vec![peer]).unwrap();
            assert_eq!(
                runtime.step().unwrap(),
                StepOutcome::Executed { results: 1 }
            );
            assert_eq!(pool.granted(), 5);
            let entry = runtime.pop_completed().unwrap();
            assert_eq!(entry.request(), request);
            assert_eq!(entry.output(), Some(&peer));
            drop(entry);
            assert_eq!(runtime.queued(), 1);
        }
        let later = runtime.submit(vec![9]).unwrap();
        for _ in 0..2 {
            assert_eq!(
                runtime.step().unwrap(),
                StepOutcome::Blocked(BlockReason::Pool {
                    requested: 8,
                    available: 4
                })
            );
        }
        drop(sibling_lease);
        assert_eq!(
            runtime.step().unwrap(),
            StepOutcome::Executed { results: 1 }
        );
        let entry = runtime.pop_completed().unwrap();
        assert_eq!(entry.request(), head);
        drop(entry);
        assert_eq!(
            runtime.step().unwrap(),
            StepOutcome::Executed { results: 1 }
        );
        assert_eq!(runtime.pop_completed().unwrap().request(), later);
    }

    #[test]
    fn pool_bypass_respects_the_scan_bound_and_does_not_execute_a_rejected_peer() {
        let pool = BytePool::new(16).shared();
        let _held = pool.reserve(8).unwrap();
        let mut executor = Encoder::new();
        executor.rejected_marker = Some(42);
        executor.bytes_per_token = 2;
        let mut cfg = config();
        cfg.max_waiting_requests = 12;
        let mut runtime = BatchRuntime::new(executor, pool, cfg).unwrap();
        runtime.submit(vec![0; 8]).unwrap();
        let mut blockers = Vec::new();
        for _ in 0..7 {
            blockers.push(runtime.submit(vec![0; 5]).unwrap());
        }
        runtime.submit(vec![42]).unwrap();
        let peer = runtime.submit(vec![1]).unwrap();
        assert_eq!(
            runtime.step().unwrap(),
            StepOutcome::Blocked(BlockReason::Pool {
                requested: 16,
                available: 8
            })
        );
        let cancelled = blockers[0];
        assert_eq!(runtime.cancel(cancelled), CancelOutcome::Queued);
        assert_eq!(
            runtime.step().unwrap(),
            StepOutcome::Cancelled { request: cancelled }
        );
        assert_eq!(runtime.pop_completed().unwrap().request(), cancelled);
        assert_eq!(
            runtime.step().unwrap(),
            StepOutcome::Executed { results: 1 }
        );
        assert_eq!(runtime.pop_completed().unwrap().request(), peer);
        assert_eq!(runtime.queued(), 8);
    }

    #[test]
    fn a_head_larger_than_the_pool_is_rejected_not_waited_out() {
        let mut executor = Encoder::new();
        executor.bytes_per_token = 1024;
        let pool = BytePool::new(1024).shared();
        let mut runtime = BatchRuntime::new(executor, pool, config()).expect("runtime");
        let oversized = runtime.submit(vec![0; 2]).expect("oversized request");
        let healthy = runtime.submit(vec![0; 1]).expect("healthy request");
        assert_eq!(
            runtime.step().expect("rejecting step"),
            StepOutcome::Rejected { request: oversized }
        );
        assert!(matches!(
            runtime.pop_completed().expect("rejection").outcome(),
            Terminal::Rejected(Rejection::RetainedOutputTooLarge {
                requested: 2048,
                capacity: 1024
            })
        ));
        assert_eq!(
            runtime.step().expect("healthy peer still runs"),
            StepOutcome::Executed { results: 1 }
        );
        assert_eq!(
            runtime.pop_completed().expect("healthy output").request(),
            healthy
        );
    }

    #[test]
    fn a_closed_pool_refuses_new_work_and_a_failed_submission_keeps_its_charge() {
        let pool = BytePool::new(1 << 19).shared();
        let mut runtime =
            BatchRuntime::new(Encoder::new(), Arc::clone(&pool), config()).expect("runtime");
        pool.close();
        assert!(matches!(
            runtime.submit(vec![1]),
            Err(RuntimeError::PoolClosed)
        ));
        drop(runtime);

        // A submission that fails after reaching the device keeps its reservation with
        // the executor: releasing it here would free pool bytes covering storage the
        // device may still read.
        let pool = BytePool::new(1 << 19).shared();
        let mut runtime = BatchRuntime::new(FailingEncoder::new(true), Arc::clone(&pool), config())
            .expect("runtime");
        runtime.submit(vec![1]).expect("request");
        let error = runtime.step().expect_err("executor failure");
        assert!(
            matches!(error, RuntimeError::Uncertain { .. }),
            "a device-visible failure is uncertain, got {error:?}"
        );
        assert_eq!(pool.granted(), 1, "the executor keeps the charge");
        assert_eq!(runtime.executor().quarantined.len(), 1);

        // Only a proven drain releases it, and no error ever carried the lease back.
        runtime.executor_mut().release_quarantined();
        assert_eq!(pool.granted(), 0);

        // A refusal before device access releases every reservation instead.
        let pool = BytePool::new(1 << 19).shared();
        let mut runtime =
            BatchRuntime::new(FailingEncoder::new(false), Arc::clone(&pool), config())
                .expect("runtime");
        runtime.submit(vec![1]).expect("request");
        assert!(matches!(runtime.step(), Err(RuntimeError::Executor { .. })));
        assert_eq!(pool.granted(), 0);
    }

    #[test]
    fn executor_and_pool_bypass_share_head_priority() {
        let pool = BytePool::new(8).shared();
        let sibling_lease = pool.reserve(4).unwrap();
        let mut encoder = Encoder::new();
        encoder.blocked_heads[0] = Some(1);
        let mut owner = BatchRuntime::new(encoder, Arc::clone(&pool), config()).unwrap();
        let first = owner.submit(vec![1; 8]).unwrap();
        for _ in 0..8 {
            let peer = owner.submit(vec![2]).unwrap();
            assert_eq!(owner.step().unwrap(), StepOutcome::Executed { results: 1 });
            assert_eq!(owner.pop_completed().unwrap().request(), peer);
        }
        let later = owner.submit(vec![3]).unwrap();
        assert!(matches!(
            owner.step().unwrap(),
            StepOutcome::Blocked(BlockReason::Executor { .. })
        ));
        owner.executor_mut().blocked_heads[0] = None;
        assert!(matches!(
            owner.step().unwrap(),
            StepOutcome::Blocked(BlockReason::Pool { .. })
        ));
        drop(sibling_lease);
        assert_eq!(owner.step().unwrap(), StepOutcome::Executed { results: 1 });
        assert_eq!(owner.pop_completed().unwrap().request(), first);
        assert_eq!(owner.step().unwrap(), StepOutcome::Executed { results: 1 });
        assert_eq!(owner.pop_completed().unwrap().request(), later);
    }

    #[test]
    fn blocked_peer_publication_before_arming_is_preserved() {
        for head_on_pool in [false, true] {
            let pool = BytePool::new(8).shared();
            let _sibling_lease = pool.reserve(4).unwrap();
            let mut encoder = Encoder::new();
            encoder.blocked_heads = [(!head_on_pool).then_some(1), Some(2)];
            let mut owner = BatchRuntime::new(encoder, pool, config()).unwrap();
            owner.submit(vec![1; 8]).unwrap();
            let peer = owner.submit(vec![2]).unwrap();
            let mut capacity = owner.capacity_wait();
            let StepOutcome::Blocked(mut reason @ BlockReason::Peers { .. }) =
                owner.step().unwrap()
            else {
                panic!("the head and peer wait on distinct resources");
            };
            owner.executor_mut().blocked_heads[1] = None;
            owner.executor().head_readiness[1].publish();
            let (waker, rx) = wake_channel();
            reason.wake_on_change(&mut capacity, &waker);
            assert_eq!(rx.try_recv(), Ok(()));
            assert!(!capacity.changed());
            assert_eq!(owner.step().unwrap(), StepOutcome::Executed { results: 1 });
            assert_eq!(owner.pop_completed().unwrap().request(), peer);
        }
    }

    #[test]
    fn executor_block_watches_pool_release_and_closure_with_a_ready_peer() {
        let pool = BytePool::new(8).shared();
        let sibling_lease = pool.reserve(8).unwrap();
        let mut encoder = Encoder::new();
        encoder.blocked_heads[0] = Some(1);
        let mut owner = BatchRuntime::new(encoder, Arc::clone(&pool), config()).unwrap();
        owner.submit(vec![1]).unwrap();
        let peer = owner.submit(vec![2]).unwrap();
        let mut capacity = owner.capacity_wait();
        let StepOutcome::Blocked(mut reason @ BlockReason::Executor { .. }) = owner.step().unwrap()
        else {
            panic!("the head is blocked and the ready peer cannot reserve");
        };
        let (waker, rx) = wake_channel();
        reason.wake_on_change(&mut capacity, &waker);
        drop(sibling_lease);
        assert_eq!(rx.try_recv(), Ok(()));
        drop((reason, capacity));
        assert_eq!(owner.step().unwrap(), StepOutcome::Executed { results: 1 });
        assert_eq!(owner.pop_completed().unwrap().request(), peer);
        let mut capacity = owner.capacity_wait();
        let StepOutcome::Blocked(mut reason @ BlockReason::Executor { .. }) = owner.step().unwrap()
        else {
            panic!("only the executor-blocked head remains");
        };
        pool.close();
        reason.wake_on_change(&mut capacity, &waker);
        assert_eq!(rx.try_recv(), Ok(()));
        assert!(matches!(owner.step(), Err(RuntimeError::PoolClosed)));
    }

    #[test]
    fn blocked_selection_carries_the_original_readiness_registration() {
        let mut runtime = runtime();
        runtime.submit(vec![1]).expect("first request");
        runtime.submit(vec![2]).expect("second request");
        runtime.executor_mut().busy = true;

        let StepOutcome::Blocked(BlockReason::Executor {
            reason: Constraint::WorkspaceBusy,
            mut wait,
        }) = runtime.step().expect("blocked step")
        else {
            panic!("expected workspace pressure")
        };
        assert_eq!(runtime.queued(), 2);
        assert!(runtime.executor().batches.is_empty());
        assert!(!wait.changed());

        // Release before arming. The original registration must retain the change.
        runtime.executor_mut().busy = false;
        runtime.executor().workspace_readiness.publish();
        assert!(wait.changed());
        let (waker, rx) = wake_channel();
        wait.wake_on_change(&waker);
        assert_eq!(rx.try_recv(), Ok(()));
        assert_eq!(
            runtime.step().expect("step after wakeup"),
            StepOutcome::Executed { results: 2 }
        );
    }

    #[test]
    fn oversized_head_is_rejected_without_blocking_later_work() {
        let mut executor = Encoder::new();
        executor.token_budget = Some(2);
        let mut runtime = BatchRuntime::new(executor, pool(), config()).expect("runtime");
        let oversized = runtime.submit(vec![1, 2, 3]).expect("oversized request");
        let first = runtime.submit(vec![4]).expect("first request");
        let second = runtime.submit(vec![5]).expect("second request");

        assert_eq!(
            runtime.step().expect("rejecting step"),
            StepOutcome::Rejected { request: oversized }
        );
        assert_eq!(runtime.queued(), 2);
        assert!(runtime.executor().batches.is_empty());

        let entry = runtime.pop_completed().expect("rejection entry");
        assert_eq!(entry.request(), oversized);
        assert_eq!(entry.retained_bytes(), 0);
        assert!(matches!(
            entry.outcome(),
            Terminal::Rejected(Rejection::Executor(Constraint::ExceedsTokenBudget {
                tokens: 3
            }))
        ));

        assert_eq!(
            runtime.step().expect("later work still runs"),
            StepOutcome::Executed { results: 2 }
        );
        assert_eq!(
            runtime.pop_completed().expect("first output").request(),
            first
        );
        assert_eq!(
            runtime.pop_completed().expect("second output").request(),
            second
        );
    }

    #[test]
    fn rejection_is_delivered_while_output_bytes_are_exhausted() {
        let mut executor = Encoder::new();
        executor.token_budget = Some(2);
        executor.bytes_per_token = 1 << 19;
        let mut runtime = BatchRuntime::new(executor, pool(), config()).expect("runtime");
        runtime.submit(vec![1]).expect("fitting request");
        assert_eq!(
            runtime.step().expect("first step"),
            StepOutcome::Executed { results: 1 }
        );
        assert_eq!(runtime.retained_output_bytes(), 1 << 19);

        let oversized = runtime.submit(vec![1, 2, 3]).expect("oversized request");
        assert_eq!(
            runtime.step().expect("rejecting step"),
            StepOutcome::Rejected { request: oversized }
        );
        assert!(
            runtime
                .pop_completed()
                .expect("first entry")
                .output()
                .is_some(),
            "earlier completions are delivered first"
        );
        assert!(matches!(
            runtime.pop_completed().expect("rejection entry").outcome(),
            Terminal::Rejected(_)
        ));
    }

    #[test]
    fn retained_rejections_count_toward_the_retained_bound() {
        // A rejection is a terminal entry the caller has not consumed, so it
        // occupies the retained-result budget exactly like an output. Counting
        // only outputs lets a rejection and a later output sit in `terminal`
        // together under `max_retained_results = 1`.
        let mut executor = Encoder::new();
        executor.token_budget = Some(2);
        let mut runtime = BatchRuntime::new(
            executor,
            pool(),
            BatchConfig {
                max_retained_results: 1,
                ..config()
            },
        )
        .expect("runtime");
        let oversized = runtime.submit(vec![1, 2, 3]).expect("oversized request");
        assert_eq!(
            runtime.step().expect("rejecting step"),
            StepOutcome::Rejected { request: oversized }
        );
        assert_eq!(runtime.retained_results(), 1);

        runtime.submit(vec![4]).expect("later request");
        assert!(
            matches!(
                runtime.step().expect("step under a full retained budget"),
                StepOutcome::Blocked(BlockReason::RetainedResults)
            ),
            "an unconsumed rejection must block execution, not share the bound"
        );
        assert_eq!(runtime.retained_results(), 1);

        // Consuming the rejection releases the budget for the queued request.
        assert!(matches!(
            runtime.pop_completed().expect("rejection entry").outcome(),
            Terminal::Rejected(_)
        ));
        assert_eq!(
            runtime.step().expect("step after consuming the rejection"),
            StepOutcome::Executed { results: 1 }
        );
        assert_eq!(runtime.retained_results(), 1);
    }

    #[test]
    fn waiting_capacity_is_bounded_independently_of_retained_results() {
        let mut runtime = BatchRuntime::new(
            Encoder::new(),
            pool(),
            BatchConfig {
                max_waiting_requests: 1,
                ..config()
            },
        )
        .expect("runtime");
        runtime.submit(vec![1]).expect("first request");
        assert!(matches!(
            runtime.submit(vec![2]),
            Err(RuntimeError::WaitingCapacityExhausted)
        ));
    }

    #[test]
    fn malformed_selection_is_still_a_runtime_error() {
        let mut runtime = runtime();
        runtime.submit(vec![1]).expect("request");
        runtime.executor_mut().over_select = true;
        assert!(matches!(
            runtime.step(),
            Err(RuntimeError::MalformedBatchSelection {
                selected: 2,
                candidates: 1
            })
        ));
    }

    #[test]
    fn a_short_completion_is_handed_back_to_the_executor() {
        let mut runtime = runtime();
        runtime.executor_mut().short_return = true;
        runtime.submit(vec![1]).expect("first request");
        runtime.submit(vec![2]).expect("second request");
        assert!(matches!(
            runtime.step(),
            Err(RuntimeError::MalformedCompletion { returned: 1, .. })
        ));
        assert_eq!(runtime.executor().retired_outputs, 1);
        assert_eq!(
            runtime.retained_results(),
            0,
            "nothing may be committed after a malformed completion"
        );
    }
}
