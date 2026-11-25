use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ribn_text::{TextModel, Usage};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Model lifecycle, not a request-capacity guarantee.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Readiness {
    Loading,
    Serving,
    Closing,
    Faulted,
}

impl Readiness {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Loading => "loading",
            Self::Serving => "serving",
            Self::Closing => "closing",
            Self::Faulted => "faulted",
        }
    }
}

pub(crate) struct State {
    phase: AtomicU8,
    created_unix: u64,
    model: RwLock<Option<TextModel>>,
    pub slots: Arc<Semaphore>,
    pub metrics: Metrics,
}

impl State {
    pub fn new(requests: usize) -> Self {
        let created_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        Self {
            phase: AtomicU8::new(Readiness::Loading as u8),
            created_unix,
            model: RwLock::new(None),
            slots: Arc::new(Semaphore::new(requests)),
            metrics: Metrics::default(),
        }
    }
    pub const fn created_unix(&self) -> u64 {
        self.created_unix
    }
    pub fn set_phase(&self, phase: Readiness) {
        self.phase.store(phase as u8, Ordering::SeqCst);
    }
    pub fn install(&self, model: TextModel) {
        *self
            .model
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(model);
        self.set_phase(Readiness::Serving);
    }
    pub fn model(&self) -> Option<TextModel> {
        self.model
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
    pub fn readiness(&self) -> Readiness {
        let phase = match self.phase.load(Ordering::SeqCst) {
            0 => Readiness::Loading,
            1 => Readiness::Serving,
            2 => Readiness::Closing,
            _ => Readiness::Faulted,
        };
        if phase == Readiness::Serving && self.model().is_some_and(|m| m.check_open().is_err()) {
            // Do not overwrite a concurrent transition to closing.
            let _ = self.phase.compare_exchange(
                Readiness::Serving as u8,
                Readiness::Faulted as u8,
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
            return self.readiness();
        }
        phase
    }
    pub fn metrics_text(&self) -> String {
        let m = &self.metrics;
        #[expect(
            clippy::cast_precision_loss,
            reason = "Prometheus seconds sum exported as floating point"
        )]
        let ttft_seconds = m.ttft_nanos_sum.load(Ordering::Relaxed) as f64 / 1e9;
        #[expect(
            clippy::cast_precision_loss,
            reason = "Prometheus seconds sum exported as floating point"
        )]
        let duration_seconds = m.request_duration_nanos_sum.load(Ordering::Relaxed) as f64 / 1e9;
        format!(
            "ribn_ready {}\nribn_connections_active {}\nribn_connections_rejected_total {}\nribn_connection_deadlines_total {}\nribn_requests_active {}\nribn_requests_total {}\nribn_requests_rejected_total {}\nribn_requests_completed_total {}\nribn_requests_failed_total {}\nribn_requests_abandoned_total {}\nribn_prompt_tokens_total {}\nribn_completion_tokens_total {}\nribn_ttft_seconds_sum {ttft_seconds:.6}\nribn_ttft_seconds_count {}\nribn_request_duration_seconds_sum {duration_seconds:.6}\nribn_request_duration_seconds_count {}\n",
            u8::from(self.readiness() == Readiness::Serving),
            m.connections.load(Ordering::Relaxed),
            m.connections_rejected.load(Ordering::Relaxed),
            m.deadlines.load(Ordering::Relaxed),
            m.active.load(Ordering::Relaxed),
            m.requests.load(Ordering::Relaxed),
            m.rejected.load(Ordering::Relaxed),
            m.completed.load(Ordering::Relaxed),
            m.failed.load(Ordering::Relaxed),
            m.abandoned.load(Ordering::Relaxed),
            m.prompt_tokens.load(Ordering::Relaxed),
            m.completion_tokens.load(Ordering::Relaxed),
            m.ttft_count.load(Ordering::Relaxed),
            m.request_duration_count.load(Ordering::Relaxed),
        )
    }
}

#[derive(Default)]
pub(crate) struct Metrics {
    pub connections: AtomicUsize,
    pub connections_rejected: AtomicU64,
    pub deadlines: AtomicU64,
    pub active: AtomicUsize,
    pub requests: AtomicU64,
    pub rejected: AtomicU64,
    pub completed: AtomicU64,
    pub failed: AtomicU64,
    pub abandoned: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub completion_tokens: AtomicU64,
    pub ttft_nanos_sum: AtomicU64,
    pub ttft_count: AtomicU64,
    pub request_duration_nanos_sum: AtomicU64,
    pub request_duration_count: AtomicU64,
}

/// Keeps transport admission charged through response consumption or abandonment.
pub(crate) struct RequestGuard {
    pub state: Arc<State>,
    _permit: OwnedSemaphorePermit,
    started_at: Instant,
    ttft_recorded: bool,
    settled: bool,
    pub started: bool,
}

impl RequestGuard {
    pub fn new(state: Arc<State>, permit: OwnedSemaphorePermit) -> Self {
        state.metrics.active.fetch_add(1, Ordering::Relaxed);
        Self {
            state,
            _permit: permit,
            started_at: Instant::now(),
            ttft_recorded: false,
            settled: false,
            started: false,
        }
    }
    pub fn record_ttft(&mut self) {
        if self.ttft_recorded {
            return;
        }
        self.ttft_recorded = true;
        let nanos = u64::try_from(self.started_at.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.state
            .metrics
            .ttft_nanos_sum
            .fetch_add(nanos, Ordering::Relaxed);
        self.state
            .metrics
            .ttft_count
            .fetch_add(1, Ordering::Relaxed);
    }
    pub fn complete(&mut self, usage: Usage) {
        self.record_ttft();
        let nanos = u64::try_from(self.started_at.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.state
            .metrics
            .request_duration_nanos_sum
            .fetch_add(nanos, Ordering::Relaxed);
        self.state
            .metrics
            .request_duration_count
            .fetch_add(1, Ordering::Relaxed);
        self.settled = true;
        self.state.metrics.completed.fetch_add(1, Ordering::Relaxed);
        self.state
            .metrics
            .prompt_tokens
            .fetch_add(u64::from(usage.prompt_tokens), Ordering::Relaxed);
        self.state
            .metrics
            .completion_tokens
            .fetch_add(u64::from(usage.completion_tokens), Ordering::Relaxed);
    }
    pub fn fail(&mut self) {
        self.settled = true;
        self.state.metrics.failed.fetch_add(1, Ordering::Relaxed);
    }
    pub fn observe_owner(&self) {
        // Read owner state instead of guessing failure scope from request errors.
        self.state.readiness();
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.state.metrics.active.fetch_sub(1, Ordering::Relaxed);
        if self.started && !self.settled {
            self.state.metrics.abandoned.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub(crate) struct ConnectionGuard(pub Arc<State>);
impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.0.metrics.connections.fetch_sub(1, Ordering::Relaxed);
    }
}
