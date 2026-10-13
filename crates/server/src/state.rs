use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

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
    model: RwLock<Option<TextModel>>,
    pub slots: Arc<Semaphore>,
    pub metrics: Metrics,
}

impl State {
    pub fn new(requests: usize) -> Self {
        Self {
            phase: AtomicU8::new(Readiness::Loading as u8),
            model: RwLock::new(None),
            slots: Arc::new(Semaphore::new(requests)),
            metrics: Metrics::default(),
        }
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
        format!(
            "ribn_ready {}\nribn_connections_active {}\nribn_connections_rejected_total {}\nribn_connection_deadlines_total {}\nribn_requests_active {}\nribn_requests_total {}\nribn_requests_rejected_total {}\nribn_requests_completed_total {}\nribn_requests_failed_total {}\nribn_requests_abandoned_total {}\nribn_prompt_tokens_total {}\nribn_completion_tokens_total {}\n",
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
            m.completion_tokens.load(Ordering::Relaxed)
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
}

/// Keeps transport admission charged through response consumption or abandonment.
pub(crate) struct RequestGuard {
    pub state: Arc<State>,
    _permit: OwnedSemaphorePermit,
    settled: bool,
    pub started: bool,
}

impl RequestGuard {
    pub fn new(state: Arc<State>, permit: OwnedSemaphorePermit) -> Self {
        state.metrics.active.fetch_add(1, Ordering::Relaxed);
        Self {
            state,
            _permit: permit,
            settled: false,
            started: false,
        }
    }
    pub fn complete(&mut self, usage: Usage) {
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
