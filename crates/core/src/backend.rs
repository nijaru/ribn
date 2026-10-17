//! Compute-backend execution boundary.
//!
//! A backend executes one prepared batch of request segments against the continuation state
//! the model hands it. It owns physical resources and completion; it does not own model
//! identity, capability metadata or a plan that describes what it already is.

use std::fmt;

use crate::execution::{BatchError, ExecutionBatch, ExecutionBatchEvent, ExecutionSegment};
use crate::state::{InferenceStateSet, StateLocation};

pub use crate::device::DeviceId;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BackendSubmissionId(u64);

impl BackendSubmissionId {
    #[must_use]
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendError {
    Unsupported(&'static str),
    InvalidBatch(BatchError),
    StateCountMismatch,
    StateMismatch(&'static str),
    UnknownSubmission(BackendSubmissionId),
    ExecutionFailed(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(capability) => write!(f, "backend does not support {capability}"),
            Self::InvalidBatch(error) => write!(f, "invalid execution batch: {error}"),
            Self::StateCountMismatch => {
                f.write_str("execution batch and inference-state counts differ")
            }
            Self::StateMismatch(reason) => write!(f, "state does not satisfy execution: {reason}"),
            Self::UnknownSubmission(id) => write!(f, "backend submission {} is unknown", id.get()),
            Self::ExecutionFailed(reason) => write!(f, "execution failed: {reason}"),
        }
    }
}

impl std::error::Error for BackendError {}

pub trait ComputeBackend: Send {
    /// The device this backend executes on. Continuation state must be resident here.
    fn device(&self) -> DeviceId;

    /// # Errors
    ///
    /// Returns an error when a state set does not satisfy its request segment, or when the
    /// batch and state counts differ.
    fn validate_execution(
        &self,
        batch: &ExecutionBatch,
        states: &[InferenceStateSet],
    ) -> Result<(), BackendError> {
        if batch.len() != states.len() {
            return Err(BackendError::StateCountMismatch);
        }
        for (segment, state) in batch.segments().iter().zip(states) {
            self.validate_segment_state(segment, state)?;
        }
        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error when state does not satisfy one request segment.
    fn validate_segment_state(
        &self,
        segment: &ExecutionSegment,
        state: &InferenceStateSet,
    ) -> Result<(), BackendError> {
        if segment
            .state_requirements()
            .iter()
            .any(|requirement| !state.contains(*requirement))
        {
            return Err(BackendError::StateMismatch(
                "segment state requirement is missing or has the wrong layout",
            ));
        }
        if state
            .token_position()
            .is_some_and(|position| position != segment.state_position())
        {
            return Err(BackendError::StateMismatch(
                "state is at a different prefix position",
            ));
        }
        if state.location() == Some(StateLocation::Device(self.device())) {
            return Ok(());
        }
        if matches!(state.location(), Some(StateLocation::Device(_))) {
            return Err(BackendError::StateMismatch(
                "state is resident on a different device",
            ));
        }
        Ok(())
    }

    /// Release backend-owned physical resources associated with one logical
    /// inference-state set. Stateless backends may keep the default no-op.
    ///
    /// This is a lifecycle operation, not model execution. Callers must not
    /// release state while a submission still owns it.
    ///
    /// # Errors
    ///
    /// Returns a backend error when physical state cannot be released safely.
    /// Successful release must be idempotent across retries. When device
    /// ownership is uncertain, return an error and retain physical resources;
    /// the runtime then retains the terminal request and logical allocations.
    fn release_inference_state(&mut self, _state: &InferenceStateSet) -> Result<(), BackendError> {
        Ok(())
    }

    /// Submit a multi-request batch without requiring host synchronization
    /// with its completion.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the batch cannot be submitted.
    fn submit(
        &mut self,
        batch: &ExecutionBatch,
        states: &mut [InferenceStateSet],
    ) -> Result<BackendSubmissionId, BackendError>;

    /// Poll one submitted batch. Completion is consumed exactly once.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the submission is unknown or completion
    /// handling fails.
    fn poll(
        &mut self,
        submission: BackendSubmissionId,
    ) -> Result<Option<ExecutionBatchEvent>, BackendError>;

    /// # Errors
    ///
    /// Returns a backend error from completion polling.
    fn wait(
        &mut self,
        submission: BackendSubmissionId,
    ) -> Result<ExecutionBatchEvent, BackendError> {
        loop {
            if let Some(event) = self.poll(submission)? {
                return Ok(event);
            }
            std::thread::yield_now();
        }
    }

    /// # Errors
    ///
    /// Returns a backend error from submission or completion.
    fn execute(
        &mut self,
        batch: &ExecutionBatch,
        states: &mut [InferenceStateSet],
    ) -> Result<ExecutionBatchEvent, BackendError> {
        let submission = self.submit(batch, states)?;
        self.wait(submission)
    }
}
