//! Prepared execution vocabulary: what one submitted batch contains and what it produced.
//!
//! A model owns how it lowers engine work into these values; a backend consumes them. There
//! is deliberately no plan/description layer here: a plan that nothing reads is bookkeeping,
//! not a contract.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use crate::request::{RequestId, SamplingParams};
use crate::state::StateRequirement;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExecutionPhase {
    Prefill,
    Decode,
    SpecDraft,
    SpecVerify,
    Encoder,
    MoEExpert,
}

/// Token payload consumed by one text-model execution segment.
///
/// Prompt chunks borrow their backing storage through an [`Arc`] and carry a
/// range, so chunked prefill does not copy prompt tokens for every scheduling
/// iteration. Decode carries the single generated token that advances the
/// autoregressive state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionTokenInput {
    Prompt {
        tokens: Arc<[u32]>,
        start: u32,
        token_count: u32,
    },
    Decode {
        token: u32,
    },
}

impl ExecutionTokenInput {
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidTokenInput`] when the range is empty,
    /// overflows, or falls outside the shared prompt buffer.
    pub fn prompt(tokens: Arc<[u32]>, start: u32, token_count: u32) -> Result<Self, BatchError> {
        if token_count == 0 {
            return Err(BatchError::InvalidTokenInput);
        }
        let start_index = usize::try_from(start).map_err(|_| BatchError::InvalidTokenInput)?;
        let count = usize::try_from(token_count).map_err(|_| BatchError::InvalidTokenInput)?;
        let end = start_index
            .checked_add(count)
            .ok_or(BatchError::InvalidTokenInput)?;
        if end > tokens.len() {
            return Err(BatchError::InvalidTokenInput);
        }
        Ok(Self::Prompt {
            tokens,
            start,
            token_count,
        })
    }

    #[must_use]
    pub const fn decode(token: u32) -> Self {
        Self::Decode { token }
    }

    #[must_use]
    pub const fn token_count(&self) -> u32 {
        match self {
            Self::Prompt { token_count, .. } => *token_count,
            Self::Decode { .. } => 1,
        }
    }

    #[must_use]
    pub fn prompt_slice(&self) -> Option<&[u32]> {
        let Self::Prompt {
            tokens,
            start,
            token_count,
        } = self
        else {
            return None;
        };
        let start = usize::try_from(*start).ok()?;
        let count = usize::try_from(*token_count).ok()?;
        tokens.get(start..start.checked_add(count)?)
    }

    #[must_use]
    pub const fn decode_token(&self) -> Option<u32> {
        match self {
            Self::Decode { token } => Some(*token),
            Self::Prompt { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionSegment {
    request: RequestId,
    phase: ExecutionPhase,
    token_count: u32,
    state_position: u32,
    state_requirements: Arc<[StateRequirement]>,
    token_input: Option<ExecutionTokenInput>,
    sampling: Option<SamplingParams>,
}

impl ExecutionSegment {
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidSegmentBatchSize`] unless `batch_size` is
    /// one or [`BatchError::ZeroWork`] when `token_count` is zero. Multi-request
    /// batching is represented by [`ExecutionBatch`].
    pub fn new(
        request: RequestId,
        phase: ExecutionPhase,
        batch_size: u32,
        token_count: u32,
        state_position: u32,
        state_requirements: Vec<StateRequirement>,
    ) -> Result<Self, BatchError> {
        Self::new_shared(
            request,
            phase,
            batch_size,
            token_count,
            state_position,
            state_requirements.into(),
        )
    }

    /// Construct a segment over an already shared state-requirement set.
    ///
    /// This is the steady-state serving constructor: the model can share one immutable
    /// requirement array across every request/iteration instead of allocating a new `Vec`
    /// for each segment.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidSegmentBatchSize`] unless `batch_size` is
    /// one or [`BatchError::ZeroWork`] when `token_count` is zero.
    pub fn new_shared(
        request: RequestId,
        phase: ExecutionPhase,
        batch_size: u32,
        token_count: u32,
        state_position: u32,
        state_requirements: Arc<[StateRequirement]>,
    ) -> Result<Self, BatchError> {
        if batch_size != 1 {
            return Err(BatchError::InvalidSegmentBatchSize);
        }
        if token_count == 0 {
            return Err(BatchError::ZeroWork);
        }
        Ok(Self {
            request,
            phase,
            token_count,
            state_position,
            state_requirements,
            token_input: None,
            sampling: None,
        })
    }

    /// Attach the token payload consumed by this segment.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidTokenInput`] when the input family does not
    /// match prefill/decode semantics or its token count differs from the
    /// scheduled segment.
    pub fn with_token_input(mut self, input: ExecutionTokenInput) -> Result<Self, BatchError> {
        let phase_matches = matches!(
            (&self.phase, &input),
            (ExecutionPhase::Prefill, ExecutionTokenInput::Prompt { .. })
                | (ExecutionPhase::Decode, ExecutionTokenInput::Decode { .. })
        );
        if !phase_matches || input.token_count() != self.token_count {
            return Err(BatchError::InvalidTokenInput);
        }
        self.token_input = Some(input);
        Ok(self)
    }

    /// Request one sampled token from the logits produced by this segment.
    /// Intermediate prefill segments normally leave sampling disabled; the
    /// final prefill segment and ordinary decode request it.
    #[must_use]
    pub const fn with_sampling(mut self, sampling: SamplingParams) -> Self {
        self.sampling = Some(sampling);
        self
    }

    #[must_use]
    pub const fn request(&self) -> RequestId {
        self.request
    }

    #[must_use]
    pub const fn phase(&self) -> ExecutionPhase {
        self.phase
    }

    #[must_use]
    pub const fn batch_size(&self) -> u32 {
        1
    }

    #[must_use]
    pub const fn token_count(&self) -> u32 {
        self.token_count
    }

    #[must_use]
    pub const fn state_position(&self) -> u32 {
        self.state_position
    }

    #[must_use]
    pub fn state_requirements(&self) -> &[StateRequirement] {
        &self.state_requirements
    }

    #[must_use]
    pub const fn token_input(&self) -> Option<&ExecutionTokenInput> {
        self.token_input.as_ref()
    }

    #[must_use]
    pub const fn sampling(&self) -> Option<SamplingParams> {
        self.sampling
    }

    #[must_use]
    pub const fn requests_sampling(&self) -> bool {
        self.sampling.is_some()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionBatch {
    segments: Vec<ExecutionSegment>,
}

impl ExecutionBatch {
    /// # Errors
    ///
    /// Returns [`BatchError::EmptyBatch`] for no work or
    /// [`BatchError::DuplicateBatchRequest`] when a request appears more than
    /// once in the same submission.
    pub fn new(segments: Vec<ExecutionSegment>) -> Result<Self, BatchError> {
        if segments.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        let mut requests = HashSet::with_capacity(segments.len());
        for segment in &segments {
            if !requests.insert(segment.request()) {
                return Err(BatchError::DuplicateBatchRequest);
            }
        }
        Ok(Self { segments })
    }

    #[must_use]
    pub fn segments(&self) -> &[ExecutionSegment] {
        &self.segments
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    #[must_use]
    pub fn total_tokens(&self) -> Option<u32> {
        self.segments.iter().try_fold(0_u32, |total, segment| {
            total.checked_add(segment.token_count())
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutionEvent {
    request: RequestId,
    phase: ExecutionPhase,
    token_count: u32,
    output_token: Option<u32>,
}

impl ExecutionEvent {
    #[must_use]
    pub const fn new(request: RequestId, phase: ExecutionPhase, token_count: u32) -> Option<Self> {
        if token_count == 0 {
            None
        } else {
            Some(Self {
                request,
                phase,
                token_count,
                output_token: None,
            })
        }
    }

    /// Attach the one committed sampled token produced by this execution.
    #[must_use]
    pub const fn with_output_token(mut self, token: u32) -> Self {
        self.output_token = Some(token);
        self
    }

    #[must_use]
    pub const fn request(self) -> RequestId {
        self.request
    }

    #[must_use]
    pub const fn phase(self) -> ExecutionPhase {
        self.phase
    }

    #[must_use]
    pub const fn token_count(self) -> u32 {
        self.token_count
    }

    #[must_use]
    pub const fn output_token(self) -> Option<u32> {
        self.output_token
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionBatchEvent {
    events: Vec<ExecutionEvent>,
}

impl ExecutionBatchEvent {
    /// # Errors
    ///
    /// Returns [`BatchError::EmptyBatch`] when the completion has no events or
    /// [`BatchError::DuplicateBatchRequest`] when a request occurs twice.
    pub fn new(events: Vec<ExecutionEvent>) -> Result<Self, BatchError> {
        if events.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        let mut requests = HashSet::with_capacity(events.len());
        for event in &events {
            if !requests.insert(event.request()) {
                return Err(BatchError::DuplicateBatchRequest);
            }
        }
        Ok(Self { events })
    }

    #[must_use]
    pub fn events(&self) -> &[ExecutionEvent] {
        &self.events
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BatchError {
    EmptyBatch,
    DuplicateBatchRequest,
    InvalidSegmentBatchSize,
    InvalidTokenInput,
    ZeroWork,
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch => f.write_str("execution batch must contain at least one segment"),
            Self::DuplicateBatchRequest => {
                f.write_str("execution batch contains duplicate request work")
            }
            Self::InvalidSegmentBatchSize => {
                f.write_str("an execution segment describes exactly one request")
            }
            Self::InvalidTokenInput => {
                f.write_str("execution token input does not match the scheduled segment")
            }
            Self::ZeroWork => f.write_str("execution segment must contain non-zero work"),
        }
    }
}

impl std::error::Error for BatchError {}
