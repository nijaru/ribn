//! Typed model-state descriptions and lifecycle boundary.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ribn_foundation::{BytePool, PoolLease, ReserveError};

static NEXT_STATE_ID: AtomicU64 = AtomicU64::new(1);

use crate::device::DeviceId;
use crate::tensor::DataType;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StateId(u64);

#[cfg_attr(not(test), allow(dead_code))]
impl StateId {
    pub(crate) const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Geometry of a paged KV continuation.
///
/// `page_tokens` is the fixed page size a model declares and never varies per
/// request; `pages` is the concrete number of pages one sequence holds. A
/// sequence's capacity is `pages * page_tokens`, and it grows by adding pages
/// rather than by reallocating a capacity-tailed buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KvStateSpec {
    layer_count: u16,
    kv_heads: u16,
    head_dim: u16,
    page_tokens: u32,
    pages: u32,
    dtype: DataType,
}

impl KvStateSpec {
    /// # Errors
    ///
    /// Returns [`StateSpecError::ZeroDimension`] when any state dimension is zero.
    pub fn new(
        layer_count: u16,
        kv_heads: u16,
        head_dim: u16,
        page_tokens: u32,
        dtype: DataType,
    ) -> Result<Self, StateSpecError> {
        Self::with_pages(layer_count, kv_heads, head_dim, page_tokens, 1, dtype)
    }

    /// # Errors
    ///
    /// Returns [`StateSpecError::ZeroDimension`] when any dimension or the page
    /// count is zero.
    pub fn with_pages(
        layer_count: u16,
        kv_heads: u16,
        head_dim: u16,
        page_tokens: u32,
        pages: u32,
        dtype: DataType,
    ) -> Result<Self, StateSpecError> {
        if layer_count == 0 || kv_heads == 0 || head_dim == 0 || page_tokens == 0 || pages == 0 {
            return Err(StateSpecError::ZeroDimension);
        }
        if pages.checked_mul(page_tokens).is_none() {
            return Err(StateSpecError::CapacityOverflow);
        }
        Ok(Self {
            layer_count,
            kv_heads,
            head_dim,
            page_tokens,
            pages,
            dtype,
        })
    }

    #[must_use]
    pub const fn layer_count(self) -> u16 {
        self.layer_count
    }

    #[must_use]
    pub const fn kv_heads(self) -> u16 {
        self.kv_heads
    }

    #[must_use]
    pub const fn head_dim(self) -> u16 {
        self.head_dim
    }

    /// Fixed number of tokens one page holds.
    #[must_use]
    pub const fn page_tokens(self) -> u32 {
        self.page_tokens
    }

    /// Pages this sequence holds.
    #[must_use]
    pub const fn pages(self) -> u32 {
        self.pages
    }

    /// Capacity in tokens: `pages * page_tokens`, validated at construction.
    #[must_use]
    pub const fn capacity_tokens(self) -> u32 {
        self.pages * self.page_tokens
    }

    #[must_use]
    pub const fn dtype(self) -> DataType {
        self.dtype
    }

    /// Pages needed to cover `tokens`, or `None` on overflow.
    #[must_use]
    pub const fn pages_for(self, tokens: u32) -> Option<u32> {
        if tokens == 0 {
            return None;
        }
        let pages = tokens.div_ceil(self.page_tokens);
        Some(pages)
    }

    /// The same page geometry sized for a concrete number of tokens.
    ///
    /// A declaration states the page geometry and the most a component holds; a
    /// concrete sequence holds the pages its own work can reach.
    ///
    /// # Errors
    /// Returns [`StateSpecError::ZeroDimension`] when the derived page count is zero.
    pub fn for_tokens(self, tokens: u32) -> Result<Self, StateSpecError> {
        let pages = self
            .pages_for(tokens)
            .ok_or(StateSpecError::ZeroDimension)?;
        Self::with_pages(
            self.layer_count,
            self.kv_heads,
            self.head_dim,
            self.page_tokens,
            pages,
            self.dtype,
        )
    }

    /// Bytes one page occupies: `layers * kv_heads * head_dim * 2 * dtype`.
    #[must_use]
    pub fn page_bytes(self) -> Option<u64> {
        let elements = u64::from(self.layer_count)
            .checked_mul(u64::from(self.kv_heads))?
            .checked_mul(u64::from(self.head_dim))?
            .checked_mul(u64::from(self.page_tokens))?
            .checked_mul(2)?;
        elements.checked_mul(self.dtype.byte_width())
    }

    /// Bytes this sequence's pages occupy.
    #[must_use]
    pub fn byte_size(self) -> Option<u64> {
        self.page_bytes()?.checked_mul(u64::from(self.pages))
    }
}

/// Storage geometry for a bank of recurrent state matrices.
///
/// This describes physical semantic state as `[matrix][row][column]`. Model
/// projection head counts are deliberately absent: a Gated-DeltaNet model can
/// tile projected key heads across state matrices without multiplying the
/// persistent state allocation by the projection-head count.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RecurrentMatrixShape {
    matrix_count: u16,
    rows: u16,
    columns: u16,
}

impl RecurrentMatrixShape {
    #[must_use]
    pub const fn new(matrix_count: u16, rows: u16, columns: u16) -> Option<Self> {
        if matrix_count == 0 || rows == 0 || columns == 0 {
            None
        } else {
            Some(Self {
                matrix_count,
                rows,
                columns,
            })
        }
    }

    #[must_use]
    pub const fn matrix_count(self) -> u16 {
        self.matrix_count
    }

    #[must_use]
    pub const fn rows(self) -> u16 {
        self.rows
    }

    #[must_use]
    pub const fn columns(self) -> u16 {
        self.columns
    }
}

/// Persistent causal-convolution history as `[channel][history_token]`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ConvolutionStateShape {
    channels: u32,
    history_tokens: u16,
}

impl ConvolutionStateShape {
    #[must_use]
    pub const fn new(channels: u32, history_tokens: u16) -> Option<Self> {
        if channels == 0 || history_tokens == 0 {
            None
        } else {
            Some(Self {
                channels,
                history_tokens,
            })
        }
    }

    #[must_use]
    pub const fn channels(self) -> u32 {
        self.channels
    }

    #[must_use]
    pub const fn history_tokens(self) -> u16 {
        self.history_tokens
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RecurrentStateSpec {
    layer_count: u16,
    matrix: RecurrentMatrixShape,
    convolution: ConvolutionStateShape,
    matrix_dtype: DataType,
    convolution_dtype: DataType,
}

impl RecurrentStateSpec {
    /// # Errors
    ///
    /// Returns [`StateSpecError::ZeroDimension`] when `layer_count` is zero.
    pub fn new(
        layer_count: u16,
        matrix: RecurrentMatrixShape,
        convolution: ConvolutionStateShape,
        matrix_dtype: DataType,
        convolution_dtype: DataType,
    ) -> Result<Self, StateSpecError> {
        if layer_count == 0 {
            return Err(StateSpecError::ZeroDimension);
        }
        Ok(Self {
            layer_count,
            matrix,
            convolution,
            matrix_dtype,
            convolution_dtype,
        })
    }

    #[must_use]
    pub const fn layer_count(self) -> u16 {
        self.layer_count
    }

    #[must_use]
    pub const fn matrix(self) -> RecurrentMatrixShape {
        self.matrix
    }

    #[must_use]
    pub const fn convolution(self) -> ConvolutionStateShape {
        self.convolution
    }

    #[must_use]
    pub const fn matrix_dtype(self) -> DataType {
        self.matrix_dtype
    }

    #[must_use]
    pub const fn convolution_dtype(self) -> DataType {
        self.convolution_dtype
    }

    #[must_use]
    pub fn byte_size(self) -> Option<u64> {
        let matrix_elements = u64::from(self.matrix.matrix_count())
            .checked_mul(u64::from(self.matrix.rows()))?
            .checked_mul(u64::from(self.matrix.columns()))?
            .checked_mul(u64::from(self.layer_count))?;
        let convolution_elements = u64::from(self.convolution.channels())
            .checked_mul(u64::from(self.convolution.history_tokens()))?
            .checked_mul(u64::from(self.layer_count))?;
        matrix_elements
            .checked_mul(self.matrix_dtype.byte_width())?
            .checked_add(convolution_elements.checked_mul(self.convolution_dtype.byte_width())?)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StateRequirement {
    FullAttentionKv(KvStateSpec),
    Recurrent(RecurrentStateSpec),
}

impl StateRequirement {
    #[must_use]
    pub fn byte_size(self) -> Option<u64> {
        match self {
            Self::FullAttentionKv(spec) => spec.byte_size(),
            Self::Recurrent(spec) => spec.byte_size(),
        }
    }

    /// This requirement sized for a sequence that reaches `tokens`.
    ///
    /// A declaration covers a maximum reach, so one declared schema derives the
    /// concrete page count of any sequence that reaches no further than `tokens`.
    /// Recurrent state is a fixed-size summary of the whole prefix and has no
    /// capacity dimension, so it is returned unchanged.
    ///
    /// # Errors
    /// Returns [`StateSpecError::ZeroDimension`] when `tokens` is zero.
    pub fn for_tokens(self, tokens: u32) -> Result<Self, StateSpecError> {
        match self {
            Self::FullAttentionKv(spec) => Ok(Self::FullAttentionKv(spec.for_tokens(tokens)?)),
            Self::Recurrent(spec) => Ok(Self::Recurrent(spec)),
        }
    }

    /// Whether this concrete requirement fits `declaration`.
    ///
    /// Same component with a matching shape and no more capacity than the declared
    /// maximum is covered. A declaration is a schema plus a bound, so concrete state
    /// that reaches less of the context is valid, while a different shape, dtype or an
    /// exceeding capacity is not.
    #[must_use]
    pub fn within(self, declaration: Self) -> bool {
        match (self, declaration) {
            (Self::FullAttentionKv(concrete), Self::FullAttentionKv(declared)) => {
                concrete.layer_count == declared.layer_count
                    && concrete.kv_heads == declared.kv_heads
                    && concrete.head_dim == declared.head_dim
                    && concrete.dtype == declared.dtype
                    && concrete.page_tokens == declared.page_tokens
                    && concrete.pages <= declared.pages
            }
            (Self::Recurrent(concrete), Self::Recurrent(declared)) => concrete == declared,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct StateHandle {
    id: StateId,
    requirement: StateRequirement,
    device: DeviceId,
    token_position: u32,
}

#[cfg_attr(not(test), allow(dead_code))]
impl StateHandle {
    pub(crate) const fn new(
        id: StateId,
        requirement: StateRequirement,
        device: DeviceId,
        token_position: u32,
    ) -> Self {
        Self {
            id,
            requirement,
            device,
            token_position,
        }
    }

    #[must_use]
    pub const fn id(&self) -> StateId {
        self.id
    }

    #[must_use]
    pub const fn requirement(&self) -> StateRequirement {
        self.requirement
    }

    #[must_use]
    pub const fn device(&self) -> DeviceId {
        self.device
    }

    #[must_use]
    pub const fn token_position(&self) -> u32 {
        self.token_position
    }
}

#[derive(Debug, Eq, Hash, PartialEq)]
pub struct KvState {
    handle: StateHandle,
    spec: KvStateSpec,
}

#[cfg_attr(not(test), allow(dead_code))]
impl KvState {
    pub(crate) fn new(handle: StateHandle, spec: KvStateSpec) -> Option<Self> {
        if matches!(handle.requirement(), StateRequirement::FullAttentionKv(actual) if actual == spec)
        {
            Some(Self { handle, spec })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn handle(&self) -> &StateHandle {
        &self.handle
    }

    #[must_use]
    pub const fn spec(&self) -> KvStateSpec {
        self.spec
    }
}

#[derive(Debug, Eq, Hash, PartialEq)]
pub struct RecurrentState {
    handle: StateHandle,
    spec: RecurrentStateSpec,
}

#[cfg_attr(not(test), allow(dead_code))]
impl RecurrentState {
    pub(crate) fn new(handle: StateHandle, spec: RecurrentStateSpec) -> Option<Self> {
        if matches!(handle.requirement(), StateRequirement::Recurrent(actual) if actual == spec) {
            Some(Self { handle, spec })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn handle(&self) -> &StateHandle {
        &self.handle
    }

    #[must_use]
    pub const fn spec(&self) -> RecurrentStateSpec {
        self.spec
    }
}

#[derive(Debug, Eq, Hash, PartialEq)]
pub enum InferenceState {
    Kv(KvState),
    Recurrent(RecurrentState),
}

impl InferenceState {
    #[must_use]
    pub const fn handle(&self) -> &StateHandle {
        match self {
            Self::Kv(state) => state.handle(),
            Self::Recurrent(state) => state.handle(),
        }
    }

    #[must_use]
    pub const fn requirement(&self) -> StateRequirement {
        self.handle().requirement()
    }

    fn set_position(&mut self, token_position: u32) {
        match self {
            Self::Kv(state) => state.handle.token_position = token_position,
            Self::Recurrent(state) => state.handle.token_position = token_position,
        }
    }
}

impl From<KvState> for InferenceState {
    fn from(value: KvState) -> Self {
        Self::Kv(value)
    }
}

impl From<RecurrentState> for InferenceState {
    fn from(value: RecurrentState) -> Self {
        Self::Recurrent(value)
    }
}

/// Typed state at one semantic prefix boundary for the transitional backend seam.
/// Models with different continuation semantics may instead keep strongly typed
/// state behind the AR runtime's `GenerationExecutor`; this enum is not a mandate
/// to universalize their physical layout or retention policy.
/// The set transfers logical continuation ownership. The manager retains its
/// byte grant until explicit completion-safe release; inspecting a handle creates
/// neither another continuation owner nor a pool lease.
///
/// ```compile_fail
/// use engine_core::state::InferenceStateSet;
/// let state = InferenceStateSet::new(Vec::new()).unwrap();
/// let duplicate = state.clone();
/// ```
#[derive(Debug, Eq, Hash, PartialEq)]
pub struct InferenceStateSet {
    states: Vec<InferenceState>,
}

impl InferenceStateSet {
    /// # Errors
    ///
    /// Returns a state error when states disagree on token position or device, or
    /// when the same state requirement appears more than once.
    pub fn new(states: Vec<InferenceState>) -> Result<Self, StateError> {
        if let Some(first) = states.first() {
            let position = first.handle().token_position();
            let device = first.handle().device();
            for (index, state) in states.iter().enumerate() {
                if state.handle().token_position() != position {
                    return Err(StateError::PositionMismatch);
                }
                if state.handle().device() != device {
                    return Err(StateError::DeviceMismatch);
                }
                if states[..index]
                    .iter()
                    .any(|known| known.requirement() == state.requirement())
                {
                    return Err(StateError::DuplicateRequirement);
                }
            }
        }
        Ok(Self { states })
    }

    /// Convenience constructor for the current Qwen hybrid path.
    ///
    /// # Errors
    ///
    /// Returns a state error if the supplied states do not share a semantic
    /// prefix boundary or duplicate a state requirement.
    pub fn try_new(
        kv: Option<KvState>,
        recurrent: Option<RecurrentState>,
    ) -> Result<Self, StateError> {
        let mut states = Vec::with_capacity(2);
        if let Some(state) = kv {
            states.push(state.into());
        }
        if let Some(state) = recurrent {
            states.push(state.into());
        }
        Self::new(states)
    }

    #[must_use]
    pub fn states(&self) -> &[InferenceState] {
        &self.states
    }

    #[must_use]
    pub fn kv(&self) -> Option<&KvState> {
        self.states.iter().find_map(|state| match state {
            InferenceState::Kv(value) => Some(value),
            InferenceState::Recurrent(_) => None,
        })
    }

    #[must_use]
    pub fn recurrent(&self) -> Option<&RecurrentState> {
        self.states.iter().find_map(|state| match state {
            InferenceState::Kv(_) => None,
            InferenceState::Recurrent(value) => Some(value),
        })
    }

    #[must_use]
    pub fn token_position(&self) -> Option<u32> {
        self.states
            .first()
            .map(|state| state.handle().token_position())
    }

    #[must_use]
    pub fn device(&self) -> Option<DeviceId> {
        self.states.first().map(|state| state.handle().device())
    }

    #[must_use]
    pub fn contains(&self, requirement: StateRequirement) -> bool {
        self.states
            .iter()
            .any(|state| state.requirement() == requirement)
    }

    /// The concrete requirements this state satisfies, in state order.
    #[must_use]
    pub fn requirements(&self) -> Vec<StateRequirement> {
        self.states
            .iter()
            .map(InferenceState::requirement)
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StateSpecError {
    ZeroDimension,
    CapacityOverflow,
}

impl fmt::Display for StateSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => f.write_str("state dimensions must be greater than zero"),
            Self::CapacityOverflow => f.write_str("state page capacity overflows tokens"),
        }
    }
}

impl std::error::Error for StateSpecError {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StateError {
    UnsupportedRequirement,
    PoolClosed,
    CapacityExceeded {
        requested_bytes: u64,
        available_bytes: u64,
    },
    SizeOverflow,
    InvalidHandle,
    DuplicateRequirement,
    PositionMismatch,
    DeviceMismatch,
    PositionRegression,
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedRequirement => f.write_str("state requirement is unsupported"),
            Self::PoolClosed => f.write_str("state reservation pool is closed"),
            Self::CapacityExceeded {
                requested_bytes,
                available_bytes,
            } => write!(
                f,
                "state allocation requires {requested_bytes} bytes, but only {available_bytes} are available"
            ),
            Self::SizeOverflow => f.write_str("state size calculation overflowed"),
            Self::InvalidHandle => f.write_str("state handle is invalid or already released"),
            Self::DuplicateRequirement => f.write_str("state requirement appears more than once"),
            Self::PositionMismatch => f.write_str("state families have different token positions"),
            Self::DeviceMismatch => f.write_str("state families are resident on different devices"),
            Self::PositionRegression => f.write_str("state token position cannot move backwards"),
        }
    }
}

impl std::error::Error for StateError {}

/// Logical provenance and positions over supplied reservation authorities.
/// This manager does not prove physical completion. Models must release backend
/// storage safely before releasing its records; abandoned records keep their charge.
#[derive(Debug)]
pub struct LogicalStateManager {
    device: DeviceId,
    pool: Arc<BytePool>,
    allocations: HashMap<StateId, AllocationRecord>,
}

#[derive(Debug)]
struct AllocationRecord {
    requirement: StateRequirement,
    device: DeviceId,
    token_position: u32,
    // One aggregate grant shared by the bundle's records, never charged twice.
    reservation: Arc<PoolLease>,
}

impl LogicalStateManager {
    #[must_use]
    pub fn new(device: DeviceId, pool: Arc<BytePool>) -> Self {
        Self {
            device,
            pool,
            allocations: HashMap::new(),
        }
    }

    #[must_use]
    pub const fn device(&self) -> DeviceId {
        self.device
    }

    #[must_use]
    pub fn capacity_bytes(&self) -> u64 {
        self.pool.capacity()
    }

    /// Register before allocation. Returning a grant and closing its authority
    /// publish; commit and unsuccessful allocation/release do not. Siblings share
    /// the source.
    #[must_use]
    pub fn capacity_wait(&self) -> ribn_foundation::ReadinessWait {
        self.pool.capacity_wait()
    }

    /// Total grants from this authority, including sibling owners.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.pool.granted()
    }

    fn new_handle(&self, requirement: StateRequirement) -> Result<StateHandle, StateError> {
        // Backend allocation keys must not alias across managers sharing a pool.
        let raw_id = NEXT_STATE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| StateError::InvalidHandle)?;
        let id = StateId::new(raw_id).ok_or(StateError::InvalidHandle)?;
        Ok(StateHandle::new(id, requirement, self.device, 0))
    }

    fn grant(&self, bytes: u64) -> Result<Arc<PoolLease>, StateError> {
        let pool = &self.pool;
        let lease = pool.reserve(bytes).map_err(|error| match error {
            ReserveError::Exhausted {
                requested,
                available,
            } => StateError::CapacityExceeded {
                requested_bytes: requested,
                available_bytes: available,
            },
            ReserveError::TooLarge { requested, .. } => StateError::CapacityExceeded {
                requested_bytes: requested,
                available_bytes: pool.available(),
            },
            ReserveError::Closed => StateError::PoolClosed,
        })?;
        Ok(Arc::new(lease))
    }

    fn insert(&mut self, handle: &StateHandle, reservation: Arc<PoolLease>) {
        self.allocations.insert(
            handle.id(),
            AllocationRecord {
                requirement: handle.requirement(),
                device: handle.device(),
                token_position: handle.token_position(),
                reservation,
            },
        );
    }

    fn reserve(&mut self, requirement: StateRequirement) -> Result<StateHandle, StateError> {
        let bytes = requirement.byte_size().ok_or(StateError::SizeOverflow)?;
        let handle = self.new_handle(requirement)?;
        let reservation = self.grant(bytes)?;
        self.insert(&handle, reservation);
        Ok(handle)
    }

    fn record_for(&self, handle: &StateHandle) -> Result<&AllocationRecord, StateError> {
        let Some(record) = self.allocations.get(&handle.id()) else {
            return Err(StateError::InvalidHandle);
        };
        if record.requirement != handle.requirement()
            || record.device != handle.device()
            || record.token_position != handle.token_position()
        {
            return Err(StateError::InvalidHandle);
        }
        Ok(record)
    }
}

impl Drop for LogicalStateManager {
    fn drop(&mut self) {
        // Logical-owner disappearance is not proof that backend storage retired.
        // Successful model teardown empties these records via explicit release.
        for (_, record) in self.allocations.drain() {
            std::mem::forget(record.reservation);
        }
    }
}

impl LogicalStateManager {
    /// Allocate a coherent bundle with one aggregate grant. Nothing commits or
    /// publishes on failure; the charge persists until its last component releases.
    ///
    /// # Errors
    ///
    /// Rejects invalid or overflowing requirements, a closed authority and
    /// insufficient aggregate capacity.
    pub fn allocate_set(
        &mut self,
        requirements: &[StateRequirement],
    ) -> Result<InferenceStateSet, StateError> {
        let mut bytes = 0_u64;
        let mut states = Vec::with_capacity(requirements.len());
        for &requirement in requirements {
            bytes = bytes
                .checked_add(requirement.byte_size().ok_or(StateError::SizeOverflow)?)
                .ok_or(StateError::SizeOverflow)?;
            let handle = self.new_handle(requirement)?;
            states.push(match requirement {
                StateRequirement::FullAttentionKv(spec) => {
                    InferenceState::Kv(KvState { handle, spec })
                }
                StateRequirement::Recurrent(spec) => {
                    InferenceState::Recurrent(RecurrentState { handle, spec })
                }
            });
        }
        // Shape/duplicate validation precedes the one atomic authority grant.
        // Failure therefore cannot publish a partial rollback that wakes itself.
        let state = InferenceStateSet::new(states)?;
        let reservation = self.grant(bytes)?;
        for value in state.states() {
            self.insert(value.handle(), Arc::clone(&reservation));
        }
        Ok(state)
    }

    /// Validate allocation provenance for every component of a set.
    ///
    /// # Errors
    ///
    /// Returns an error for stale, released or foreign allocation handles.
    pub fn validate(&self, state: &InferenceStateSet) -> Result<(), StateError> {
        for value in state.states() {
            self.record_for(value.handle())?;
        }
        Ok(())
    }

    /// # Errors
    ///
    /// Returns [`StateError`] when storage cannot satisfy the KV allocation.
    pub fn allocate_kv(&mut self, spec: KvStateSpec) -> Result<KvState, StateError> {
        let handle = self.reserve(StateRequirement::FullAttentionKv(spec))?;
        KvState::new(handle, spec).ok_or(StateError::UnsupportedRequirement)
    }

    /// # Errors
    ///
    /// Returns [`StateError`] when storage cannot satisfy the recurrent allocation.
    pub fn allocate_recurrent(
        &mut self,
        spec: RecurrentStateSpec,
    ) -> Result<RecurrentState, StateError> {
        let handle = self.reserve(StateRequirement::Recurrent(spec))?;
        RecurrentState::new(handle, spec).ok_or(StateError::UnsupportedRequirement)
    }

    /// Commit a batch of positions atomically. On error, both manager records and
    /// every supplied position remain unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid handles, regressing positions or mismatched lengths.
    pub fn commit_batch(
        &mut self,
        states: &mut [InferenceStateSet],
        token_positions: &[u32],
    ) -> Result<(), StateError> {
        if states.len() != token_positions.len() {
            return Err(StateError::PositionMismatch);
        }
        for (state, &position) in states.iter().zip(token_positions) {
            for value in state.states() {
                let record = self.record_for(value.handle())?;
                if position < record.token_position {
                    return Err(StateError::PositionRegression);
                }
            }
        }
        for (state, &position) in states.iter_mut().zip(token_positions) {
            for value in &mut state.states {
                self.allocations
                    .get_mut(&value.handle().id())
                    .ok_or(StateError::InvalidHandle)?
                    .token_position = position;
                value.set_position(position);
            }
        }
        Ok(())
    }

    /// Advance one state set without relinquishing ownership on failure.
    ///
    /// # Errors
    ///
    /// Returns an error when the transition cannot be committed.
    pub fn commit(
        &mut self,
        state: &mut InferenceStateSet,
        token_position: u32,
    ) -> Result<(), StateError> {
        self.commit_batch(std::slice::from_mut(state), &[token_position])
    }

    /// Release every allocation in a set, retaining all of them on error.
    ///
    /// # Errors
    ///
    /// Returns an error when an allocation cannot be released.
    pub fn release_set(&mut self, state: &InferenceStateSet) -> Result<(), StateError> {
        for value in state.states() {
            self.record_for(value.handle())?;
        }
        for value in state.states() {
            self.release(value.handle())?;
        }
        Ok(())
    }

    /// # Errors
    ///
    /// Returns [`StateError::InvalidHandle`] when the handle is unknown or was released.
    pub fn release(&mut self, handle: &StateHandle) -> Result<(), StateError> {
        self.record_for(handle)?;
        self.allocations.remove(&handle.id());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allocate(manager: &mut LogicalStateManager) -> InferenceStateSet {
        let spec = KvStateSpec::new(1, 1, 2, 4, DataType::F16).expect("spec");
        let kv = manager.allocate_kv(spec).expect("allocation");
        InferenceStateSet::try_new(Some(kv), None).expect("set")
    }

    fn recurrent_requirement() -> StateRequirement {
        let matrix = RecurrentMatrixShape::new(2, 4, 4).expect("matrix");
        let convolution = ConvolutionStateShape::new(8, 3).expect("convolution");
        StateRequirement::Recurrent(
            RecurrentStateSpec::new(2, matrix, convolution, DataType::F16, DataType::F32)
                .expect("recurrent spec"),
        )
    }

    #[test]
    fn a_declaration_covers_the_pages_its_sequences_can_reach() {
        // Page geometry is the model's declaration; a sequence holds whole pages.
        let declared = KvStateSpec::new(2, 4, 8, 16, DataType::F16).expect("declared spec");
        let page_bytes = declared.page_bytes().expect("page bytes");
        assert_eq!(declared.capacity_tokens(), 16);
        assert_eq!(declared.pages_for(16), Some(1));
        assert_eq!(declared.pages_for(17), Some(2));
        assert_eq!(declared.pages_for(64), Some(4));
        assert_eq!(declared.pages_for(0), None);
        assert!(matches!(
            KvStateSpec::with_pages(1, 1, 2, 2, u32::MAX, DataType::F16),
            Err(StateSpecError::CapacityOverflow)
        ));

        let short = declared.for_tokens(16).expect("short reach");
        let long = declared.for_tokens(64).expect("long reach");
        assert_eq!(short.byte_size(), page_bytes.checked_mul(1));
        assert_eq!(long.byte_size(), page_bytes.checked_mul(4));
        assert!(
            StateRequirement::FullAttentionKv(short)
                .within(StateRequirement::FullAttentionKv(long))
        );
        assert!(
            !StateRequirement::FullAttentionKv(long)
                .within(StateRequirement::FullAttentionKv(short)),
            "capacity above the bound is not covered"
        );

        // A different page size, shape or dtype is a different declaration.
        let other_page = KvStateSpec::new(2, 4, 8, 32, DataType::F16).expect("other page");
        let wider = KvStateSpec::new(2, 4, 8, 16, DataType::F32).expect("wider dtype");
        let deeper = KvStateSpec::new(3, 4, 8, 16, DataType::F16).expect("deeper");
        for incompatible in [other_page, wider, deeper] {
            assert!(!StateRequirement::FullAttentionKv(short).within(
                StateRequirement::FullAttentionKv(incompatible.for_tokens(16).expect("concrete"))
            ));
        }

        // A recurrent summary has no capacity dimension: it is the same requirement
        // however far the sequence reached, and it never covers a KV declaration.
        let recurrent = recurrent_requirement();
        assert_eq!(recurrent.for_tokens(16).expect("unchanged"), recurrent);
        assert!(!recurrent.within(StateRequirement::FullAttentionKv(long)));
        assert!(!StateRequirement::FullAttentionKv(short).within(recurrent));
        assert!(recurrent.within(recurrent));
        assert!(matches!(
            recurrent.for_tokens(0),
            Ok(StateRequirement::Recurrent(_))
        ));
        assert!(matches!(
            StateRequirement::FullAttentionKv(declared).for_tokens(0),
            Err(StateSpecError::ZeroDimension)
        ));
    }

    #[test]
    fn bundle_reservation_is_atomic_and_failure_publishes_nothing() {
        let kv =
            StateRequirement::FullAttentionKv(KvStateSpec::new(1, 1, 2, 4, DataType::F16).unwrap());
        let recurrent = recurrent_requirement();
        let total = kv.byte_size().unwrap() + recurrent.byte_size().unwrap();
        let pool = BytePool::new(total - 1).shared();
        let mut manager = LogicalStateManager::new(DeviceId::new(0), Arc::clone(&pool));
        let wait = pool.capacity_wait();
        assert!(matches!(
            manager.allocate_set(&[kv, recurrent]),
            Err(StateError::CapacityExceeded { .. })
        ));
        assert_eq!(
            manager.allocate_set(&[kv, kv]).unwrap_err(),
            StateError::DuplicateRequirement
        );
        assert_eq!(pool.granted(), 0);
        assert!(
            !wait.changed(),
            "neither aggregate refusal nor validation rolls back a partial grant"
        );
        let state = manager.allocate_set(&[kv]).unwrap();
        manager.release_set(&state).unwrap();
        assert_eq!(pool.granted(), 0);
    }

    #[test]
    fn bundle_charge_survives_partial_component_release_and_invalid_set_release() {
        let requirements = [
            StateRequirement::FullAttentionKv(KvStateSpec::new(1, 1, 2, 4, DataType::F16).unwrap()),
            recurrent_requirement(),
        ];
        let total = requirements.iter().map(|r| r.byte_size().unwrap()).sum();
        let pool = BytePool::new(total).shared();
        let mut manager = LogicalStateManager::new(DeviceId::new(0), Arc::clone(&pool));
        let state = manager.allocate_set(&requirements).unwrap();
        let wait = pool.capacity_wait();
        manager.release(state.kv().unwrap().handle()).unwrap();
        assert_eq!(
            pool.granted(),
            total,
            "one bundle grant stays with its surviving component"
        );
        assert!(!wait.changed());
        assert_eq!(manager.release_set(&state), Err(StateError::InvalidHandle));
        assert_eq!(pool.granted(), total);
        manager
            .release(state.recurrent().unwrap().handle())
            .unwrap();
        assert_eq!(pool.granted(), 0);
        assert!(wait.changed());
    }

    #[test]
    fn supplied_authority_counts_siblings_and_closes_without_refunding_live_state() {
        let device = DeviceId::new(0);
        let pool = BytePool::new(64).shared();
        let mut first = LogicalStateManager::new(device, Arc::clone(&pool));
        let mut peer = LogicalStateManager::new(device, Arc::clone(&pool));
        let state = allocate(&mut first);
        let sibling = pool.reserve(32).unwrap();
        assert_eq!(first.used_bytes(), 64);
        assert_eq!(peer.used_bytes(), 64);
        let wait = peer.capacity_wait();
        assert!(matches!(
            peer.allocate_set(&state.requirements()),
            Err(StateError::CapacityExceeded { .. })
        ));
        assert!(!wait.changed());
        drop(sibling);
        assert!(wait.changed());
        let peer_state = peer.allocate_set(&state.requirements()).unwrap();
        assert_ne!(
            peer_state.kv().unwrap().handle().id(),
            state.kv().unwrap().handle().id()
        );
        let wait = peer.capacity_wait();
        pool.close();
        assert!(wait.changed());
        assert_eq!(pool.granted(), 64);
        assert_eq!(
            peer.allocate_set(&state.requirements()).unwrap_err(),
            StateError::PoolClosed
        );
        assert_eq!(peer.release_set(&state), Err(StateError::InvalidHandle));
        first.release_set(&state).unwrap();
        peer.release_set(&peer_state).unwrap();
        assert_eq!(pool.granted(), 0);
    }

    #[test]
    fn dropping_an_unretired_logical_owner_does_not_refund_its_shared_charge() {
        let device = DeviceId::new(0);
        let pool = BytePool::new(32).shared();
        let mut manager = LogicalStateManager::new(device, Arc::clone(&pool));
        let state = allocate(&mut manager);
        drop(state);
        drop(manager);
        assert_eq!(
            pool.granted(),
            32,
            "logical disappearance cannot prove physical reuse"
        );
        assert!(matches!(
            pool.reserve(1),
            Err(ReserveError::Exhausted { .. })
        ));
    }

    #[test]
    fn independent_managers_reject_each_others_allocations() {
        let mut first = LogicalStateManager::new(
            DeviceId::new(0),
            ribn_foundation::BytePool::new(1024).shared(),
        );
        let mut second = LogicalStateManager::new(
            DeviceId::new(0),
            ribn_foundation::BytePool::new(1024).shared(),
        );
        let mut first_state = allocate(&mut first);
        let second_state = allocate(&mut second);
        assert_ne!(
            first_state.states()[0].handle().id(),
            second_state.states()[0].handle().id()
        );
        assert_eq!(
            second.commit(&mut first_state, 1),
            Err(StateError::InvalidHandle)
        );
        assert_eq!(
            second.release_set(&first_state),
            Err(StateError::InvalidHandle)
        );
        second
            .validate(&second_state)
            .expect("own allocation remains valid");
        first
            .validate(&first_state)
            .expect("foreign rejection leaves owner valid");
        first.release_set(&first_state).unwrap();
        second.release_set(&second_state).unwrap();
    }

    #[test]
    fn batch_commit_prevalidates_all_states_before_advancing_any() {
        let mut manager = LogicalStateManager::new(
            DeviceId::new(0),
            ribn_foundation::BytePool::new(1024).shared(),
        );
        let first = allocate(&mut manager);
        let mut second = allocate(&mut manager);
        manager.commit(&mut second, 3).expect("advance second");
        let mut states = vec![first, second];
        assert_eq!(
            manager.commit_batch(&mut states, &[1, 2]),
            Err(StateError::PositionRegression)
        );
        assert_eq!(states[0].token_position(), Some(0));
        assert_eq!(states[1].token_position(), Some(3));
        for state in &states {
            manager.validate(state).expect("unchanged record");
        }
        manager.commit_batch(&mut states, &[1, 4]).expect("retry");
        assert_eq!(states[0].token_position(), Some(1));
        assert_eq!(states[1].token_position(), Some(4));
        for state in &states {
            manager.release_set(state).unwrap();
        }
    }
}
