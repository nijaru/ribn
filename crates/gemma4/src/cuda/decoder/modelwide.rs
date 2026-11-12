//! Depth-one model-wide physical owner, separate from per-sequence decoders.
use super::{
    Arc, BytePool, CudaBf16ComputeError, CudaBf16Error, DeviceCache, Gemma4Config, Gemma4Error,
    Gemma4Source, HostLookups, backend, buffers, project_logits, run_rows,
};
use buffers::CacheConstruction;
use cudarc::driver::CudaContext;
use engine_nvidia::{CudaBf16Commands, CudaBf16Stream, CudaBf16Submission};
use ribn::{Admission, ExecutionError, SequenceId};
use std::{collections::HashMap, error::Error, fmt, mem::ManuallyDrop};

mod executor;
mod packed;
mod preparation;
pub use executor::Gemma4CudaExecutor;
use preparation::{Construction, Preparation, Prepared};

const MAX_SEQUENCES: usize = 4;
struct PhysicalSequence {
    caches: Vec<Option<DeviceCache>>,
    capacity: usize,
    position: usize,
    retired: bool,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Pending,
    Failed,
    Closed,
}
struct Physical {
    model: Prepared,
    states: HashMap<SequenceId, PhysicalSequence>,
    construction: Option<CacheConstruction>,
    commands: Option<CudaBf16Commands>,
    pending: Option<CudaBf16Submission>,
    members: Option<[SequenceId; 2]>,
    failure: Option<CudaBf16ComputeError>,
    phase: Phase,
}

/// Model-wide CUDA execution. Shared weights, cache ownership and batch workspace
/// live here, not in a collection of per-request decoder handles. The ordinary CUDA
/// text loader limits automatic selection to the qualified device, normalized
/// configuration and prepared context; other configurations retain serial execution.
pub struct Gemma4CudaModelWide {
    physical: Option<ManuallyDrop<Box<Physical>>>,
}
impl Gemma4CudaModelWide {
    /// Match the device, normalized configuration and prepared context used by
    /// model-wide qualification, independently of scratch-selected prefill lanes.
    ///
    /// This pure query neither loads resources nor enables automatic selection.
    /// A match is not a quality guarantee for arbitrary checkpoint weights; the
    /// recorded full-model evidence remains scoped to its pinned artifact.
    #[must_use]
    pub fn matches_qualification_scope(
        config: &Gemma4Config,
        device: &str,
        capacity: usize,
    ) -> bool {
        super::preparation::modelwide_scope(config, device, capacity)
    }

    /// Prepare native resources on a private, uncapturable stream domain.
    /// # Errors
    /// Returns an owning diagnostic for failed partial physical construction.
    pub fn load(
        source: &Gemma4Source,
        context: &Arc<CudaContext>,
        pool: &Arc<BytePool>,
        capacity: usize,
        scratch_entries: usize,
    ) -> Result<Self, Gemma4CudaModelWideLoadError> {
        let device = context.name().map_err(|e| {
            Gemma4CudaModelWideLoadError::plain(Gemma4Error::new(
                "CUDA model-wide device",
                e.to_string(),
            ))
        })?;
        let plan = super::preparation::prepare(
            source.config(),
            capacity,
            scratch_entries,
            &device,
            &super::preparation::MODEL_WIDE_PREFILL_LANES,
        )
        .map_err(Gemma4CudaModelWideLoadError::plain)?;
        let host = HostLookups::load(source).map_err(Gemma4CudaModelWideLoadError::plain)?;
        let stream = CudaBf16Stream::new(context)
            .map_err(|e| Gemma4CudaModelWideLoadError::plain(model_error(&e)))?;
        let mut preparation = Construction::new(
            Preparation::new(
                host,
                stream,
                pool,
                capacity,
                plan.rows,
                plan.scratch_entries,
            )
            .map_err(Gemma4CudaModelWideLoadError::plain)?,
        );
        if let Err(cause) = preparation.build(source) {
            return Err(Gemma4CudaModelWideLoadError {
                cause,
                retirement: Some(preparation),
            });
        }
        Ok(Self {
            physical: Some(ManuallyDrop::new(Box::new(Physical {
                model: preparation.finish(),
                states: HashMap::with_capacity(MAX_SEQUENCES),
                construction: None,
                commands: None,
                pending: None,
                members: None,
                failure: None,
                phase: Phase::Idle,
            }))),
        })
    }
    fn owner(&self) -> &Physical {
        self.physical.as_ref().expect("live physical owner")
    }
    fn owner_mut(&mut self) -> &mut Physical {
        self.physical.as_mut().expect("live physical owner")
    }
    #[must_use]
    pub fn config(&self) -> &Gemma4Config {
        &self.owner().model.host.config
    }
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.owner().model.capacity
    }
    #[must_use]
    pub fn position(&self, sequence: SequenceId) -> Option<usize> {
        self.owner()
            .states
            .get(&sequence)
            .filter(|state| !state.retired)
            .map(|s| s.position)
    }

    /// Construct a request-reachable cache under this execution owner.
    /// # Errors
    /// Uncertain construction/cleanup faults the owner; clean pool refusal is local.
    /// # Panics
    /// Only if an internal construction ownership invariant is violated.
    pub fn admit(
        &mut self,
        sequence: SequenceId,
        capacity: usize,
    ) -> Result<Admission, Gemma4Error> {
        let owner = self.owner_mut();
        owner.idle()?;
        if owner.states.contains_key(&sequence) {
            return Err(Gemma4Error::new(
                "CUDA physical admission",
                "duplicate sequence incarnation",
            ));
        }
        if capacity == 0 || capacity > owner.model.capacity || owner.states.len() == MAX_SEQUENCES {
            return Ok(Admission::Rejected(ExecutionError::new(
                "physical continuation exceeds prepared limits",
            )));
        }
        owner.phase = Phase::Failed;
        owner.construction = Some(CacheConstruction::new(
            owner.model.host.config.layers().len(),
        ));
        let result = owner
            .construction
            .as_mut()
            .expect("installed cache construction")
            .build_with(&owner.model.host.config, capacity, |rows, cols, staging| {
                owner
                    .model
                    .stream
                    .matrix(rows, cols, &owner.model.pool, staging)
            });
        if let Err(error) = result {
            let local = matches!(&error, CudaBf16Error::Reserve(_));
            owner.failure = Some(error.into());
            if local
                && owner
                    .construction
                    .as_mut()
                    .expect("retained construction")
                    .close()
                    .is_ok()
            {
                owner.construction = None;
                let error = owner.failure.take().expect("stored reservation refusal");
                owner.phase = Phase::Idle;
                return Ok(Admission::Rejected(ExecutionError::new(error.to_string())));
            }
            return Err(backend(
                owner
                    .failure
                    .as_ref()
                    .expect("retained construction failure"),
            ));
        }
        let caches = owner
            .construction
            .take()
            .expect("complete cache construction")
            .caches;
        owner.states.insert(
            sequence,
            PhysicalSequence {
                caches,
                capacity,
                position: 0,
                retired: false,
            },
        );
        owner.phase = Phase::Idle;
        Ok(Admission::Ready)
    }

    /// Qualified single-request prefill/scalar fallback on the same owned resources.
    /// # Errors
    /// Rejects invalid input before mutation; physical failures permanently fault the owner.
    pub fn prefill(
        &mut self,
        sequence: SequenceId,
        tokens: &[u32],
        readout: bool,
    ) -> Result<Option<Vec<u16>>, Gemma4Error> {
        let owner = self.owner_mut();
        owner.idle()?;
        owner.model.host.config.validate_text_tokens(tokens)?;
        let state = owner
            .states
            .get_mut(&sequence)
            .filter(|state| !state.retired)
            .ok_or_else(|| Gemma4Error::new("CUDA physical prefill", "unknown sequence"))?;
        let end = state
            .position
            .checked_add(tokens.len())
            .filter(|&n| !tokens.is_empty() && n <= state.capacity)
            .ok_or_else(|| Gemma4Error::new("CUDA physical prefill", "invalid accepted range"))?;
        owner.phase = Phase::Failed;
        let result = (|| {
            let Prepared {
                host,
                cuda,
                executors,
                scalar,
                prefill,
                factors,
                ..
            } = &mut owner.model;
            let mut offset = 0;
            for buffers in prefill.iter_mut().rev() {
                let rows = buffers.hidden.shape().0;
                while tokens.len() - offset >= rows {
                    run_rows(
                        &host.config,
                        host,
                        cuda,
                        executors,
                        buffers,
                        factors,
                        &mut state.caches,
                        state.position + offset,
                        &tokens[offset..offset + rows],
                    )?;
                    offset += rows;
                }
                if offset == tokens.len() {
                    return if readout {
                        project_logits(&host.config, cuda, executors, buffers).map(Some)
                    } else {
                        Ok(None)
                    };
                }
            }
            for token in &tokens[offset..] {
                run_rows(
                    &host.config,
                    host,
                    cuda,
                    executors,
                    scalar,
                    factors,
                    &mut state.caches,
                    state.position + offset,
                    std::slice::from_ref(token),
                )?;
                offset += 1;
            }
            if readout {
                project_logits(&host.config, cuda, executors, scalar).map(Some)
            } else {
                Ok(None)
            }
        })();
        match result {
            Ok(logits) => {
                state.position = end;
                owner.phase = Phase::Idle;
                Ok(logits)
            }
            Err(error) => Err(owner.fault(error)),
        }
    }

    /// Enqueue a complete two-sequence forward, with independent positions/KV.
    /// No sequence position commits until terminal acknowledgement and readout.
    /// # Errors
    /// Whole-pair validation precedes device mutation; physical failure is owner-fatal.
    /// # Panics
    /// Only if an internal command ownership invariant is violated.
    pub fn start_pair(
        &mut self,
        sequences: [SequenceId; 2],
        tokens: [u32; 2],
    ) -> Result<(), Gemma4Error> {
        let owner = self.owner_mut();
        owner.idle()?;
        owner.model.host.config.validate_text_tokens(&tokens)?;
        if sequences[0] == sequences[1] {
            return Err(Gemma4Error::new(
                "CUDA physical pair",
                "duplicate sequence incarnation",
            ));
        }
        let [Some(first), Some(second)] = owner
            .states
            .get_disjoint_mut([&sequences[0], &sequences[1]])
        else {
            return Err(Gemma4Error::new("CUDA physical pair", "unknown sequence"));
        };
        if first.retired
            || second.retired
            || first.position >= first.capacity
            || second.position >= second.capacity
        {
            return Err(Gemma4Error::new(
                "CUDA physical pair",
                "cache reach exhausted",
            ));
        }
        let position = |value| {
            u32::try_from(value)
                .map_err(|_| Gemma4Error::new("CUDA physical pair", "position exceeds u32"))
        };
        let positions = [position(first.position)?, position(second.position)?];
        owner.phase = Phase::Failed;
        let result = (|| {
            let Prepared {
                host,
                cuda,
                executors,
                packed,
                factors,
                stream,
                ..
            } = &mut owner.model;
            let limit = host
                .config
                .layers()
                .len()
                .checked_mul(64)
                .and_then(|n| n.checked_add(8))
                .ok_or(CudaBf16Error::Invalid("model command bound overflow"))?;
            owner.commands = Some(CudaBf16Commands::new(stream, limit)?);
            packed.prepare(host, cuda, executors, factors, tokens, positions)?;
            packed.enqueue(
                host,
                cuda,
                executors,
                [first, second],
                owner.commands.as_mut().expect("installed command owner"),
            )?;
            owner.pending = Some(
                owner
                    .commands
                    .take()
                    .expect("recorded command owner")
                    .submit()?,
            );
            Ok::<_, CudaBf16ComputeError>(())
        })();
        match result {
            Ok(()) => {
                owner.members = Some(sequences);
                owner.phase = Phase::Pending;
                Ok(())
            }
            Err(error) => Err(owner.fault(error)),
        }
    }

    /// Query without waiting; acknowledge and read both complete rows only when ready.
    /// # Errors
    /// Terminal/query/readout failures retain this owner for explicit retirement.
    /// # Panics
    /// Only if an internal submitted-state ownership invariant is violated.
    pub fn poll_pair(&mut self) -> Result<Option<[Vec<u16>; 2]>, Gemma4Error> {
        let owner = self.owner_mut();
        if owner.phase != Phase::Pending {
            return Err(Gemma4Error::new("CUDA physical pair", "no pending pair"));
        }
        owner.phase = Phase::Failed;
        let result = (|| {
            let pending = owner.pending.as_mut().expect("submitted pair");
            if !pending.poll()? {
                return Ok(None);
            }
            pending.retire()?;
            let logits = owner.model.packed.read(&owner.model.host.config)?;
            Ok::<_, CudaBf16ComputeError>(Some(logits))
        })();
        match result {
            Ok(None) => {
                owner.phase = Phase::Pending;
                Ok(None)
            }
            Ok(Some(logits)) => {
                // Unrelated cancellation never waits inside release(). Its actual
                // caches stay charged until this pair's acknowledgement.
                owner
                    .release_retired()
                    .map_err(|error| model_error(&error))?;
                for sequence in owner.members.take().expect("submitted member mapping") {
                    owner
                        .states
                        .get_mut(&sequence)
                        .expect("retained submitted state")
                        .position += 1;
                }
                owner.pending = None;
                owner.phase = Phase::Idle;
                Ok(Some(logits))
            }
            Err(error) => Err(owner.fault(error)),
        }
    }
    /// Release continuation without waiting for an unrelated pending pair.
    /// Such cancellation retains its charged caches until acknowledgement.
    /// # Errors
    /// Rejects submitted members/faulted owners; failed release retains caches.
    pub fn release(&mut self, sequence: SequenceId) -> Result<(), Gemma4Error> {
        let owner = self.owner_mut();
        if owner.phase == Phase::Closed {
            return Ok(());
        }
        if owner.phase == Phase::Pending {
            if owner
                .members
                .is_some_and(|members| members.contains(&sequence))
            {
                return Err(Gemma4Error::new(
                    "CUDA physical release",
                    "submitted member is still owned",
                ));
            }
            if let Some(state) = owner.states.get_mut(&sequence) {
                state.retired = true;
            }
            return Ok(());
        }
        owner.idle()?;
        owner.phase = Phase::Failed;
        if let Some(state) = owner.states.get_mut(&sequence) {
            for cache in state.caches.iter_mut().flatten() {
                cache.close().map_err(|e| model_error(&e))?;
            }
            owner.states.remove(&sequence);
        }
        owner.phase = Phase::Idle;
        Ok(())
    }
    /// Permanently close execution, retaining the same owner if any drain/free fails.
    /// # Errors
    /// Retry this operation after a failed settlement; execution never resumes.
    pub fn close(&mut self) -> Result<(), CudaBf16Error> {
        self.owner_mut().close()
    }
}
impl Physical {
    fn release_retired(&mut self) -> Result<(), CudaBf16Error> {
        for state in self.states.values_mut().filter(|state| state.retired) {
            for cache in state.caches.iter_mut().flatten() {
                cache.close()?;
            }
        }
        self.states.retain(|_, state| !state.retired);
        Ok(())
    }
    fn idle(&self) -> Result<(), Gemma4Error> {
        if self.phase == Phase::Idle {
            Ok(())
        } else {
            Err(Gemma4Error::new(
                "CUDA physical owner",
                "busy, faulted or closed",
            ))
        }
    }
    fn fault(&mut self, error: CudaBf16ComputeError) -> Gemma4Error {
        self.phase = Phase::Failed;
        self.failure = Some(error);
        backend(self.failure.as_ref().expect("stored physical failure"))
    }
    fn close(&mut self) -> Result<(), CudaBf16Error> {
        if self.phase == Phase::Closed {
            return Ok(());
        }
        self.phase = Phase::Failed;
        self.model.stream.synchronize()?;
        if let Some(commands) = &mut self.commands {
            commands.retire()?;
        }
        if let Some(pending) = &mut self.pending {
            pending.retire()?;
        }
        if let Some(error) = &mut self.failure {
            error.retire()?;
        }
        self.commands = None;
        self.pending = None;
        self.members = None;
        if let Some(construction) = &mut self.construction {
            construction.close()?;
        }
        self.construction = None;
        for state in self.states.values_mut() {
            for cache in state.caches.iter_mut().flatten() {
                cache.close()?;
            }
        }
        self.states.clear();
        self.model.close()?;
        self.phase = Phase::Closed;
        Ok(())
    }
}
impl Drop for Gemma4CudaModelWide {
    fn drop(&mut self) {
        if let Some(mut owner) = self.physical.take()
            && owner.close().is_ok()
        {
            drop(ManuallyDrop::into_inner(owner));
        }
    }
}

/// Failed model-wide construction retains the complete partial physical owner.
pub struct Gemma4CudaModelWideLoadError {
    cause: Gemma4Error,
    retirement: Option<Construction>,
}
impl Gemma4CudaModelWideLoadError {
    fn plain(cause: Gemma4Error) -> Self {
        Self {
            cause,
            retirement: None,
        }
    }
    #[must_use]
    pub fn cause(&self) -> &Gemma4Error {
        &self.cause
    }
    #[must_use]
    pub fn retirement_pending(&self) -> bool {
        self.retirement.is_some()
    }
    /// Retire partial preparation, retaining it on failure.
    /// # Errors
    /// Failed completion/free keeps this same owner available for retry.
    pub fn retire(&mut self) -> Result<(), CudaBf16Error> {
        if let Some(owner) = &mut self.retirement {
            owner.close()?;
        }
        self.retirement = None;
        Ok(())
    }
}
impl fmt::Display for Gemma4CudaModelWideLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl fmt::Debug for Gemma4CudaModelWideLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gemma4CudaModelWideLoadError")
            .field("cause", &self.cause)
            .field("retirement_pending", &self.retirement_pending())
            .finish_non_exhaustive()
    }
}
impl Error for Gemma4CudaModelWideLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.cause)
    }
}
fn model_error(error: &CudaBf16Error) -> Gemma4Error {
    Gemma4Error::new("CUDA model-wide execution", error.to_string())
}
