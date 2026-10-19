//! Model-owned BF16 materialization and compute bindings. No decoder or token executor.

mod linear;
mod norm;

use crate::{Gemma4Config, Gemma4Error, Gemma4Source, ParameterId};
use cudarc::driver::CudaStream;
use engine_core::{DataType, WeightTensorSpec};
use engine_nvidia::{CudaBf16Error, CudaBf16Weights};
use ribn_foundation::{AllocationId, BytePool};
use std::{collections::BTreeMap, error::Error, fmt, sync::Arc};

/// Model interpretation or physical preparation failure.
#[derive(Debug)]
pub enum Gemma4CudaError {
    Model(Gemma4Error),
    Backend(CudaBf16Error),
}
impl fmt::Display for Gemma4CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(e) => e.fmt(f),
            Self::Backend(e) => e.fmt(f),
        }
    }
}
impl Error for Gemma4CudaError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(match self {
            Self::Model(e) => e,
            Self::Backend(e) => e,
        })
    }
}

/// A load error retains any partial device materialization for explicit cleanup.
/// Dropping the diagnostic does not assert device completion: the backend's
/// defensive retirement/quarantine remains responsible for uncertain storage.
pub struct Gemma4CudaLoadError {
    cause: Gemma4CudaError,
    retirement: Option<Box<CudaBf16Weights>>,
}
impl Gemma4CudaLoadError {
    #[must_use]
    pub const fn cause(&self) -> &Gemma4CudaError {
        &self.cause
    }
    #[must_use]
    pub const fn retirement_pending(&self) -> bool {
        self.retirement.is_some()
    }
    /// Retire partial construction. Failed cleanup keeps the same owner and charge.
    /// # Errors
    /// Returns a drain/free/quarantine error; never retry loading into this set.
    pub fn retire(&mut self) -> Result<(), CudaBf16Error> {
        if let Some(weights) = &mut self.retirement {
            weights.close()?;
        }
        self.retirement = None;
        Ok(())
    }
    fn model(error: Gemma4Error) -> Self {
        Self {
            cause: Gemma4CudaError::Model(error),
            retirement: None,
        }
    }
    fn backend(error: CudaBf16Error, retirement: Option<Box<CudaBf16Weights>>) -> Self {
        Self {
            cause: Gemma4CudaError::Backend(error),
            retirement,
        }
    }
}
impl fmt::Debug for Gemma4CudaLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gemma4CudaLoadError")
            .field("cause", &self.cause)
            .field("retirement_pending", &self.retirement_pending())
            .finish_non_exhaustive()
    }
}
impl fmt::Display for Gemma4CudaLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl Error for Gemma4CudaLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.cause)
    }
}

/// One complete, immutable BF16 text weight set. The model owns parameter-role
/// interpretation; NVIDIA owns layout, transfer completion and physical retirement.
/// This is not executable Gemma support or a whole-inference resource bound.
pub struct Gemma4CudaWeights {
    config: Gemma4Config,
    slots: BTreeMap<ParameterId, usize>,
    storage: Box<CudaBf16Weights>,
}
impl Gemma4CudaWeights {
    /// Materialize only the source's fully preflighted active text inventory.
    /// Readers retain their selected files; no paths are reopened and tied weights
    /// are uploaded once. Keep selected files unchanged through every alias.
    /// # Errors
    /// The error retains any partial storage for explicit retirement. Allocation
    /// refusal has no partial materialization; upload failure never publishes a set.
    /// # Panics
    /// Only if NVIDIA violates its complete-upload publication invariant.
    pub fn load(
        source: &Gemma4Source,
        stream: Arc<CudaStream>,
        pool: &Arc<BytePool>,
        staging_bytes: usize,
    ) -> Result<Self, Gemma4CudaLoadError> {
        let mut slots = BTreeMap::new();
        let mut specs = Vec::with_capacity(source.parameters().len());
        for (&id, parameter) in source.parameters() {
            let spec = parameter.spec();
            let dimensions = spec
                .shape
                .iter()
                .map(|&n| {
                    u64::try_from(n).map_err(|_| {
                        Gemma4CudaLoadError::model(Gemma4Error::new(
                            &spec.name,
                            "dimension exceeds u64",
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let descriptor = WeightTensorSpec::new(&spec.name, dimensions, DataType::BF16)
                .map_err(|e| {
                    Gemma4CudaLoadError::model(Gemma4Error::new(&spec.name, e.to_string()))
                })?;
            slots.insert(id, specs.len());
            specs.push(descriptor);
        }
        let config = source.config().clone();
        let mut storage = Box::new(
            CudaBf16Weights::allocate(specs, stream, pool, staging_bytes)
                .map_err(|e| Gemma4CudaLoadError::backend(e, None))?,
        );
        for (slot, parameter) in source.parameters().values().enumerate() {
            if let Err(error) = storage.upload(slot, &mut parameter.reader()) {
                return Err(Gemma4CudaLoadError::backend(error, Some(storage)));
            }
        }
        assert!(
            storage.ready(),
            "every preflighted parameter uploaded successfully"
        );
        Ok(Self {
            config,
            slots,
            storage,
        })
    }
    #[must_use]
    pub const fn config(&self) -> &Gemma4Config {
        &self.config
    }
    /// Logical role to physical slot binding; metadata is not a lifetime grant.
    #[must_use]
    pub fn parameter_slot(&self, id: ParameterId) -> Option<usize> {
        self.slots.get(&id).copied()
    }
    #[must_use]
    #[expect(
        clippy::missing_panics_doc,
        reason = "private complete-source binding invariant"
    )]
    pub fn token_embedding_slot(&self) -> usize {
        *self
            .slots
            .get(&ParameterId::TokenEmbedding)
            .expect("complete source contains the embedding")
    }
    /// Exact same physical slot, never a separately scaled/materialized head.
    #[must_use]
    pub fn output_projection_slot(&self) -> usize {
        self.token_embedding_slot()
    }
    #[must_use]
    pub fn allocation_id(&self) -> AllocationId {
        self.storage.allocation_id()
    }
    #[must_use]
    pub fn allocated_bytes(&self) -> u64 {
        self.storage.allocated_bytes()
    }
    /// Bounded, completion-safe diagnostic readback; returned bytes are caller-owned.
    /// # Errors
    /// Rejects missing roles, closed/poisoned sets, invalid ranges or transfer failure.
    pub fn read_chunk(
        &mut self,
        id: ParameterId,
        offset: u64,
        len: usize,
    ) -> Result<Vec<u8>, Gemma4CudaError> {
        let slot = self.parameter_slot(id).ok_or_else(|| {
            Gemma4CudaError::Model(Gemma4Error::new(
                "parameter",
                format!("{id:?} has no active allocation"),
            ))
        })?;
        self.storage
            .read_chunk(slot, offset, len)
            .map_err(Gemma4CudaError::Backend)
    }
    /// Establish retirement before releasing this set's pool charge. Idempotent.
    /// # Errors
    /// Failed drain retains a retry owner; uncertain free remains quarantined.
    pub fn close(&mut self) -> Result<(), CudaBf16Error> {
        self.storage.close()
    }
}
