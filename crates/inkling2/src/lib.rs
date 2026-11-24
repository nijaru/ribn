//! Model-local Inkling 2 / Inkling 2 Loop text configuration, GGUF inventory,
//! continuation geometry, and correctness-first CPU reference forward.
//!
//! Loop pass weight sharing, cross-pass depth banks, relative-position bias,
//! short causal convolutions, and `MoE` routing remain local to this crate rather
//! than leaking into request scheduling or unrelated model loaders.

pub mod config;
pub mod continuation;
pub mod cpu;
pub mod executor;
pub mod parameters;
pub mod source;

pub use config::{
    ArchitectureKind, AttentionKind, ExecutionStep, Inkling2Config, LoopConfig,
    PhysicalLayerGeometry,
};
pub use continuation::{ContinuationStep, KvCachePrecision, SlotContinuationStep};
pub use cpu::{Inkling2CpuModel, Inkling2CpuState};
pub use executor::Inkling2Executor;
pub use parameters::{
    GlobalParameter, LayerParameter, LoopController, LoopParameter, ParameterId, ParameterSpec,
};
pub use source::{Inkling2Source, TensorEncoding};

use std::fmt;

/// Typed error returned by Inkling 2 configuration, preflight, continuation, or execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Inkling2Error {
    context: String,
    reason: String,
}

impl Inkling2Error {
    #[must_use]
    pub fn new(context: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            context: context.into(),
            reason: reason.into(),
        }
    }

    #[must_use]
    pub fn context(&self) -> &str {
        &self.context
    }

    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for Inkling2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "inkling2 {}: {}", self.context, self.reason)
    }
}

impl std::error::Error for Inkling2Error {}
