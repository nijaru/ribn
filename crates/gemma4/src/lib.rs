//! Gemma 4 dense, causal, BF16 text-source preparation.
//!
//! Configuration, parameter interpretation and continuation semantics belong here,
//! not in HF/SafeTensors adapters or token scheduling. This is host preflight, not
//! an executable decoder, multimodal processor or qualified CUDA implementation.

mod config;
mod continuation;
mod parameters;
mod source;

pub use config::{AttentionKind, Gemma4Config, LayerGeometry, Rope};
pub use continuation::{AttentionSpan, KvOwnerStep, KvStep, RetainedSpan};
pub use parameters::{LayerParameter, ParameterId, ParameterSpec};
pub use source::{ExcludedParameter, Exclusion, Gemma4Source, StoredParameter};

use std::fmt;

/// Invalid model semantics or source inventory. No partial source is published.
#[derive(Debug)]
pub struct Gemma4Error {
    pub field: String,
    pub reason: String,
}

impl Gemma4Error {
    pub(crate) fn new(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for Gemma4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gemma 4 {}: {}", self.field, self.reason)
    }
}
impl std::error::Error for Gemma4Error {}
