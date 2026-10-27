//! Model-local Gemma 4 vision encoder and projection, independent of token scheduling.
mod config;
mod encoder;
mod ops;
mod parameters;
mod patches;
mod source;
pub use config::Gemma4VisionConfig;
pub use encoder::{Gemma4CpuVisionEncoder, VisionImage, VisionStage};
pub use parameters::{
    VisionLinear, VisionLinearPart, VisionNorm, VisionParameterId, VisionParameterSpec,
};
pub use patches::Gemma4ImagePatches;
pub use source::{Gemma4VisionCpuWeights, Gemma4VisionSource, StoredVisionParameter};
