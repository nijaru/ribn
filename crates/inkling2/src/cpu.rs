//! Correctness-first pure-Rust CPU reference forward for Inkling 2 / Inkling 2 Loop.
//!
//! Keeps quantized GGUF tensors (`Q8_0`, `IQ4_NL`, `F16`, `F32`) in their compact
//! encoded representation and executes every model-local operator directly in
//! safe Rust for deterministic verification against pinned reference logits.

pub mod model;
pub mod ops;
pub mod weights;

pub use model::{Inkling2CpuModel, Inkling2CpuState, LayerTap};
pub use weights::{QuantizedMatrix, QuantizedTensor3d, f16_to_f32, f32_to_f16};
