//! Host BF16 execution primitives for the CPU backend.
//!
//! These are concrete semantics over prepared host bytes, not a portable tensor or
//! operator framework. They allocate ordinary heap vectors, need no device or pool,
//! and are the arithmetic a CPU decoder step composes. Model roles stay in the
//! caller; the backend only sees shapes and native BF16 encodings.

mod attention;
mod decoder;
mod elementwise;
mod linear;
mod norm;
mod rotary;
mod weights;

pub use attention::{AttentionConfig, attention, attention_with_weights};
pub use decoder::{Gemma4CpuDecoder, argmax};
pub use elementwise::{add, gelu_tanh, multiply, ple_combine, scale, softcap};
pub use linear::{bf16, linear, linear_f32, project};

/// Fallible host allocation, so a model-sized request returns an error instead of aborting.
pub(crate) fn try_vector<T: Clone>(length: usize, value: T) -> Result<Vec<T>, crate::Gemma4Error> {
    let mut vector = Vec::new();
    vector
        .try_reserve_exact(length)
        .map_err(|_| crate::Gemma4Error::new("CPU allocation", "host allocation refused"))?;
    vector.resize(length, value);
    Ok(vector)
}
pub use norm::{rms_norm, rms_norm_scale_free};
pub use rotary::{apply_rotary, factors, inverse_frequencies};
pub use weights::{CpuParameter, Gemma4CpuWeights};
