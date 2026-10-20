//! Apple Metal BF16 compute primitives.
//!
//! This backend targets the Apple GPU (Metal), not the Neural Engine: batch-decode
//! work is bandwidth-bound and needs dynamic shapes and custom kernels, which the
//! CoreML-only ANE path does not provide. Only macOS builds contain code; other
//! targets compile an empty crate so the workspace stays portable.

#[cfg(target_os = "macos")]
mod bf16;

#[cfg(target_os = "macos")]
pub use bf16::{
    MetalAttentionConfig, MetalBf16Buffer, MetalBf16Error, MetalBf16Executor, MetalLinearShape,
    MetalProjectionNorm, MetalRotaryFactors,
};
