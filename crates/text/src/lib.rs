//! Reusable text-generation frontend for Ribn.
//!
//! This crate owns text input formatting, tokenization, incremental decoding,
//! bounded offline batching, and ordinary generation results. The scheduler and
//! token runtime below it see only encoded tokens and know nothing about chat
//! messages or UTF-8 streams.

#[cfg(feature = "cuda")]
mod cuda;
mod error;
mod gemma4;
#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
mod gemma_load;
mod gguf;
mod input;
mod model;
mod process;
mod stream;

#[cfg(feature = "cuda")]
pub use cuda::{LoadOptions, MemoryReport};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use engine_gemma4::Gemma4MetalFfn;
pub use error::TextError;
#[cfg(any(all(feature = "metal", target_os = "macos"), feature = "cuda"))]
pub use gemma_load::GemmaLoadReport;
pub use gemma4::GemmaProcessor;
pub use gguf::GgufProcessor;
pub use input::{Message, TextInput};
pub use model::{TextBatch, TextConfig, TextModel, TextOwner, TextRequest, TextResponse};
pub use process::{EncodedInput, ProcessorLimits, TextDecoder, TextProcessor};
pub use ribn::{FinishReason, GenerationOptions, Sampling, Usage};
pub use stream::{MAX_STOP_BYTES, MAX_STOP_SEQUENCES, TextEvent, TextStream};
