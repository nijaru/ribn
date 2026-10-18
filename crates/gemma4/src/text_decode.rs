//! Gemma's exact Replace → `ByteFallback` → Fuse pipeline. A fallback run cannot
//! be emitted early: one invalid trailing byte invalidates the entire run in HF.

use crate::{Gemma4Error, Gemma4TextProcessor};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Request-local token budget, retained fallback bytes and emitted fragment bytes.
/// These are independent of input/chat limits and caller-collected output.
#[derive(Clone, Copy, Debug)]
pub struct DecodeLimits {
    pub tokens: usize,
    /// Decoded vocabulary piece bytes, before a delayed run is joined to it.
    pub token_bytes: usize,
    pub pending_bytes: usize,
    pub delta_bytes: usize,
}

/// Owned irrevocable decoding, sharing only immutable vocabulary. Callers choose
/// stop/control ingestion and must finish at every normal/cancelled terminal.
pub struct Gemma4TextDecoder {
    tokenizer: Arc<Tokenizer>,
    pending: Vec<u8>,
    limits: DecodeLimits,
    tokens: usize,
    failed: bool,
}

impl Gemma4TextProcessor {
    /// Reserve the worst-case fallback run before accepting generation. No model
    /// or processor borrow is retained; the decoder can outlive both.
    /// # Errors
    /// Rejects zero bounds or a token budget exceeding the pending-byte envelope.
    pub fn decoder(&self, limits: DecodeLimits) -> Result<Gemma4TextDecoder, Gemma4Error> {
        if limits.tokens == 0
            || limits.token_bytes == 0
            || limits.delta_bytes == 0
            || limits.tokens > limits.pending_bytes
        {
            return Err(Gemma4Error::new(
                "decode limits",
                "positive bounds and pending bytes covering the token budget are required",
            ));
        }
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(limits.tokens)
            .map_err(|e| Gemma4Error::new("decode allocation", e.to_string()))?;
        Ok(Gemma4TextDecoder {
            tokenizer: self.tokenizer.clone(),
            pending,
            limits,
            tokens: 0,
            failed: false,
        })
    }
}

impl Gemma4TextDecoder {
    /// Newly irrevocable text, possibly empty while collecting fallback bytes.
    /// # Errors
    /// Rejects unknown IDs, token/delta limits and all calls after an error.
    pub fn push(&mut self, id: u32) -> Result<String, Gemma4Error> {
        if self.failed {
            return Err(Gemma4Error::new(
                "decode",
                "decoder is invalid after an earlier error",
            ));
        }
        let result = self.push_inner(id);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn push_inner(&mut self, id: u32) -> Result<String, Gemma4Error> {
        self.tokens = self
            .tokens
            .checked_add(1)
            .filter(|&n| n <= self.limits.tokens)
            .ok_or_else(|| Gemma4Error::new("decode", "token limit exceeded"))?;
        let token = self
            .tokenizer
            .id_to_token(id)
            .ok_or_else(|| Gemma4Error::new("decode", "unknown token ID"))?;
        if let Some(byte) = fallback(&token) {
            // One byte per token, with the entire token budget reserved at construction.
            self.pending.push(byte);
            return Ok(String::new());
        }
        // Replace precedes ByteFallback: fallback-encoded U+2581 stays literal.
        let piece = if token.contains('▁') {
            token.replace('▁', " ")
        } else {
            token
        };
        if piece.len() > self.limits.token_bytes {
            return Err(Gemma4Error::new(
                "decoded token bytes",
                "piece limit exceeded",
            ));
        }
        let length = self
            .pending_output_len()?
            .checked_add(piece.len())
            .ok_or_else(|| Gemma4Error::new("decoded bytes", "length overflow"))?;
        self.check_delta(length)?;
        let mut output = self.flush()?;
        if output.is_empty() {
            return Ok(piece);
        }
        output.push_str(&piece);
        Ok(output)
    }
    /// Consume and finalize a trailing run exactly once.
    /// # Errors
    /// Rejects failed decoders or final output exceeding the delta bound.
    pub fn finish(mut self) -> Result<String, Gemma4Error> {
        if self.failed {
            return Err(Gemma4Error::new(
                "decode",
                "cannot finalize a failed decoder",
            ));
        }
        self.flush()
    }
    fn check_delta(&self, bytes: usize) -> Result<(), Gemma4Error> {
        if bytes > self.limits.delta_bytes {
            return Err(Gemma4Error::new("decoded bytes", "delta limit exceeded"));
        }
        Ok(())
    }
    fn pending_output_len(&self) -> Result<usize, Gemma4Error> {
        if std::str::from_utf8(&self.pending).is_ok() {
            Ok(self.pending.len())
        } else {
            self.pending
                .len()
                .checked_mul(3)
                .ok_or_else(|| Gemma4Error::new("decoded bytes", "length overflow"))
        }
    }
    fn flush(&mut self) -> Result<String, Gemma4Error> {
        let length = self.pending_output_len()?;
        self.check_delta(length)?;
        let output = if std::str::from_utf8(&self.pending).is_ok() {
            // Keep the pre-reserved run storage across boundaries, not a new growing Vec.
            std::str::from_utf8(&self.pending)
                .expect("validated UTF-8")
                .to_owned()
        } else {
            "\u{fffd}".repeat(self.pending.len())
        };
        self.pending.clear();
        Ok(output)
    }
}

fn fallback(token: &str) -> Option<u8> {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        u8::from_str_radix(&token[3..5], 16).ok()
    } else {
        None
    }
}
