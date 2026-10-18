//! Gemma's exact Replace → `ByteFallback` → Fuse pipeline. A fallback run cannot
//! be emitted early: one invalid trailing byte invalidates the entire run in HF.

use crate::{Gemma4Error, Gemma4TextProcessor};

/// Request-local irrevocable text decoding, preserving special markers. Callers
/// choose which stop/control tokens to ingest, and must finish at every terminal
/// (including a stop token they do not ingest) to flush a trailing fallback run.
/// This does not adapt the existing GGUF stateless frontend codec.
pub struct Gemma4TextDecoder<'a> {
    processor: &'a Gemma4TextProcessor,
    pending: Vec<u8>,
    tokens: usize,
    emitted: usize,
    failed: bool,
}

impl Gemma4TextProcessor {
    #[must_use]
    pub const fn decoder(&self) -> Gemma4TextDecoder<'_> {
        Gemma4TextDecoder {
            processor: self,
            pending: Vec::new(),
            tokens: 0,
            emitted: 0,
            failed: false,
        }
    }
}

impl Gemma4TextDecoder<'_> {
    /// Returns newly irrevocable text, possibly empty while collecting bytes.
    /// # Errors
    /// Rejects unknown IDs, token/output limits and all calls after an error.
    /// An error permanently invalidates this decoder; do not retry the token.
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
            .filter(|&n| n <= self.processor.limits.tokens)
            .ok_or_else(|| Gemma4Error::new("decode", "token limit exceeded"))?;
        let token = self
            .processor
            .tokenizer
            .id_to_token(id)
            .ok_or_else(|| Gemma4Error::new("decode", "unknown token ID"))?;
        if let Some(byte) = fallback(&token) {
            self.pending.push(byte);
            return Ok(String::new());
        }
        // Replace runs before ByteFallback, so decoded fallback bytes containing
        // U+2581 must not be subjected to replacement a second time.
        let piece = if token.contains('▁') {
            token.replace('▁', " ")
        } else {
            token
        };
        let mut output = self.flush()?;
        self.charge(piece.len())?;
        if output.is_empty() {
            return Ok(piece);
        }
        output.push_str(&piece);
        Ok(output)
    }
    /// Consume the decoder and finalize a trailing run exactly once.
    /// # Errors
    /// Rejects failed decoders or final output that exceeds its byte limit.
    pub fn finish(mut self) -> Result<String, Gemma4Error> {
        if self.failed {
            return Err(Gemma4Error::new(
                "decode",
                "cannot finalize a failed decoder",
            ));
        }
        self.flush()
    }
    fn charge(&mut self, bytes: usize) -> Result<(), Gemma4Error> {
        self.emitted = self
            .emitted
            .checked_add(bytes)
            .filter(|&n| n <= self.processor.limits.rendered_bytes)
            .ok_or_else(|| Gemma4Error::new("decoded bytes", "limit exceeded"))?;
        Ok(())
    }
    fn flush(&mut self) -> Result<String, Gemma4Error> {
        let valid = std::str::from_utf8(&self.pending).is_ok();
        let length = if valid {
            self.pending.len()
        } else {
            self.pending
                .len()
                .checked_mul(3)
                .ok_or_else(|| Gemma4Error::new("decoded bytes", "length overflow"))?
        };
        self.charge(length)?;
        let bytes = std::mem::take(&mut self.pending);
        if valid {
            String::from_utf8(bytes).map_err(|e| Gemma4Error::new("decode", e.to_string()))
        } else {
            Ok("\u{fffd}".repeat(bytes.len()))
        }
    }
}

fn fallback(token: &str) -> Option<u8> {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        u8::from_str_radix(&token[3..5], 16).ok()
    } else {
        None
    }
}
