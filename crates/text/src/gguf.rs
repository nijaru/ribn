//! GGUF text policy and strict incremental UTF-8 decoding.
use crate::{EncodedInput, ProcessorLimits, TextDecoder, TextError, TextInput, TextProcessor};
use engine_gguf::{ChatMessage, ChatTemplateOptions, GgufTokenizer};
use std::{io, sync::Arc};

/// GGUF tokenizer and chat template with enforced text bounds.
pub struct GgufProcessor {
    tokenizer: GgufTokenizer,
    limits: ProcessorLimits,
}

impl GgufProcessor {
    #[must_use]
    pub const fn new(tokenizer: GgufTokenizer, limits: ProcessorLimits) -> Self {
        Self { tokenizer, limits }
    }

    fn render_chat(&self, messages: &[ChatMessage]) -> Result<String, TextError> {
        let mut rendered = LimitedText::new(self.limits.max_rendered_bytes);
        // Preserve the existing Qwen CLI behavior: generation prompt on,
        // thinking off until reasoning controls have an explicit public API.
        let options = ChatTemplateOptions::new(true, false);
        match self
            .tokenizer
            .render_chat_to(messages, options, &mut rendered)
        {
            Ok(()) => Ok(rendered.text),
            Err(_) if rendered.exceeded.is_some() => Err(rendered.exceeded_error()),
            Err(error) => Err(TextError::processor(error)),
        }
    }
}

impl TextProcessor for GgufProcessor {
    fn limits(&self) -> ProcessorLimits {
        self.limits
    }

    fn encode(&self, input: TextInput) -> Result<EncodedInput, TextError> {
        // The retention boundary checks the same rule before queuing; this keeps
        // the processor authoritative for direct callers.
        self.limits.check_input(&input)?;
        let chat = matches!(&input, TextInput::Chat(_));
        let tokens = match input {
            TextInput::Tokens(tokens) => tokens,
            TextInput::Prompt(text) => {
                self.tokenizer.encode(&text).map_err(TextError::processor)?
            }
            TextInput::Chat(messages) => {
                if messages.is_empty() {
                    return Err(TextError::InvalidInput(
                        "chat input must contain at least one message",
                    ));
                }
                let messages = messages
                    .into_iter()
                    .map(|message| ChatMessage::new(message.role, message.content))
                    .collect::<Vec<_>>();
                let rendered = self.render_chat(&messages)?;
                self.tokenizer
                    .encode_rendered(&rendered)
                    .map_err(TextError::processor)?
            }
        };
        if tokens.len() > self.limits.max_prompt_tokens {
            return Err(TextError::limit(
                "prompt tokens",
                self.limits.max_prompt_tokens,
                tokens.len(),
            ));
        }
        // A chat turn ends at the artifact's declared EOS IDs and at the
        // delimiters its own prompt used; raw prompts and token inputs carry no
        // template structure to delimit.
        let stop_tokens = if chat {
            self.tokenizer.chat_stop_token_ids(&tokens)
        } else {
            self.tokenizer.stop_token_ids()
        };
        Ok(EncodedInput::new(tokens, stop_tokens))
    }

    fn decoder(self: Arc<Self>, output_tokens: usize) -> Result<Box<dyn TextDecoder>, TextError> {
        if output_tokens == 0
            || self.limits.max_decode_pending_bytes < 3
            || self
                .limits
                .max_decoded_token_bytes
                .checked_add(3)
                .is_none_or(|n| n > self.limits.max_decoded_delta_bytes)
        {
            return Err(TextError::InvalidInput(
                "GGUF decoding needs a positive budget, three pending bytes and delta space for a piece plus three bytes",
            ));
        }
        Ok(Box::new(GgufDecoder {
            processor: self,
            bytes: Vec::new(),
            utf8: Utf8Decoder::default(),
        }))
    }
}

struct GgufDecoder {
    processor: Arc<GgufProcessor>,
    bytes: Vec<u8>,
    utf8: Utf8Decoder,
}
impl TextDecoder for GgufDecoder {
    fn push(&mut self, token: u32) -> Result<String, TextError> {
        self.processor
            .tokenizer
            .decode_token_into(
                token,
                &mut self.bytes,
                self.processor.limits.max_decoded_token_bytes,
            )
            .map_err(|source| TextError::Decode {
                token,
                source: Arc::new(source),
            })?;
        let text = self.utf8.push(token, &self.bytes)?;
        check_delta(text, self.processor.limits.max_decoded_delta_bytes)
    }
    fn finish(mut self: Box<Self>) -> Result<String, TextError> {
        check_delta(
            self.utf8.finish(),
            self.processor.limits.max_decoded_delta_bytes,
        )
    }
}

fn check_delta(text: String, limit: usize) -> Result<String, TextError> {
    if text.len() > limit {
        return Err(TextError::limit("decoded delta bytes", limit, text.len()));
    }
    Ok(text)
}

/// A writer that fails instead of retaining more than its byte bound.
struct LimitedText {
    text: String,
    limit: usize,
    /// Attempted size at the point the bound was crossed, for diagnostics.
    exceeded: Option<usize>,
}

impl LimitedText {
    const fn new(limit: usize) -> Self {
        Self {
            text: String::new(),
            limit,
            exceeded: None,
        }
    }

    fn exceeded_error(&self) -> TextError {
        TextError::limit(
            "rendered prompt bytes",
            self.limit,
            self.exceeded.unwrap_or(self.limit),
        )
    }
}

impl io::Write for LimitedText {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.text.len()) {
            self.exceeded = Some(self.text.len().saturating_add(bytes.len()));
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "rendered prompt exceeds its byte bound",
            ));
        }
        // Templates render text; a non-UTF-8 write is a template defect and must
        // not be silently lossy.
        let text = std::str::from_utf8(bytes).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "template emitted invalid UTF-8")
        })?;
        self.text.push_str(text);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct Utf8Decoder {
    pending: Vec<u8>,
}

impl Utf8Decoder {
    /// Decode one token's bytes, retaining an incomplete final code point. The
    /// retained prefix is shorter than one code point, so it stays bounded.
    pub(crate) fn push(&mut self, token: u32, bytes: &[u8]) -> Result<String, TextError> {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(text) => {
                let text = text.to_owned();
                self.pending.clear();
                Ok(text)
            }
            Err(error) if error.error_len().is_none() => {
                let valid = error.valid_up_to();
                let prefix = std::str::from_utf8(&self.pending[..valid])
                    .expect("prefix validated by from_utf8")
                    .to_owned();
                self.pending.drain(..valid);
                Ok(prefix)
            }
            Err(source) => Err(TextError::Encoding { token, source }),
        }
    }

    /// Flush an incomplete final code point exactly once. Terminal bookkeeping,
    /// not recovery: the replacement character is the only honest output.
    pub(crate) fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            String::new()
        } else {
            let trailing = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            trailing
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Utf8Decoder;

    #[test]
    fn incremental_decoder_retains_split_codepoints() {
        let mut decoder = Utf8Decoder::default();
        assert_eq!(decoder.push(1, &[0xe2]).unwrap(), "");
        assert_eq!(decoder.push(2, &[0x82]).unwrap(), "");
        assert_eq!(decoder.push(3, &[0xac]).unwrap(), "€");
    }

    #[test]
    fn terminal_incomplete_codepoint_is_replaced_once() {
        let mut decoder = Utf8Decoder::default();
        assert_eq!(decoder.push(1, &[0xe2]).unwrap(), "");
        assert_eq!(decoder.finish(), "\u{fffd}");
        assert!(decoder.finish().is_empty());
    }

    #[test]
    fn invalid_bytes_report_the_offending_token() {
        let mut decoder = Utf8Decoder::default();
        let error = decoder.push(7, &[0xff]).expect_err("invalid UTF-8");
        assert!(matches!(error, super::TextError::Encoding { token: 7, .. }));
    }
}
