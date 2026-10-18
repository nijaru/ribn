//! Gemma package text semantics above the token runtime, not a model executor.

use crate::{
    EncodedInput, Message, ProcessorLimits, TextDecoder, TextError, TextInput, TextProcessor,
};
use engine_gemma4::{
    DecodeLimits, Gemma4TextDecoder, Gemma4TextProcessor, TextLimits, TextMessage,
};
use ribn_hf::LocalModelPackage;
use std::sync::Arc;

/// Host text preparation and owned decoding for Gemma 4. Pairing it with an
/// executor does not establish that executor's model or device correctness.
pub struct GemmaProcessor {
    processor: Gemma4TextProcessor,
    limits: ProcessorLimits,
}

impl GemmaProcessor {
    /// Load canonical assets without opening weights. Keep package members
    /// unchanged during loading; loaded vocabulary/template are then owned.
    /// # Errors
    /// Rejects invalid bounds or unsupported model/processor assets.
    pub fn open(package: &LocalModelPackage, limits: ProcessorLimits) -> Result<Self, TextError> {
        if !limits.is_valid() || limits.max_retained_bytes < size_of::<Message>() {
            return Err(TextError::InvalidInput(
                "positive text bounds covering one message are required",
            ));
        }
        let processor = Gemma4TextProcessor::open(
            package,
            TextLimits {
                input_bytes: limits.max_input_bytes,
                rendered_bytes: limits.max_rendered_bytes,
                tokens: limits.max_prompt_tokens,
                messages: limits.max_retained_bytes / size_of::<Message>(),
            },
        )
        .map_err(TextError::processor)?;
        Ok(Self { processor, limits })
    }
}

impl TextProcessor for GemmaProcessor {
    fn limits(&self) -> ProcessorLimits {
        self.limits
    }
    fn encode(&self, input: TextInput) -> Result<EncodedInput, TextError> {
        self.limits.check_input(&input)?;
        let tokens = match input {
            TextInput::Tokens(ids) => {
                self.processor
                    .validate_input(&ids)
                    .map_err(TextError::processor)?;
                ids
            }
            TextInput::Prompt(text) => {
                self.processor.encode(&text).map_err(TextError::processor)?
            }
            TextInput::Chat(messages) => {
                let borrowed = messages
                    .iter()
                    .map(|message| TextMessage {
                        role: &message.role,
                        content: &message.content,
                    })
                    .collect::<Vec<_>>();
                self.processor
                    .encode_chat(&borrowed)
                    .map_err(TextError::processor)?
            }
        };
        Ok(EncodedInput::new(
            tokens,
            self.processor.stop_token_ids().to_vec(),
        ))
    }
    fn decoder(self: Arc<Self>, output_tokens: usize) -> Result<Box<dyn TextDecoder>, TextError> {
        if output_tokens > self.limits.max_decode_pending_bytes {
            return Err(TextError::limit(
                "decoder pending bytes",
                self.limits.max_decode_pending_bytes,
                output_tokens,
            ));
        }
        let decoder = self
            .processor
            .decoder(DecodeLimits {
                tokens: output_tokens,
                token_bytes: self.limits.max_decoded_token_bytes,
                pending_bytes: self.limits.max_decode_pending_bytes,
                delta_bytes: self.limits.max_decoded_delta_bytes,
            })
            .map_err(TextError::processor)?;
        Ok(Box::new(Decoder(decoder)))
    }
}
struct Decoder(Gemma4TextDecoder);
impl TextDecoder for Decoder {
    fn push(&mut self, token: u32) -> Result<String, TextError> {
        self.0.push(token).map_err(|source| TextError::Decode {
            token,
            source: Arc::new(source),
        })
    }
    fn finish(self: Box<Self>) -> Result<String, TextError> {
        self.0.finish().map_err(TextError::processor)
    }
}
