//! Text-only processor loaded from one unchanged local package. This does not
//! execute a model, process media or provide a frontend submission/retention bound.

use crate::DecodeLimits;
use crate::{Gemma4Config, Gemma4Error, config::token_id};
use minijinja::{Environment, context};
use ribn_hf::LocalModelPackage;
use serde::Serialize;
use serde_json::Value;
use std::io::{Read, Write};
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Per-call bounds, also enforced for direct callers. Rendered bytes bounds both
/// rendered chat and whole-sequence decoded text. Frontends must additionally
/// bound aggregate retained work before queueing it.
#[derive(Clone, Copy, Debug)]
pub struct TextLimits {
    pub input_bytes: usize,
    pub rendered_bytes: usize,
    pub tokens: usize,
    pub messages: usize,
}

/// Borrowed text chat, without tool calls, reasoning fields or media payloads.
/// Supported roles are system, developer, user and assistant. The actual template
/// owns role formatting, consecutive-turn handling, trimming and thought removal.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct TextMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

pub struct Gemma4TextProcessor {
    config: Gemma4Config,
    pub(crate) tokenizer: Arc<Tokenizer>,
    template: Environment<'static>,
    stops: Vec<u32>,
    pub(crate) limits: TextLimits,
}

impl Gemma4TextProcessor {
    /// Load tokenizer, canonical template and generation stop IDs from this
    /// package. Uses the package's retained config value, not a second config open.
    /// Keep package members unchanged throughout loading. Loaded processor assets
    /// are owned in memory; this is not an atomic package snapshot or weight load.
    /// # Errors
    /// Rejects invalid/unsupported assets or nonpositive/infeasible limits.
    pub fn open(package: &LocalModelPackage, limits: TextLimits) -> Result<Self, Gemma4Error> {
        let config = Gemma4Config::from_json(package.config())?;
        match std::fs::symlink_metadata(package.root().join("chat_templates")) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(error("chat_templates", e)),
            Ok(_) => {
                return Err(Gemma4Error::new(
                    "chat_templates",
                    "alternate template directories are unsupported",
                ));
            }
        }
        if [
            limits.input_bytes,
            limits.rendered_bytes,
            limits.tokens,
            limits.messages,
        ]
        .contains(&0)
            || limits.tokens > config.max_position_embeddings()
        {
            return Err(Gemma4Error::new(
                "processor limits",
                "positive bounds within model context are required",
            ));
        }
        let tokenizer = load_tokenizer(package, &config)?;
        let metadata: Value =
            serde_json::from_slice(&asset(package, "tokenizer_config.json", 1024 * 1024)?)
                .map_err(|e| error("tokenizer_config.json", e))?;
        if metadata["tokenizer_class"] != "GemmaTokenizer" || metadata["backend"] != "tokenizers" {
            return Err(Gemma4Error::new(
                "tokenizer_config.json",
                "only the canonical Gemma tokenizer backend is supported",
            ));
        }
        let generation: Value =
            serde_json::from_slice(&asset(package, "generation_config.json", 1024 * 1024)?)
                .map_err(|e| error("generation_config.json", e))?;
        if token_id(&generation, "bos_token_id")? != config.bos
            || token_id(&generation, "pad_token_id")? != config.pad
        {
            return Err(Gemma4Error::new(
                "generation_config.json",
                "BOS/PAD disagree with model config",
            ));
        }
        let bos = metadata["bos_token"]
            .as_str()
            .ok_or_else(|| Gemma4Error::new("bos_token", "missing string"))?;
        if tokenizer.token_to_id(bos) != Some(config.bos) {
            return Err(Gemma4Error::new(
                "bos_token",
                "tokenizer and model disagree",
            ));
        }
        let stops: Vec<u32> = generation["eos_token_id"]
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                Gemma4Error::new("eos_token_id", "expected nonempty generation stop list")
            })?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .filter(|&id| {
                        tokenizer.id_to_token(id).is_some()
                            && usize::try_from(id).is_ok_and(|n| n < config.vocab_size())
                    })
                    .ok_or_else(|| Gemma4Error::new("eos_token_id", "invalid vocabulary ID"))
            })
            .collect::<Result<_, _>>()?;
        if stops
            .iter()
            .enumerate()
            .any(|(i, id)| stops[..i].contains(id))
        {
            return Err(Gemma4Error::new("eos_token_id", "duplicate stop ID"));
        }
        let template = load_template(package, bos)?;
        Ok(Self {
            config,
            tokenizer: Arc::new(tokenizer),
            template,
            stops,
            limits,
        })
    }

    #[must_use]
    pub fn stop_token_ids(&self) -> &[u32] {
        &self.stops
    }

    /// Raw prompts do not acquire BOS or EOS. Chat templates insert their own BOS.
    /// # Errors
    /// Rejects excessive bytes/tokens, empty input or unsupported token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Gemma4Error> {
        self.encode_bounded(text, self.limits.input_bytes)
    }
    fn encode_bounded(&self, text: &str, bytes: usize) -> Result<Vec<u32>, Gemma4Error> {
        check(text.len(), bytes, "input bytes")?;
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| error("tokenization", e))?;
        self.validate_input(encoded.get_ids())?;
        Ok(encoded.get_ids().to_vec())
    }
    /// Validate raw token-ID input with the same bounds/semantics as encoded text.
    /// # Errors
    /// Rejects zero/excessive tokens, OOV IDs and unsupported media placeholders.
    pub fn validate_input(&self, ids: &[u32]) -> Result<(), Gemma4Error> {
        if ids.is_empty() {
            return Err(Gemma4Error::new("input", "at least one token is required"));
        }
        check(ids.len(), self.limits.tokens, "input tokens")?;
        self.config.validate_text_tokens(ids)
    }
    /// Render the publisher's template with generation prompt on, thinking off.
    /// # Errors
    /// Rejects unknown roles, oversized input/output and template errors/fuel exhaustion.
    pub fn render_chat(&self, messages: &[TextMessage<'_>]) -> Result<String, Gemma4Error> {
        if messages.is_empty() {
            return Err(Gemma4Error::new("chat", "at least one message is required"));
        }
        check(messages.len(), self.limits.messages, "message count")?;
        let mut input = 0_usize;
        for message in messages {
            if !matches!(message.role, "system" | "developer" | "user" | "assistant") {
                return Err(Gemma4Error::new(
                    "chat role",
                    "only system/developer/user/assistant text messages are supported",
                ));
            }
            input = input
                .checked_add(message.role.len())
                .and_then(|n| n.checked_add(message.content.len()))
                .ok_or_else(|| Gemma4Error::new("input bytes", "overflow"))?;
            check(input, self.limits.input_bytes, "input bytes")?;
        }
        let mut out = LimitedOutput {
            bytes: Vec::new(),
            limit: self.limits.rendered_bytes,
        };
        self.template.get_template("chat").map_err(|e| error("chat template", e))?
            .render_captured_to(context! { messages => messages, add_generation_prompt => true, enable_thinking => false }, &mut out)
            .map_err(|e| error("chat render", e))?;
        String::from_utf8(out.bytes).map_err(|e| error("chat render", e))
    }
    /// # Errors
    /// Returns the same chat rendering and text input errors as the separate operations.
    pub fn encode_chat(&self, messages: &[TextMessage<'_>]) -> Result<Vec<u32>, Gemma4Error> {
        self.encode_bounded(&self.render_chat(messages)?, self.limits.rendered_bytes)
    }
    /// Whole-sequence decode, preserving special markers and without Python
    /// post-decoding whitespace cleanup. This is not an incremental text stream.
    /// # Errors
    /// Rejects unknown IDs, excessive token count/decoded bytes or codec errors.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Gemma4Error> {
        check(ids.len(), self.limits.tokens, "decoded tokens")?;
        if ids.is_empty() {
            return Ok(String::new());
        }
        let mut decoder = self.decoder(DecodeLimits {
            tokens: ids.len(),
            token_bytes: self.limits.rendered_bytes,
            pending_bytes: ids.len(),
            delta_bytes: self.limits.rendered_bytes,
        })?;
        let mut output = String::new();
        for &id in ids {
            let piece = decoder.push(id)?;
            check(
                output.len().saturating_add(piece.len()),
                self.limits.rendered_bytes,
                "decoded bytes",
            )?;
            output.push_str(&piece);
        }
        let trailing = decoder.finish()?;
        check(
            output.len().saturating_add(trailing.len()),
            self.limits.rendered_bytes,
            "decoded bytes",
        )?;
        output.push_str(&trailing);
        Ok(output)
    }
}

fn load_template(
    package: &LocalModelPackage,
    bos: &str,
) -> Result<Environment<'static>, Gemma4Error> {
    let source = String::from_utf8(asset(package, "chat_template.jinja", 1024 * 1024)?)
        .map_err(|e| error("chat_template.jinja", e))?;
    let mut template = Environment::new();
    template.set_fuel(Some(1_000_000));
    template.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    template.add_filter("trim", python_trim);
    template.add_function(
        "raise_exception",
        |message: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    template.add_global("bos_token", bos.to_owned());
    template
        .add_template_owned("chat", source)
        .map_err(|e| error("chat_template.jinja", e))?;
    Ok(template)
}

// Python str.strip also treats U+001C..U+001F as whitespace; Rust's
// Unicode White_Space predicate (and MiniJinja's default trim) does not.
fn python_trim(value: &str, characters: Option<&str>) -> String {
    value
        .trim_matches(|c: char| {
            characters.map_or_else(
                || c.is_whitespace() || matches!(c, '\u{001c}'..='\u{001f}'),
                |set| set.contains(c),
            )
        })
        .to_owned()
}

fn load_tokenizer(
    package: &LocalModelPackage,
    config: &Gemma4Config,
) -> Result<Tokenizer, Gemma4Error> {
    let bytes = asset(package, "tokenizer.json", 64 * 1024 * 1024)?;
    {
        let definition: Value =
            serde_json::from_slice(&bytes).map_err(|e| error("tokenizer.json", e))?;
        if definition["model"]["type"] != "BPE"
            || !definition["model"]["dropout"].is_null()
            || definition["decoder"]
                != serde_json::json!({"type":"Sequence","decoders":[
                    {"type":"Replace","pattern":{"String":"▁"},"content":" "},
                    {"type":"ByteFallback"},{"type":"Fuse"}]})
        {
            return Err(Gemma4Error::new(
                "tokenizer.json",
                "expected deterministic Gemma BPE and byte-fallback decoder",
            ));
        }
    }
    let tokenizer = Tokenizer::from_bytes(bytes).map_err(|e| error("tokenizer.json", e))?;
    if tokenizer.get_vocab_size(true) != config.vocab_size()
        || tokenizer.get_padding().is_some()
        || tokenizer.get_truncation().is_some()
    {
        return Err(Gemma4Error::new(
            "tokenizer.json",
            "vocabulary mismatch or implicit padding/truncation",
        ));
    }
    Ok(tokenizer)
}

fn asset(package: &LocalModelPackage, name: &str, limit: usize) -> Result<Vec<u8>, Gemma4Error> {
    let path = package.package_file(name).map_err(|e| error(name, e))?;
    let file = std::fs::File::open(path).map_err(|e| error(name, e))?;
    let max = u64::try_from(limit).map_err(|e| error(name, e))?;
    if file.metadata().map_err(|e| error(name, e))?.len() > max {
        return Err(Gemma4Error::new(name, "asset byte limit exceeded"));
    }
    let mut bytes = Vec::new();
    file.take(max + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| error(name, e))?;
    check(bytes.len(), limit, name)?;
    Ok(bytes)
}
fn check(actual: usize, max: usize, name: &str) -> Result<(), Gemma4Error> {
    if actual > max {
        return Err(Gemma4Error::new(name, format!("{actual} exceeds {max}")));
    }
    Ok(())
}
fn error(name: &str, e: impl std::fmt::Display) -> Gemma4Error {
    Gemma4Error::new(name, e.to_string())
}
struct LimitedOutput {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for LimitedOutput {
    fn write(&mut self, input: &[u8]) -> std::io::Result<usize> {
        if input.len() > self.limit - self.bytes.len() {
            return Err(std::io::Error::other("rendered byte limit exceeded"));
        }
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
