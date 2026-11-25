use ribn_text::{GenerationOptions, Message, TextInput, TextRequest, Usage};
use serde::Deserialize;
use serde::de::{self, SeqAccess, Visitor};
use serde_json::{Value, json};

use crate::ServerConfig;
use crate::response::ApiError;

pub(crate) const MAX_MESSAGES: usize = 128;
const MAX_CONTENT_PARTS: usize = 64;

#[derive(Clone, Copy)]
pub(crate) enum Operation {
    Completion,
    Chat,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    model: String,
    prompt: Option<String>,
    #[serde(default, deserialize_with = "messages")]
    messages: Option<Vec<ChatMessage>>,
    #[serde(default)]
    pub stream: bool,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    n: Option<u32>,
    repetition_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    logprobs: Option<u32>,
    stop: Option<StopSequence>,
    ignore_eos: Option<bool>,
    seed: Option<u64>,
    stream_options: Option<StreamOptions>,
}

/// `OpenAI` accepts one stop string or a bounded array of them.
#[derive(Deserialize)]
#[serde(untagged)]
enum StopSequence {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TextPart {
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

/// `OpenAI` chat message content: plain text string or bounded text-part array.
#[derive(Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(#[serde(deserialize_with = "content_parts")] Vec<TextPart>),
}

impl MessageContent {
    fn into_text(self) -> Result<String, ApiError> {
        match self {
            Self::Text(text) => Ok(text),
            Self::Parts(parts) => {
                if parts.is_empty() {
                    return Err(ApiError::bad("message content parts must not be empty"));
                }
                let mut total_len = 0_usize;
                for part in &parts {
                    if part.kind != "text" {
                        return Err(ApiError::bad("only text content parts are supported"));
                    }
                    total_len = total_len.saturating_add(part.text.len());
                }
                let mut out = String::with_capacity(total_len);
                for part in parts {
                    out.push_str(&part.text);
                }
                Ok(out)
            }
        }
    }
}

fn content_parts<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<TextPart>, D::Error> {
    struct Parts;
    impl<'de> Visitor<'de> for Parts {
        type Value = Vec<TextPart>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("an array of at most 64 text content parts")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut result = Vec::new();
            while let Some(part) = seq.next_element()? {
                if result.len() == MAX_CONTENT_PARTS {
                    return Err(de::Error::custom("too many content parts"));
                }
                result.push(part);
            }
            Ok(result)
        }
    }
    d.deserialize_seq(Parts)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatMessage {
    role: String,
    content: MessageContent,
}

/// Stop large arrays while parsing, rather than after retaining every message.
fn messages<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<Vec<ChatMessage>>, D::Error> {
    struct Messages;
    impl<'de> Visitor<'de> for Messages {
        type Value = Option<Vec<ChatMessage>>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("an array of at most 128 text messages")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut result = Vec::new();
            while let Some(message) = seq.next_element()? {
                if result.len() == MAX_MESSAGES {
                    return Err(de::Error::custom("too many chat messages"));
                }
                result.push(message);
            }
            Ok(Some(result))
        }
    }
    d.deserialize_seq(Messages)
}

impl Request {
    pub(crate) fn prepare(
        self,
        operation: Operation,
        config: &ServerConfig,
        supports_sampling: bool,
    ) -> Result<(TextRequest, bool), ApiError> {
        if self.model != config.model_name {
            return Err(ApiError::bad("unknown model"));
        }
        if self.n.is_some_and(|n| n != 1)
            || self
                .repetition_penalty
                .is_some_and(|n| n.to_bits() != 1.0_f32.to_bits())
            || self
                .presence_penalty
                .is_some_and(|n| !n.is_finite() || n != 0.0)
            || self
                .frequency_penalty
                .is_some_and(|n| !n.is_finite() || n != 0.0)
            || self.logprobs.is_some()
        {
            return Err(ApiError::bad(
                "only n=1, repetition_penalty=1, zero presence/frequency penalties and logprobs=null are supported",
            ));
        }
        let temperature = self.temperature.unwrap_or(0.0);
        let top_p = self.top_p.unwrap_or(1.0);
        let top_k = self.top_k.unwrap_or(0);
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(ApiError::bad("temperature must be finite and nonnegative"));
        }
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(ApiError::bad("top_p must be in (0, 1]"));
        }
        if !supports_sampling
            && (temperature != 0.0 || top_p.to_bits() != 1.0_f32.to_bits() || top_k != 0)
        {
            return Err(ApiError::bad(
                "only greedy temperature=0, top_p=1, n=1, repetition_penalty=1 and logprobs=null are supported",
            ));
        }
        if self.stream_options.is_some() && !self.stream {
            return Err(ApiError::bad("stream_options requires stream=true"));
        }
        let max_tokens = match (operation, self.max_tokens, self.max_completion_tokens) {
            (Operation::Completion, _, Some(_)) => {
                return Err(ApiError::bad(
                    "max_completion_tokens is only supported for chat completions",
                ));
            }
            (Operation::Chat, Some(_), Some(_)) => {
                return Err(ApiError::bad(
                    "specify either max_tokens or max_completion_tokens, not both",
                ));
            }
            (_, max_t, max_c) => max_c.or(max_t).unwrap_or(32),
        };
        if max_tokens == 0 || max_tokens > config.max_output_tokens {
            return Err(ApiError::bad("max_tokens exceeds the supported range"));
        }
        let input = match (operation, self.prompt, self.messages) {
            (Operation::Completion, Some(prompt), None) => TextInput::Prompt(prompt),
            (Operation::Chat, None, Some(messages)) if !messages.is_empty() => {
                if messages
                    .iter()
                    .any(|m| !matches!(m.role.as_str(), "system" | "user" | "assistant"))
                {
                    return Err(ApiError::bad(
                        "only system, user and assistant roles are supported",
                    ));
                }
                let mut parsed = Vec::with_capacity(messages.len());
                for m in messages {
                    parsed.push(Message::new(m.role, m.content.into_text()?));
                }
                TextInput::Chat(parsed)
            }
            _ => {
                return Err(ApiError::bad(
                    "supply prompt for completions or nonempty messages for chat",
                ));
            }
        };
        let mut options = GenerationOptions {
            max_output_tokens: max_tokens,
            ..GenerationOptions::default()
        };
        options.sampling.temperature = temperature;
        options.sampling.top_p = top_p;
        options.sampling.top_k = top_k;
        options.sampling.seed = self.seed;
        let stops = match self.stop {
            None => Vec::new(),
            Some(StopSequence::One(stop)) => vec![stop],
            Some(StopSequence::Many(stops)) => stops,
        };
        // Bounds are enforced by the text layer before enqueue, so an invalid
        // stop list settles as a request-local 400 rather than a partial run.
        let request = TextRequest::new(input, options)
            .with_stops(stops)
            .with_ignore_eos(self.ignore_eos.unwrap_or(false));
        Ok((
            request,
            self.stream_options.is_some_and(|o| o.include_usage),
        ))
    }
}

pub(crate) struct Output {
    pub operation: Operation,
    pub id: String,
    pub model: String,
    pub created: u64,
}

impl Output {
    pub(crate) fn chunk(&self, text: &str, finish: Option<&str>) -> Value {
        let (object, choice) = match self.operation {
            Operation::Completion => (
                "text_completion",
                json!({
                    "index": 0, "text": text, "logprobs": null, "finish_reason": finish
                }),
            ),
            Operation::Chat => (
                "chat.completion.chunk",
                json!({
                    "index": 0, "delta": if finish.is_none() { json!({"content": text}) } else { json!({}) },
                    "finish_reason": finish
                }),
            ),
        };
        self.envelope(object, json!([choice]))
    }

    pub(crate) fn role(&self) -> Value {
        self.envelope(
            "chat.completion.chunk",
            json!([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]),
        )
    }

    pub(crate) fn usage(&self, usage: Usage) -> Value {
        let mut value = self.envelope(
            match self.operation {
                Operation::Chat => "chat.completion.chunk",
                Operation::Completion => "text_completion",
            },
            json!([]),
        );
        value["usage"] = usage_json(usage);
        value
    }

    pub(crate) fn complete(&self, text: &str, finish: &str, usage: Usage) -> Value {
        let (object, choice) = match self.operation {
            Operation::Completion => (
                "text_completion",
                json!({
                    "index": 0, "text": text, "logprobs": null, "finish_reason": finish
                }),
            ),
            Operation::Chat => (
                "chat.completion",
                json!({
                    "index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish
                }),
            ),
        };
        let mut value = self.envelope(object, json!([choice]));
        value["usage"] = usage_json(usage);
        value
    }

    fn envelope(&self, object: &str, choices: Value) -> Value {
        let mut value = json!({"id": self.id, "object": object, "created": self.created,
            "model": self.model});
        value["choices"] = choices;
        value
    }
}

fn usage_json(usage: Usage) -> Value {
    json!({"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens()})
}
