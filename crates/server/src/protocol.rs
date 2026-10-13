use ribn_text::{GenerationOptions, Message, TextInput, Usage};
use serde::Deserialize;
use serde::de::{self, SeqAccess, Visitor};
use serde_json::{Value, json};

use crate::ServerConfig;
use crate::response::ApiError;

pub(crate) const MAX_MESSAGES: usize = 128;

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
    temperature: Option<f32>,
    top_p: Option<f32>,
    n: Option<u32>,
    stream_options: Option<StreamOptions>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatMessage {
    role: String,
    content: String,
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
    ) -> Result<(TextInput, GenerationOptions, bool), ApiError> {
        if self.model != config.model_name {
            return Err(ApiError::bad("unknown model"));
        }
        if self.temperature.is_some_and(|n| n != 0.0)
            || self.top_p.is_some_and(|n| n.to_bits() != 1.0_f32.to_bits())
            || self.n.is_some_and(|n| n != 1)
        {
            return Err(ApiError::bad(
                "only greedy temperature=0, top_p=1 and n=1 are supported",
            ));
        }
        if self.stream_options.is_some() && !self.stream {
            return Err(ApiError::bad("stream_options requires stream=true"));
        }
        let max_tokens = self.max_tokens.unwrap_or(32);
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
                TextInput::Chat(
                    messages
                        .into_iter()
                        .map(|m| Message::new(m.role, m.content))
                        .collect(),
                )
            }
            _ => {
                return Err(ApiError::bad(
                    "supply prompt for completions or nonempty messages for chat",
                ));
            }
        };
        let options = GenerationOptions {
            max_output_tokens: max_tokens,
            ..GenerationOptions::default()
        };
        Ok((
            input,
            options,
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
