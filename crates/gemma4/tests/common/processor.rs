//! Shared small package assets for native processor and facade integration tests.
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
pub(crate) struct Fixture(pub(crate) PathBuf);
impl Fixture {
    pub(crate) fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ribn-gemma4-processor-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        let fixture = Self(path);
        let mut config: Value = serde_json::from_str(include_str!(
            "../../../../benchmarks/gemma4-reference/config.json"
        ))
        .unwrap();
        config["text_config"]["vocab_size"] = json!(268);
        config["text_config"]["vocab_size_per_layer_input"] = json!(268);
        config["image_token_id"] = json!(7);
        config["audio_token_id"] = json!(8);
        config["video_token_id"] = json!(9);
        config["eos_token_id"] = json!([1, 4]);
        fixture.json("config.json", &config);
        // Package resolution only: no weights may be opened by text processing.
        std::fs::write(fixture.0.join("model.safetensors"), []).unwrap();
        let symbols = [
            "<pad>",
            "<eos>",
            "<bos>",
            "<unk>",
            "<turn|>",
            "<|tool_response>",
            "▁",
            "<|image|>",
            "<|audio|>",
            "<|video|>",
            "X",
            "é",
        ];
        let mut vocab = serde_json::Map::new();
        let mut added = Vec::new();
        for (i, symbol) in symbols.into_iter().enumerate() {
            vocab.insert(symbol.to_owned(), json!(i));
            if ![6, 10, 11].contains(&i) {
                added.push(json!({"id":i,"content":symbol,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}));
            }
        }
        for byte in 0..256 {
            vocab.insert(format!("<0x{byte:02X}>"), json!(12 + byte));
        }
        fixture.json("tokenizer.json", &json!({"version":"1.0","truncation":null,"padding":null,
            "added_tokens":added,"normalizer":{"type":"Replace","pattern":{"String":" "},"content":"▁"},"pre_tokenizer":null,"post_processor":null,
            "decoder":{"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]},
            "model":{"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":true,"ignore_merges":false,"vocab":vocab,"merges":[]}}));
        fixture.json(
            "tokenizer_config.json",
            &json!({"bos_token":"<bos>","tokenizer_class":"GemmaTokenizer","backend":"tokenizers"}),
        );
        fixture.json(
            "generation_config.json",
            &json!({"bos_token_id":2,"pad_token_id":0,"eos_token_id":[1,4,5]}),
        );
        std::fs::write(
            fixture.0.join("chat_template.jinja"),
            include_str!("../../../../benchmarks/gemma4-reference/chat_template.jinja"),
        )
        .unwrap();
        fixture
    }
    pub(crate) fn json(&self, name: &str, value: &Value) {
        std::fs::write(self.0.join(name), serde_json::to_vec(value).unwrap()).unwrap();
    }
    pub(crate) fn change(&self, name: &str, edit: impl FnOnce(&mut Value)) {
        let mut value = serde_json::from_slice(&std::fs::read(self.0.join(name)).unwrap()).unwrap();
        edit(&mut value);
        self.json(name, &value);
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
