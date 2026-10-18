use engine_gemma4::{Gemma4Config, Gemma4TextProcessor, TextLimits, TextMessage};
use ribn_hf::LocalModelPackage;
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ribn-gemma4-processor-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        let fixture = Self(path);
        let mut config: Value = serde_json::from_str(include_str!(
            "../../../benchmarks/gemma4-reference/config.json"
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
            include_str!("../../../benchmarks/gemma4-reference/chat_template.jinja"),
        )
        .unwrap();
        fixture
    }
    fn json(&self, name: &str, value: &Value) {
        std::fs::write(self.0.join(name), serde_json::to_vec(value).unwrap()).unwrap();
    }
    fn change(&self, name: &str, edit: impl FnOnce(&mut Value)) {
        let mut value = serde_json::from_slice(&std::fs::read(self.0.join(name)).unwrap()).unwrap();
        edit(&mut value);
        self.json(name, &value);
    }
    fn processor(
        &self,
        limits: TextLimits,
    ) -> Result<Gemma4TextProcessor, engine_gemma4::Gemma4Error> {
        Gemma4TextProcessor::open(&LocalModelPackage::open(&self.0).unwrap(), limits)
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn limits() -> TextLimits {
    TextLimits {
        input_bytes: 65_536,
        rendered_bytes: 131_072,
        tokens: 8192,
        messages: 128,
    }
}

#[test]
fn real_template_preserves_reference_chat_subset_without_weights_or_reopen() {
    fn shared<T: Send + Sync>(_: &T) {}
    let fixture = Fixture::new();
    let processor = fixture.processor(limits()).unwrap();
    shared(&processor);
    let corpus: Value = serde_json::from_str(include_str!(
        "../../../benchmarks/gemma4-reference/processor-cases.json"
    ))
    .unwrap();
    for row in corpus["chat"].as_array().unwrap() {
        let messages: Vec<_> = row["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| TextMessage {
                role: m["role"].as_str().unwrap(),
                content: m["content"].as_str().unwrap(),
            })
            .collect();
        assert_eq!(processor.render_chat(&messages).unwrap(), row["rendered"]);
        assert_eq!(processor.encode_chat(&messages).unwrap()[0], 2);
    }
    assert_eq!(processor.stop_token_ids(), [1, 4, 5]);
    assert_eq!(
        processor.encode("X").unwrap(),
        [10],
        "raw prompts do not acquire BOS"
    );
    assert!(processor.encode("").is_err());
    assert!(processor.validate_input(&[]).is_err());
    // Every asset has already been retained. Later calls never reopen a path.
    drop(fixture);
    assert_eq!(processor.encode("X").unwrap(), [10]);
}

#[test]
fn fallback_runs_finalize_as_a_unit_without_rewriting_emitted_text() {
    let fixture = Fixture::new();
    let processor = fixture.processor(limits()).unwrap();
    let byte = |n: u32| 12 + n;
    for (ids, expected) in [
        (vec![byte(0xC3), byte(0xA9)], "é"),
        (vec![byte(0x61), byte(0xFF)], "��"),
        (vec![byte(0xC3), byte(0xA9), byte(0xFF)], "���"),
        (vec![byte(0xC3)], "�"),
        (vec![byte(0xE2), byte(0x96), byte(0x81)], "▁"),
        (vec![6, byte(0xC3), 10, byte(0xA9)], " �X�"),
        (vec![2, 1, 4, 5], "<bos><eos><turn|><|tool_response>"),
    ] {
        assert_eq!(processor.decode(&ids).unwrap(), expected);
        let mut stream = processor.decoder();
        let mut output = String::new();
        for &id in &ids {
            let piece = stream.push(id).unwrap();
            output.push_str(&piece);
            assert!(expected.starts_with(&output), "fragments are irrevocable");
        }
        output.push_str(&stream.finish().unwrap());
        assert_eq!(output, expected);
    }
    let mut stream = processor.decoder();
    assert_eq!(stream.push(byte(0x61)).unwrap(), "");
    assert_eq!(stream.push(byte(0xFF)).unwrap(), "");
    assert_eq!(stream.push(10).unwrap(), "��X");
    assert_eq!(stream.finish().unwrap(), "");
}

#[test]
fn request_bounds_unknown_ids_and_media_reject_before_success() {
    let fixture = Fixture::new();
    let processor = fixture
        .processor(TextLimits {
            input_bytes: 8,
            rendered_bytes: 12,
            tokens: 4,
            messages: 1,
        })
        .unwrap();
    assert!(processor.encode("123456789").is_err());
    assert!(processor.validate_input(&[10; 5]).is_err());
    for id in [7, 8, 9, 268] {
        assert!(processor.validate_input(&[id]).is_err());
    }
    for text in ["<|image|>", "<|audio|>", "<|video|>"] {
        assert!(fixture.processor(limits()).unwrap().encode(text).is_err());
    }
    assert!(processor.render_chat(&[]).is_err());
    assert!(
        processor
            .render_chat(&[TextMessage {
                role: "tool",
                content: ""
            }])
            .is_err()
    );
    assert!(
        processor
            .render_chat(
                &[TextMessage {
                    role: "user",
                    content: "a"
                }; 2]
            )
            .is_err()
    );
    assert!(
        processor
            .render_chat(&[TextMessage {
                role: "user",
                content: "a"
            }])
            .is_err(),
        "rendered bytes, not merely source bytes"
    );
    assert!(processor.decode(&[268]).is_err());
    let mut stream = processor.decoder();
    assert!(stream.push(268).is_err());
    assert!(stream.push(10).is_err());
    assert!(stream.finish().is_err());
    let tight = fixture
        .processor(TextLimits {
            rendered_bytes: 1,
            ..limits()
        })
        .unwrap();
    let mut stream = tight.decoder();
    assert_eq!(stream.push(12 + 0xC3).unwrap(), "");
    assert!(stream.finish().is_err());
    let mut stream = processor.decoder();
    for _ in 0..4 {
        stream.push(10).unwrap();
    }
    assert!(stream.push(10).is_err());
    let real = Gemma4Config::from_json(
        &serde_json::from_str(include_str!(
            "../../../benchmarks/gemma4-reference/config.json"
        ))
        .unwrap(),
    )
    .unwrap();
    real.validate_text_tokens(&[5, 255_999, 256_000, 258_882, 258_883])
        .unwrap();
    for id in [258_880, 258_881, 258_884, 262_144] {
        assert!(real.validate_text_tokens(&[id]).is_err());
    }
}

#[test]
fn malformed_or_overridden_assets_and_invalid_limits_never_publish_a_processor() {
    let fixture = Fixture::new();
    assert!(
        fixture
            .processor(TextLimits {
                tokens: 0,
                ..limits()
            })
            .is_err()
    );
    assert!(
        fixture
            .processor(TextLimits {
                tokens: 131_073,
                ..limits()
            })
            .is_err()
    );
    for value in [json!([1, 1]), json!([]), json!([268]), json!(1)] {
        fixture.change("generation_config.json", |v| v["eos_token_id"] = value);
        assert!(fixture.processor(limits()).is_err());
    }
    fixture.change("generation_config.json", |v| {
        v["eos_token_id"] = json!([1, 4, 5]);
    });
    fixture.change("tokenizer.json", |v| v["model"]["dropout"] = json!(0.1));
    assert!(fixture.processor(limits()).is_err());
    fixture.change("tokenizer.json", |v| v["model"]["dropout"] = Value::Null);
    fixture.change("tokenizer_config.json", |v| v["bos_token"] = json!("<unk>"));
    assert!(fixture.processor(limits()).is_err());
    fixture.change("tokenizer_config.json", |v| v["bos_token"] = json!("<bos>"));
    std::fs::create_dir(fixture.0.join("chat_templates")).unwrap();
    assert!(fixture.processor(limits()).is_err());
    std::fs::remove_dir(fixture.0.join("chat_templates")).unwrap();
    std::fs::write(fixture.0.join("chat_template.jinja"), "{% broken %}").unwrap();
    assert!(fixture.processor(limits()).is_err());
    let file = std::fs::File::create(fixture.0.join("chat_template.jinja")).unwrap();
    file.set_len(1024 * 1024 + 1).unwrap();
    assert!(fixture.processor(limits()).is_err());
}
