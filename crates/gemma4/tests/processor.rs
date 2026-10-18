use engine_gemma4::{DecodeLimits, Gemma4Config, Gemma4TextProcessor, TextLimits, TextMessage};
use ribn_hf::LocalModelPackage;
use serde_json::{Value, json};
#[path = "common/processor.rs"]
mod processor_fixture;
use processor_fixture::Fixture;
impl Fixture {
    fn processor(
        &self,
        limits: TextLimits,
    ) -> Result<Gemma4TextProcessor, engine_gemma4::Gemma4Error> {
        Gemma4TextProcessor::open(&LocalModelPackage::open(&self.0).unwrap(), limits)
    }
}

fn decode_limits(tokens: usize, delta_bytes: usize) -> DecodeLimits {
    DecodeLimits {
        tokens,
        token_bytes: delta_bytes,
        pending_bytes: tokens,
        delta_bytes,
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
    let mut decoder = processor.decoder(decode_limits(2, 16)).unwrap();
    drop(processor);
    assert_eq!(decoder.push(10).unwrap(), "X");
    assert_eq!(decoder.finish().unwrap(), "");
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
        let mut stream = processor
            .decoder(decode_limits(ids.len(), 131_072))
            .unwrap();
        let mut output = String::new();
        for &id in &ids {
            let piece = stream.push(id).unwrap();
            output.push_str(&piece);
            assert!(expected.starts_with(&output), "fragments are irrevocable");
        }
        output.push_str(&stream.finish().unwrap());
        assert_eq!(output, expected);
    }
    let mut stream = processor.decoder(decode_limits(3, 131_072)).unwrap();
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
    let mut stream = processor.decoder(decode_limits(4, 12)).unwrap();
    assert!(stream.push(268).is_err());
    assert!(stream.push(10).is_err());
    assert!(stream.finish().is_err());
    let tight = fixture
        .processor(TextLimits {
            rendered_bytes: 1,
            ..limits()
        })
        .unwrap();
    let mut stream = tight.decoder(decode_limits(4, 1)).unwrap();
    assert_eq!(stream.push(12 + 0xC3).unwrap(), "");
    assert!(stream.finish().is_err());
    let mut stream = processor.decoder(decode_limits(4, 12)).unwrap();
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
fn decoder_budget_is_independent_of_input_limits_and_rejects_infeasible_storage() {
    let fixture = Fixture::new();
    let processor = fixture
        .processor(TextLimits {
            tokens: 4,
            ..limits()
        })
        .unwrap();
    for invalid in [
        DecodeLimits {
            tokens: 0,
            ..decode_limits(4, 12)
        },
        DecodeLimits {
            pending_bytes: 3,
            ..decode_limits(4, 12)
        },
        DecodeLimits {
            token_bytes: 0,
            ..decode_limits(4, 12)
        },
        DecodeLimits {
            delta_bytes: 0,
            ..decode_limits(4, 12)
        },
        DecodeLimits {
            tokens: usize::MAX,
            pending_bytes: usize::MAX,
            ..decode_limits(4, 12)
        },
    ] {
        assert!(processor.decoder(invalid).is_err());
    }
    let mut decoder = processor.decoder(decode_limits(8, 12)).unwrap();
    for _ in 0..8 {
        assert_eq!(decoder.push(10).unwrap(), "X");
    }
    assert!(decoder.push(10).is_err());
    assert!(decoder.finish().is_err());
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
