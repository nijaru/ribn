//! Real-artifact Inkling 2 Mini GGUF preflight, oracle tokenization, and CPU forward checks.
//!
//! These tests are `#[ignore]`d by default so host CI does not require the 16.4 GiB
//! GGUF artifact. Run explicitly on a host that has the pinned artifact and oracle:
//!
//! ```sh
//! RIBN_INKLING_GGUF=/path/to/Inkling-2-Mini-IQ4_NL.gguf \
//! RIBN_INKLING_ORACLE=/path/to/inkling-oracle-e6d624b/logits \
//!   cargo test -p engine-inkling2 --release --test real_artifact -- --ignored --nocapture --test-threads=1
//! ```

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::fs;
use std::path::PathBuf;

use engine_inkling2::{ArchitectureKind, Inkling2CpuModel, Inkling2Source, KvCachePrecision};

fn env_path(var: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(var)
            .unwrap_or_else(|_| panic!("set {var} to run the real-artifact Inkling 2 test")),
    )
}

fn read_oracle_tokens(oracle_dir: &std::path::Path) -> Vec<u32> {
    let text = fs::read_to_string(oracle_dir.join("tokens.txt")).expect("read tokens.txt");
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.trim().parse::<u32>().expect("token id"))
        .collect()
}

fn read_oracle_logits(oracle_dir: &std::path::Path, file: &str) -> Vec<f32> {
    let bytes = fs::read(oracle_dir.join(file)).expect("read oracle logits");
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(idx, _)| idx)
}

fn centered_rmse_and_max_abs(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let mean_a = a.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let mean_b = b.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let mut sq_centered = 0.0_f64;
    let mut max_abs = 0.0_f32;
    let mut max_centered = 0.0_f32;
    for (&va, &vb) in a.iter().zip(b) {
        let diff = (va - vb).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        let cdiff = (f64::from(va) - mean_a) - (f64::from(vb) - mean_b);
        sq_centered += cdiff * cdiff;
        if (cdiff.abs() as f32) > max_centered {
            max_centered = cdiff.abs() as f32;
        }
    }
    ((sq_centered / n).sqrt() as f32, max_abs, max_centered)
}

#[test]
#[ignore = "requires local Inkling-2-Mini-IQ4_NL.gguf and oracle logits"]
fn preflight_and_oracle_tokenization_match_pinned_artifact() {
    let gguf_path = env_path("RIBN_INKLING_GGUF");
    let oracle_dir = env_path("RIBN_INKLING_ORACLE");

    let source = Inkling2Source::open(&gguf_path).expect("preflight Inkling 2 Mini GGUF");
    let cfg = source.config();
    assert_eq!(cfg.architecture(), ArchitectureKind::Looped);
    assert_eq!(cfg.physical_layer_count(), 24);
    assert_eq!(cfg.state_slot_count(), 48);
    assert_eq!(source.tensor_count(), 704);
    assert_eq!(cfg.vocab_unpadded(), 200_058);
    assert_eq!(cfg.vocab_padded(), 201_024);

    // Exact tokenization check against replay.cpp's oracle text and tokens.txt
    let oracle_tokens = read_oracle_tokens(&oracle_dir);
    let text = "In the archive, the blue notebook belongs to Ada. The red one belongs to Lin.\n\
                Who owns the blue notebook? Answer: Ada.\n";
    let piece = source
        .tokenizer()
        .encode(text)
        .expect("encode oracle text with Inkling 2 tokenizer");
    assert!(!piece.is_empty());
    let repeated: Vec<u32> = (0..oracle_tokens.len())
        .map(|i| piece[i % piece.len()])
        .collect();
    assert_eq!(
        repeated, oracle_tokens,
        "Inkling 2 tokenizer output must match pinned llama.cpp tokens.txt exactly"
    );
    eprintln!(
        "preflight ok: {} tensors, 48 state slots, tokenizer exact match ({} tokens, piece len {})",
        source.tensor_count(),
        oracle_tokens.len(),
        piece.len()
    );
}

#[test]
#[ignore = "requires local Inkling-2-Mini-IQ4_NL.gguf and oracle logits (~17 GiB RAM)"]
fn cpu_forward_matches_pinned_oracle_prefix_checkpoints() {
    let gguf_path = env_path("RIBN_INKLING_GGUF");
    let oracle_dir = env_path("RIBN_INKLING_ORACLE");

    let source = Inkling2Source::open(&gguf_path).expect("preflight");
    let oracle_tokens = read_oracle_tokens(&oracle_dir);

    let t0 = std::time::Instant::now();
    let model = Inkling2CpuModel::load_from_source(&source).expect("load CPU model");
    eprintln!(
        "loaded Inkling2CpuModel in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    let mut state = model.new_state(KvCachePrecision::F16);

    for (start, end, oracle_file) in [
        (0_usize, 8_usize, "64-8.f32"),
        (8_usize, 32_usize, "64-32.f32"),
    ] {
        let step_t0 = std::time::Instant::now();
        let logits = model
            .forward_tokens(&mut state, &oracle_tokens[start..end], true)
            .expect("forward checkpoint")
            .expect("logits");
        let elapsed = step_t0.elapsed().as_secs_f64();

        assert_eq!(logits.len(), 200_058);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");

        let oracle = read_oracle_logits(&oracle_dir, oracle_file);
        let got_top = argmax(&logits);
        let ref_top = argmax(&oracle);
        let (c_rmse, max_abs, max_centered) = centered_rmse_and_max_abs(&logits, &oracle);
        eprintln!(
            "checkpoint {end:4} ({elapsed:.2}s): argmax got={got_top} ref={ref_top} | centered_rmse={c_rmse:.5} max_abs={max_abs:.5} max_centered={max_centered:.5}"
        );
        assert_eq!(
            got_top, ref_top,
            "argmax mismatch at checkpoint {end}: got {got_top} vs ref {ref_top}"
        );
        assert!(
            c_rmse < 0.25,
            "centered logit RMSE {c_rmse} too large at checkpoint {end}"
        );
    }
}

#[test]
#[ignore = "requires local HF thinkingmachines/Inkling-2-Mini snapshot and oracle tokens.txt"]
#[allow(clippy::too_many_lines)]
fn hf_tokenizer_and_chat_template_match_pinned_oracle_on_host() {
    use engine_gguf::{ChatMessage, ChatTemplateOptions, GgufTokenizer, MetadataValue};
    use std::collections::BTreeMap;

    let hf_dir = env_path("RIBN_INKLING_HF_DIR");
    let oracle_dir = env_path("RIBN_INKLING_ORACLE");

    let tok_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(hf_dir.join("tokenizer.json")).expect("read tokenizer.json"),
    )
    .expect("parse tokenizer.json");
    let template =
        fs::read_to_string(hf_dir.join("chat_template.jinja")).expect("read chat_template.jinja");

    let mut tokens = vec![String::new(); 200_058];
    let mut token_types = vec![1_i32; 200_058];
    for (piece, id_val) in tok_json["model"]["vocab"]
        .as_object()
        .expect("vocab object")
    {
        let id = usize::try_from(id_val.as_u64().expect("id")).unwrap();
        tokens[id].clone_from(piece);
    }
    for entry in tok_json["added_tokens"].as_array().expect("added_tokens") {
        let id = usize::try_from(entry["id"].as_u64().expect("id")).unwrap();
        tokens[id] = entry["content"].as_str().expect("content").to_owned();
        // Control / unused tokens match GGUF converter types (3 = control, 5 = unused)
        token_types[id] = if tokens[id].starts_with("<|unused") {
            5
        } else {
            3
        };
    }
    let merges: Vec<String> = tok_json["model"]["merges"]
        .as_array()
        .expect("merges array")
        .iter()
        .map(|m| {
            if let Some(s) = m.as_str() {
                s.to_owned()
            } else {
                let arr = m.as_array().expect("pair array");
                format!(
                    "{} {}",
                    arr[0].as_str().expect("left"),
                    arr[1].as_str().expect("right")
                )
            }
        })
        .collect();

    let mut metadata = BTreeMap::new();
    metadata.insert(
        "tokenizer.ggml.model".to_owned(),
        MetadataValue::String("gpt2".to_owned()),
    );
    metadata.insert(
        "tokenizer.ggml.pre".to_owned(),
        MetadataValue::String("inkling2".to_owned()),
    );
    metadata.insert(
        "tokenizer.ggml.tokens".to_owned(),
        MetadataValue::Array(tokens.into_iter().map(MetadataValue::String).collect()),
    );
    metadata.insert(
        "tokenizer.ggml.token_type".to_owned(),
        MetadataValue::Array(token_types.into_iter().map(MetadataValue::I32).collect()),
    );
    metadata.insert(
        "tokenizer.ggml.merges".to_owned(),
        MetadataValue::Array(merges.into_iter().map(MetadataValue::String).collect()),
    );
    metadata.insert(
        "tokenizer.ggml.eos_token_id".to_owned(),
        MetadataValue::U32(200_006),
    );
    metadata.insert(
        "tokenizer.ggml.padding_token_id".to_owned(),
        MetadataValue::U32(199_999),
    );
    metadata.insert(
        "tokenizer.chat_template".to_owned(),
        MetadataValue::String(template),
    );

    let tokenizer = GgufTokenizer::from_metadata(&metadata).expect("build Inkling 2 GgufTokenizer");
    assert_eq!(tokenizer.bos_token_id(), None);

    // 1. Exact match against all 1,036 pinned llama.cpp oracle tokens!
    let oracle_tokens = read_oracle_tokens(&oracle_dir);
    let text = "In the archive, the blue notebook belongs to Ada. The red one belongs to Lin.\n\
                Who owns the blue notebook? Answer: Ada.\n";
    let piece = tokenizer.encode(text).expect("encode oracle text");
    let repeated: Vec<u32> = (0..oracle_tokens.len())
        .map(|i| piece[i % piece.len()])
        .collect();
    assert_eq!(
        repeated, oracle_tokens,
        "engine-gguf inkling2 tokenizer must match pinned llama.cpp tokens.txt bit-for-bit"
    );
    assert_eq!(tokenizer.decode(&piece).expect("roundtrip decode"), text);

    // 2. Real Inkling 2 Mini chat_template.jinja rendering & special-token encoding!
    let messages = [
        ChatMessage::new("system", "Be concise."),
        ChatMessage::new("user", "Hello!"),
        ChatMessage::new("assistant", "Hi there."),
        ChatMessage::new("user", "How are you?"),
    ];
    let rendered = tokenizer
        .render_chat(&messages, ChatTemplateOptions::new(true, false))
        .expect("render real Inkling 2 chat template");
    assert_eq!(
        rendered,
        "<|message_system|><|content_text|>Be concise.<|end_message|>\
         <|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>\
         <|message_user|><|content_text|>Hello!<|end_message|>\
         <|message_model|><|content_text|>Hi there.<|end_message|><|content_model_end_sampling|>\
         <|message_user|><|content_text|>How are you?<|end_message|>"
    );
    let chat_ids = tokenizer
        .encode_rendered(&rendered)
        .expect("encode rendered Inkling 2 chat");
    assert_eq!(chat_ids[0], 200_002); // <|message_system|>
    assert_eq!(chat_ids[1], 200_004); // <|content_text|>
    assert!(chat_ids.contains(&200_000)); // <|message_user|>
    assert!(chat_ids.contains(&200_001)); // <|message_model|>
    assert!(chat_ids.contains(&200_006)); // <|content_model_end_sampling|>
    assert!(chat_ids.contains(&200_010)); // <|end_message|>
    eprintln!(
        "Mac HF verification PASSED: {}/{} oracle tokens exact, real chat_template.jinja rendered ({} tokens)",
        repeated.len(),
        oracle_tokens.len(),
        chat_ids.len()
    );
}
