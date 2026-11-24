//! End-to-end synthetic GGUF v3 preflight, mixed-encoding (`F32`/`Q8_0`/`IQ4_NL`)
//! CPU loading, and `ribn::Engine` + `ribn-text` runtime lifecycle tests for Inkling 2 Loop.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use engine_gguf::byte_token_symbol;
use engine_inkling2::cpu::f32_to_f16;
use engine_inkling2::{
    ArchitectureKind, Inkling2Config, Inkling2CpuModel, Inkling2Executor, Inkling2Source,
    KvCachePrecision, ParameterId, parameters,
};
use ribn::driver::{Driver, DriverConfig};
use ribn::{Engine, FinishReason, GenerationOptions, Sampling};
use ribn_text::{
    GgufProcessor, Message, ProcessorLimits, TextConfig, TextEvent, TextInput, TextOwner,
};

const HIDDEN: usize = 32;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const REL_DIM: usize = 4;
const REL_EXTENT: usize = 8;
const SLIDING_WINDOW: usize = 5;
const BLOCK_SIZE: usize = 2;
const CONV_KERNEL: usize = 4;
const EXPERTS: usize = 4;
const EXPERTS_USED: usize = 2;
const FF_EXP: usize = 32;
const VOCAB_UNPADDED: usize = 262; // 256 byte symbols + 6 Inkling control markers
const VOCAB_PADDED: usize = 288;
const CORE_START: usize = 2;
const CORE_END: usize = 4;
const PASSES: usize = 3;
const GATE_RANK: usize = 4;

const CTRL_USER: u32 = 256;
const CTRL_MODEL: u32 = 257;
const CTRL_SYSTEM: u32 = 258;
const CTRL_TEXT: u32 = 259;
const CTRL_END_MSG: u32 = 260;
const CTRL_EOS: u32 = 261;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_gguf_path(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "ribn-inkling2-{tag}-{}-{n}.gguf",
        std::process::id()
    ))
}

enum SyntheticMutation {
    None,
    DropTensor(&'static str),
    CorruptShape(&'static str),
    AddUnexpectedTensor,
    WrongPretokenizer,
}

fn push_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn push_kv_str(buf: &mut Vec<u8>, k: &str, v: &str) {
    push_str(buf, k);
    buf.extend_from_slice(&8_u32.to_le_bytes());
    push_str(buf, v);
}

fn push_kv_u32(buf: &mut Vec<u8>, k: &str, v: u32) {
    push_str(buf, k);
    buf.extend_from_slice(&4_u32.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_kv_f32(buf: &mut Vec<u8>, k: &str, v: f32) {
    push_str(buf, k);
    buf.extend_from_slice(&6_u32.to_le_bytes());
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_kv_bool(buf: &mut Vec<u8>, k: &str, v: bool) {
    push_str(buf, k);
    buf.extend_from_slice(&7_u32.to_le_bytes());
    buf.push(u8::from(v));
}

fn push_kv_bool_arr(buf: &mut Vec<u8>, k: &str, vals: &[bool]) {
    push_str(buf, k);
    buf.extend_from_slice(&9_u32.to_le_bytes());
    buf.extend_from_slice(&7_u32.to_le_bytes());
    buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
    for &v in vals {
        buf.push(u8::from(v));
    }
}

fn push_kv_i32_arr(buf: &mut Vec<u8>, k: &str, vals: &[i32]) {
    push_str(buf, k);
    buf.extend_from_slice(&9_u32.to_le_bytes());
    buf.extend_from_slice(&5_u32.to_le_bytes());
    buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
    for &v in vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn push_kv_str_arr(buf: &mut Vec<u8>, k: &str, vals: &[String]) {
    push_str(buf, k);
    buf.extend_from_slice(&9_u32.to_le_bytes());
    buf.extend_from_slice(&8_u32.to_le_bytes());
    buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
    for v in vals {
        push_str(buf, v);
    }
}

fn encode_q8_0(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert!(cols.is_multiple_of(32));
    let mut out = Vec::with_capacity(rows * (cols / 32) * 34);
    for r in 0..rows {
        for b in 0..(cols / 32) {
            let scale = f32_to_f16(0.015_625);
            out.extend_from_slice(&scale.to_le_bytes());
            for j in 0..32 {
                let mix = r
                    .wrapping_mul(37)
                    .wrapping_add(r.wrapping_mul(r))
                    .wrapping_add(b * 7 + j * 11 + seed);
                let v = ((mix % 63) as i8) - 31;
                out.push(v.cast_unsigned());
            }
        }
    }
    out
}

fn encode_iq4_nl(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert!(cols.is_multiple_of(32));
    let mut out = Vec::with_capacity(rows * (cols / 32) * 18);
    for r in 0..rows {
        for b in 0..(cols / 32) {
            let scale = f32_to_f16(0.003_906_25);
            out.extend_from_slice(&scale.to_le_bytes());
            for j in 0..16 {
                let lo = ((r * 5 + b * 3 + j + seed) % 16) as u8;
                let hi = ((r * 7 + b * 11 + j * 3 + seed) % 16) as u8;
                out.push(lo | (hi << 4));
            }
        }
    }
    out
}

fn encode_f32(elems: usize, seed: usize, unit: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(elems * 4);
    for i in 0..elems {
        let v = if unit {
            1.0_f32
        } else {
            (((i * 31 + seed * 13) % 997) as f32 / 997.0 - 0.5) * 0.25
        };
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[allow(clippy::too_many_lines)]
fn write_synthetic_inkling2_gguf(path: &Path, mutation: &SyntheticMutation) {
    // Build matching config metadata first so we can derive exact tensor specs.
    let mut meta_bytes = Vec::new();
    let mut kv_count = 0_u64;
    macro_rules! kv {
        ($f:ident, $k:expr, $v:expr) => {{
            $f(&mut meta_bytes, $k, $v);
            kv_count += 1;
        }};
    }

    kv!(push_kv_str, "general.architecture", "inkling2_loop");
    kv!(push_kv_u32, "inkling2_loop.block_count", 6);
    kv!(push_kv_u32, "inkling2_loop.context_length", 64);
    kv!(push_kv_u32, "inkling2_loop.embedding_length", HIDDEN as u32);
    kv!(
        push_kv_i32_arr,
        "inkling2_loop.attention.head_count",
        &[HEADS as i32; 6]
    );
    kv!(
        push_kv_i32_arr,
        "inkling2_loop.attention.head_count_kv",
        &[KV_HEADS as i32; 6]
    );
    kv!(
        push_kv_u32,
        "inkling2_loop.attention.key_length",
        HEAD_DIM as u32
    );
    kv!(
        push_kv_u32,
        "inkling2_loop.attention.value_length",
        HEAD_DIM as u32
    );
    kv!(
        push_kv_f32,
        "inkling2_loop.attention.layer_norm_rms_epsilon",
        1e-6
    );
    kv!(push_kv_f32, "inkling2_loop.attention.qk_clip", 22.627_417);
    kv!(
        push_kv_u32,
        "inkling2_loop.attention.relative_length",
        REL_DIM as u32
    );
    kv!(
        push_kv_u32,
        "inkling2_loop.attention.relative_extent",
        REL_EXTENT as u32
    );
    kv!(push_kv_f32, "inkling2_loop.attention.relative_clip", 8.0);
    kv!(push_kv_u32, "inkling2_loop.attention.log_scale_floor", 8);
    kv!(push_kv_f32, "inkling2_loop.attention.log_scale_alpha", 0.1);
    kv!(
        push_kv_u32,
        "inkling2_loop.attention.sliding_window",
        SLIDING_WINDOW as u32
    );
    kv!(
        push_kv_bool_arr,
        "inkling2_loop.attention.sliding_window_pattern",
        &[true, false, true, false, true, false]
    );
    kv!(
        push_kv_u32,
        "inkling2_loop.attn_res.block_size",
        BLOCK_SIZE as u32
    );
    kv!(
        push_kv_u32,
        "inkling2_loop.shortconv.l_cache",
        CONV_KERNEL as u32
    );
    kv!(push_kv_u32, "inkling2_loop.vocab_size", VOCAB_PADDED as u32);
    kv!(
        push_kv_u32,
        "inkling2_loop.vocab_size_unpadded",
        VOCAB_UNPADDED as u32
    );
    kv!(push_kv_u32, "inkling2_loop.expert_count", EXPERTS as u32);
    kv!(
        push_kv_u32,
        "inkling2_loop.expert_used_count",
        EXPERTS_USED as u32
    );
    kv!(
        push_kv_u32,
        "inkling2_loop.expert_feed_forward_length",
        FF_EXP as u32
    );
    kv!(push_kv_f32, "inkling2_loop.expert_weights_scale", 8.0);
    kv!(push_kv_f32, "inkling2_loop.activation.situ_beta", 4.0);
    kv!(
        push_kv_f32,
        "inkling2_loop.activation.situ_linear_beta",
        25.0
    );
    kv!(push_kv_f32, "inkling2_loop.final_logit_softcapping", 30.0);
    kv!(push_kv_u32, "inkling2_loop.rope.dimension_count", 0);
    kv!(push_kv_u32, "inkling2_loop.loop.start", CORE_START as u32);
    kv!(push_kv_u32, "inkling2_loop.loop.end", CORE_END as u32);
    kv!(push_kv_u32, "inkling2_loop.loop.count", PASSES as u32);
    kv!(
        push_kv_u32,
        "inkling2_loop.loop.gate_rank",
        GATE_RANK as u32
    );
    kv!(push_kv_bool, "inkling2_loop.loop.varying", false);

    // Embedded Inkling 2 GPT-2/BPE tokenizer (deliberately no bos_token_id!)
    let mut vocab: Vec<String> = (0..=255_u8)
        .map(|b| byte_token_symbol(b).to_string())
        .collect();
    vocab.extend(
        [
            "<|message_user|>",
            "<|message_model|>",
            "<|message_system|>",
            "<|content_text|>",
            "<|end_message|>",
            "<|content_model_end_sampling|>",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    assert_eq!(vocab.len(), VOCAB_UNPADDED);
    let mut token_types = vec![1_i32; 256];
    token_types.extend([3_i32; 6]);

    let pre = match mutation {
        SyntheticMutation::WrongPretokenizer => "qwen35",
        _ => "inkling2",
    };
    let chat_template = "{%- for m in messages -%}\
        {%- set tok = '<|message_user|>' if m.role == 'user' else ('<|message_model|>' if m.role == 'assistant' else '<|message_system|>') -%}\
        {%- set rendered -%}{{- tok ~ '<|content_text|>' ~ m.content ~ '<|end_message|>' -}}{%- if m.role == 'assistant' -%}<|content_model_end_sampling|>{%- endif -%}{%- endset -%}\
        {%- if m.role == 'assistant' -%}{%- generation -%}{{- rendered -}}{%- endgeneration -%}{%- else -%}{{- rendered -}}{%- endif -%}\
        {%- endfor -%}";

    kv!(push_kv_str, "tokenizer.ggml.model", "gpt2");
    kv!(push_kv_str, "tokenizer.ggml.pre", pre);
    kv!(push_kv_str_arr, "tokenizer.ggml.tokens", &vocab);
    kv!(push_kv_i32_arr, "tokenizer.ggml.token_type", &token_types);
    kv!(push_kv_str_arr, "tokenizer.ggml.merges", &[]);
    kv!(push_kv_u32, "tokenizer.ggml.eos_token_id", CTRL_EOS);
    kv!(push_kv_str, "tokenizer.chat_template", chat_template);

    // Derive canonical tensor inventory from a temporary header parse or directly:
    let mut header_only = Vec::new();
    header_only.extend_from_slice(b"GGUF");
    header_only.extend_from_slice(&3_u32.to_le_bytes());
    header_only.extend_from_slice(&0_u64.to_le_bytes());
    header_only.extend_from_slice(&kv_count.to_le_bytes());
    header_only.extend_from_slice(&meta_bytes);
    header_only.resize(header_only.len().next_multiple_of(32), 0);
    let tmp_meta_path = temp_gguf_path("meta-only");
    fs::write(&tmp_meta_path, &header_only).unwrap();
    let cfg =
        Inkling2Config::from_gguf(&engine_gguf::GgufFile::open(&tmp_meta_path).unwrap()).unwrap();
    fs::remove_file(&tmp_meta_path).unwrap();

    // Prepare encoded tensor payloads using mixed F32 + Q8_0 + IQ4_NL matching Inkling 2 Mini!
    let mut entries: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    for (idx, spec) in parameters::specifications(&cfg).into_iter().enumerate() {
        if let SyntheticMutation::DropTensor(target) = mutation
            && spec.name == *target
        {
            continue;
        }
        let mut dims = spec.dims.clone();
        if let SyntheticMutation::CorruptShape(target) = mutation
            && spec.name == *target
        {
            dims[0] += 1;
        }
        let ne0 = usize::try_from(dims[0]).unwrap();
        let rows: usize = dims[1..]
            .iter()
            .map(|&d| usize::try_from(d).unwrap())
            .product();

        let (ggml_type, payload) = match spec.id {
            ParameterId::Global(
                engine_inkling2::GlobalParameter::TokenEmbedding
                | engine_inkling2::GlobalParameter::Output,
            )
            | ParameterId::Layer {
                parameter:
                    engine_inkling2::LayerParameter::AttnQ
                    | engine_inkling2::LayerParameter::AttnK
                    | engine_inkling2::LayerParameter::AttnV
                    | engine_inkling2::LayerParameter::AttnOutput
                    | engine_inkling2::LayerParameter::AttnGate
                    | engine_inkling2::LayerParameter::AttnRel
                    | engine_inkling2::LayerParameter::AttnSinkProj
                    | engine_inkling2::LayerParameter::AttnKConvProj
                    | engine_inkling2::LayerParameter::AttnVConvProj,
                ..
            } => (8_u32, encode_q8_0(rows, ne0, idx + 1)),
            ParameterId::Layer {
                layer,
                parameter:
                    engine_inkling2::LayerParameter::FfnGateUpExps
                    | engine_inkling2::LayerParameter::FfnDownExps,
            } => {
                // Layers 0..4 use IQ4_NL; layers 4..6 use Q8_0 (matching Mini's 20 IQ4_NL + 4 Q8_0 split)
                if layer < 4 {
                    (20_u32, encode_iq4_nl(rows, ne0, idx + 1))
                } else {
                    (8_u32, encode_q8_0(rows, ne0, idx + 1))
                }
            }
            _ => {
                let unit = spec.name.contains("norm") || spec.name.contains("_clip");
                (0_u32, encode_f32(ne0 * rows, idx + 1, unit))
            }
        };
        entries.push((spec.name, dims, ggml_type, payload));
    }

    if matches!(mutation, SyntheticMutation::AddUnexpectedTensor) {
        entries.push((
            "unexpected.extra.weight".into(),
            vec![HIDDEN as u64],
            0_u32,
            encode_f32(HIDDEN, 999, false),
        ));
    }

    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(b"GGUF");
    file_bytes.extend_from_slice(&3_u32.to_le_bytes());
    file_bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(&kv_count.to_le_bytes());
    file_bytes.extend_from_slice(&meta_bytes);

    let mut data_section = Vec::new();
    for (name, dims, ggml_type, payload) in entries {
        let aligned = data_section.len().next_multiple_of(32);
        data_section.resize(aligned, 0);
        let offset = data_section.len() as u64;
        data_section.extend_from_slice(&payload);

        push_str(&mut file_bytes, &name);
        file_bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            file_bytes.extend_from_slice(&d.to_le_bytes());
        }
        file_bytes.extend_from_slice(&ggml_type.to_le_bytes());
        file_bytes.extend_from_slice(&offset.to_le_bytes());
    }
    let dir_aligned = file_bytes.len().next_multiple_of(32);
    file_bytes.resize(dir_aligned, 0);
    file_bytes.extend_from_slice(&data_section);
    fs::write(path, file_bytes).unwrap();
}

#[test]
fn preflight_validates_mixed_encoding_gguf_retains_descriptor_and_rejects_defects() {
    let valid_path = temp_gguf_path("valid");
    write_synthetic_inkling2_gguf(&valid_path, &SyntheticMutation::None);

    let source = Inkling2Source::open(&valid_path).expect("preflight valid synthetic GGUF");
    assert_eq!(source.config().architecture(), ArchitectureKind::Looped);
    assert_eq!(source.config().state_slot_count(), 10);

    // Unlink path after open: pinned descriptor must still load every tensor cleanly!
    fs::remove_file(&valid_path).unwrap();
    let model = Inkling2CpuModel::load_from_source(&source).expect("load after unlink");
    let mut st = model.new_state(KvCachePrecision::F16);
    let logits = model
        .forward_tokens(&mut st, &[65, 66, 67], true)
        .unwrap()
        .unwrap();
    assert_eq!(logits.len(), VOCAB_UNPADDED);
    assert!(logits.iter().all(|v| v.is_finite()));

    // Adversarial preflight rejections
    for (tag, mutation) in [
        (
            "missing",
            SyntheticMutation::DropTensor("blk.2.loop_msg_query.weight"),
        ),
        (
            "shape",
            SyntheticMutation::CorruptShape("blk.0.attn_k_conv.weight"),
        ),
        ("extra", SyntheticMutation::AddUnexpectedTensor),
        ("pretok", SyntheticMutation::WrongPretokenizer),
    ] {
        let bad_path = temp_gguf_path(tag);
        write_synthetic_inkling2_gguf(&bad_path, &mutation);
        assert!(
            Inkling2Source::open(&bad_path).is_err(),
            "mutation {tag} must fail preflight"
        );
        fs::remove_file(&bad_path).unwrap();
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn executor_concurrent_continuation_cancellation_and_chat_preserve_isolation() {
    let path = temp_gguf_path("runtime");
    write_synthetic_inkling2_gguf(&path, &SyntheticMutation::None);
    let source = Inkling2Source::open(&path).unwrap();
    let tokenizer = source.tokenizer().clone();
    let model = Arc::new(Inkling2CpuModel::load_from_source(&source).unwrap());
    fs::remove_file(&path).unwrap();

    // Reference isolated output for prompt A & prompt B (crossing SWA window = 5!)
    let prompt_a = b"Hello world!"
        .iter()
        .map(|&b| u32::from(b))
        .collect::<Vec<_>>();
    let prompt_b = b"Ada owns blue."
        .iter()
        .map(|&b| u32::from(b))
        .collect::<Vec<_>>();
    let ref_gen = |prompt: &[u32], steps: usize| -> Vec<u32> {
        let mut st = model.new_state(KvCachePrecision::F16);
        let mut out = Vec::new();
        let mut logits = model
            .forward_tokens(&mut st, prompt, true)
            .unwrap()
            .unwrap();
        for step in 0..steps {
            let tok = logits
                .iter()
                .enumerate()
                .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
                .map(|(i, _)| i as u32)
                .unwrap();
            out.push(tok);
            if tok == CTRL_EOS {
                break;
            }
            if step + 1 < steps {
                logits = model
                    .forward_tokens(&mut st, &[tok], true)
                    .unwrap()
                    .unwrap();
            }
        }
        out
    };
    let expected_a = ref_gen(&prompt_a, 4);
    let expected_b = ref_gen(&prompt_b, 4);

    // Real ribn::Engine + Driver + TextOwner + GgufProcessor concurrent run + cancellation!
    let executor = Inkling2Executor::new(Arc::clone(&model), 2, 48, KvCachePrecision::F16).unwrap();
    let engine = Engine::with_defaults(executor).unwrap();
    let driver_cfg = DriverConfig::for_engine(&engine);
    let (shutdown, handle) = Driver::spawn(engine, driver_cfg).unwrap();
    let processor = Arc::new(GgufProcessor::new(tokenizer, ProcessorLimits::default()));
    let mut owner = TextOwner::new(processor, shutdown, handle, TextConfig::default()).unwrap();
    let text_model = owner.model();

    let opts = |max_output_tokens: u32| GenerationOptions {
        max_output_tokens,
        sampling: Sampling::default(),
        stop_tokens: Vec::new(),
    };

    // Start and immediately drop a stream to verify cancellation releases its slot cleanly!
    let stream_cancel = text_model
        .stream_blocking(TextInput::Tokens(prompt_a.clone()), opts(4))
        .unwrap();
    drop(stream_cancel);

    // Two concurrent sequences filling max_sequences=2 across the sliding-window boundary
    let stream_a = text_model
        .stream_blocking(TextInput::Tokens(prompt_a), opts(4))
        .unwrap();
    let stream_b = text_model
        .stream_blocking(TextInput::Tokens(prompt_b), opts(4))
        .unwrap();

    let collect_tokens = |mut stream: ribn_text::TextStream| -> Vec<u32> {
        let mut got = Vec::new();
        while let Some(ev) = stream.next_blocking() {
            match ev.expect("stream event") {
                TextEvent::Delta {
                    token: Some(tok), ..
                } => got.push(tok),
                TextEvent::Delta { token: None, .. } | TextEvent::Finished { .. } => {}
            }
        }
        got
    };
    assert_eq!(collect_tokens(stream_a), expected_a);
    assert_eq!(collect_tokens(stream_b), expected_b);

    // Non-greedy sampling is rejected cleanly at admission without faulting the owner!
    let bad_opts = GenerationOptions {
        max_output_tokens: 2,
        sampling: Sampling {
            temperature: 0.7,
            ..Sampling::default()
        },
        stop_tokens: Vec::new(),
    };
    let rejected = text_model
        .generate_blocking(TextInput::Prompt("Hi".into()), bad_opts)
        .expect("request-local admission rejection returns a terminal response");
    assert!(matches!(rejected.reason, FinishReason::Failed(_)));
    assert_eq!(rejected.usage.completion_tokens, 0);

    // Chat path through real GgufProcessor using Inkling 2's {%- generation -%} chat template!
    let chat_resp = text_model
        .generate_blocking(
            TextInput::Chat(vec![
                Message::new("system", "Brief"),
                Message::new("user", "Hi"),
                Message::new("assistant", "Yo"),
                Message::new("user", "Go"),
            ]),
            opts(2),
        )
        .unwrap();
    assert!(chat_resp.usage.prompt_tokens > 8);
    assert!(chat_resp.usage.completion_tokens >= 1);

    // Clean shutdown
    owner.shutdown().unwrap();

    // Keep control marker IDs explicit in fixture.
    assert_eq!(
        [CTRL_USER, CTRL_MODEL, CTRL_SYSTEM, CTRL_TEXT, CTRL_END_MSG],
        [256, 257, 258, 259, 260]
    );
}
