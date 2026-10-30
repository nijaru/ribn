//! End-to-end native CUDA Gemma forwarding against archived common histories.
#![cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use engine_gemma4::{
    Gemma4Config, Gemma4CudaDecoder, Gemma4CudaModel, Gemma4Source, GemmaDecode, argmax,
};
use engine_nvidia::CudaBf16Matrix;
use ribn_foundation::{BytePool, ReserveError};
use ribn_hf::LocalModelPackage;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn environment(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name)
            .unwrap_or_else(|| panic!("{name} must name the archived evidence file")),
    )
}

#[test]
#[ignore = "loads the 15 GB package and runs the CUDA decoder; serialize device runs"]
fn cuda_decoder_reproduces_the_archived_greedy_tokens() {
    let package = LocalModelPackage::open(environment("RIBN_GEMMA_PACKAGE")).unwrap();
    let config = Gemma4Config::from_json(
        &serde_json::from_slice(&std::fs::read(package.root().join("config.json")).unwrap())
            .unwrap(),
    )
    .unwrap();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(environment("RIBN_GEMMA_REFERENCE")).unwrap())
            .unwrap();
    assert_eq!(reference["schema_version"], 1);
    assert_eq!(reference["complete"], true);
    assert_eq!(reference["precision"], "bf16");
    let case = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "chat")
        .expect("archived chat case");
    let prompt: Vec<u32> = case["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
        .collect();
    let steps: Vec<u32> = case["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|step| u32::try_from(step["chosen"].as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(prompt.len(), 18);
    assert_eq!(steps.len(), 4);

    let source = Gemma4Source::open(&package).unwrap();
    assert_eq!(source.config(), &config);
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(20_u64 << 30).shared();
    let started = Instant::now();
    let model = Gemma4CudaModel::load(&source, stream, &pool, 32, 512).expect("CUDA model");
    assert_eq!(model.prefill_rows(), 1, "scalar bitwise control");
    let model = Arc::new(Mutex::new(model));
    let mut decoder = Gemma4CudaDecoder::shared(&model).unwrap();
    let loaded = started.elapsed();

    let prefill = Instant::now();
    let mut logits = decoder.prefill(&prompt).expect("prompt prefill");
    let prefilled = prefill.elapsed();
    let mut expected_logits = vec![logits.clone()];
    // Whole-chunk rejection must not consume a valid prefix or alter live KV.
    for invalid in [
        Vec::new(),
        vec![prompt[0], u32::try_from(config.vocab_size()).unwrap()],
        vec![prompt[0]; 32],
    ] {
        assert!(decoder.prefill(&invalid).is_err());
        assert_eq!(decoder.position(), prompt.len());
        assert!(decoder.advance_prefill(&invalid).is_err());
        assert_eq!(decoder.position(), prompt.len());
    }
    let mut step_times = Vec::with_capacity(steps.len() - 1);
    let mut chosen = argmax(&logits).unwrap();
    assert_eq!(u64::from(chosen), u64::from(steps[0]), "step 0");
    for (index, expected) in steps.iter().enumerate().skip(1) {
        let step = Instant::now();
        logits = decoder.decode(chosen).expect("decode step");
        step_times.push(step.elapsed());
        expected_logits.push(logits.clone());
        chosen = argmax(&logits).unwrap();
        assert_eq!(u64::from(chosen), u64::from(*expected), "step {index}");
    }
    assert_eq!(decoder.position(), prompt.len() + steps.len() - 1);
    assert_eq!(logits.len(), config.vocab_size());
    println!("load {loaded:?}, prefill {prefilled:?}, decode {step_times:?}");
    // Single-token heads and partial prompt chunks must produce exactly the same
    // logits and continuation as skipping heads within one complete prefill.
    for chunk_size in [1, 7] {
        for state_only in [false, true] {
            decoder.reset();
            assert_eq!(decoder.position(), 0);
            logits = prefill_chunked(&mut decoder, &prompt, chunk_size, state_only);
            assert_logits(&logits, &expected_logits[0]);
            for (token, expected) in steps.iter().zip(&expected_logits[1..]) {
                logits = decoder.decode(*token).unwrap();
                assert_logits(&logits, expected);
            }
            assert_eq!(decoder.position(), prompt.len() + steps.len() - 1);
        }
    }
    decoder.reset();
    assert_eq!(decoder.position(), 0);
}

#[test]
#[ignore = "loads the 15 GB package and crosses the sliding window; serialize device runs"]
fn cuda_decoder_reproduces_archived_window_boundary_continuations() {
    let package = LocalModelPackage::open(environment("RIBN_GEMMA_PACKAGE")).unwrap();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(environment("RIBN_GEMMA_REFERENCE")).unwrap())
            .unwrap();
    assert_eq!(reference["schema_version"], 1);
    assert_eq!(reference["complete"], true);
    assert_eq!(reference["precision"], "bf16");
    let source = Gemma4Source::open(&package).unwrap();
    let config = source.config().clone();
    let capacity = config.sliding_window().checked_add(4).unwrap();
    let scratch = capacity.checked_mul(config.num_attention_heads()).unwrap() * 8;
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let pool = BytePool::new(20_u64 << 30).shared();
    let model = Gemma4CudaModel::load(&source, stream, &pool, capacity, scratch).unwrap();
    assert_eq!(
        model.prefill_rows(),
        8,
        "scratch ceiling preserves the qualified eight-row fallback"
    );
    let model = Arc::new(Mutex::new(model));
    let mut decoder = Gemma4CudaDecoder::shared(&model).unwrap();
    drop(source);
    drop(package);

    // Ordinary tails, a no-tail 512-token head, and a group crossing position
    // 512 internally (one scalar prefix token followed by 512 submitted tokens).
    for (name, first_step, prefix) in [
        ("chat", 0, 0),
        ("window_511", 0, 0),
        ("window_513", 0, 0),
        ("window_511", 1, 0),
        ("window_513", 0, 1),
    ] {
        let case = reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .expect("archived window case");
        let steps = case["steps"].as_array().unwrap();
        assert_eq!(steps.len(), 4);
        decoder.reset();
        let valid = u32::try_from(case["prompt_ids"][0].as_u64().unwrap()).unwrap();
        let mut invalid_tail = vec![valid; 16];
        invalid_tail.push(u32::try_from(config.vocab_size()).unwrap());
        for invalid in [invalid_tail, vec![valid; capacity + 1]] {
            assert!(decoder.prefill(&invalid).is_err());
            assert_eq!(decoder.position(), 0);
        }
        let mut expected_logits = Vec::new();
        for (index, step) in steps.iter().enumerate().skip(first_step) {
            let input: Vec<u32> = step[if index == first_step {
                "history_ids"
            } else {
                "input_ids"
            }]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
            .collect();
            if index != first_step {
                assert_eq!(input.len(), 1);
            }
            let logits = if index == first_step && prefix != 0 {
                decoder.prefill(&input[..prefix]).unwrap();
                decoder.prefill(&input[prefix..]).unwrap()
            } else {
                decoder.prefill(&input).unwrap()
            };
            assert_eq!(logits.len(), config.vocab_size());
            assert!(logits.iter().all(|bits| bits & 0x7f80 != 0x7f80));
            expected_logits.push(logits.clone());
            assert_eq!(
                u64::from(argmax(&logits).unwrap()),
                step["chosen"].as_u64().unwrap(),
                "{name} step {index}"
            );
            assert_eq!(
                decoder.position(),
                step["history_ids"].as_array().unwrap().len()
            );
        }
        check_state_only_history(&mut decoder, case, first_step, prefix, &expected_logits);
    }
}

#[test]
#[ignore = "loads the 15 GB package and allocates CUDA caches; serialize device runs"]
fn cuda_partial_cache_construction_retires_allocations_before_refund() {
    let package = LocalModelPackage::open(environment("RIBN_GEMMA_PACKAGE")).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    let config = source.config().clone();
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let pool = BytePool::new(20_u64 << 30).shared();
    let model = Arc::new(Mutex::new(
        Gemma4CudaModel::load(&source, Arc::clone(&stream), &pool, 32, 4096).unwrap(),
    ));
    assert_eq!(model.lock().unwrap().prefill_rows(), 16);
    let baseline = pool.available();
    let mut charges = Vec::new();
    for (layer, geometry) in config.layers().iter().enumerate() {
        if geometry.kv_owner != layer {
            continue;
        }
        let mut matrix = CudaBf16Matrix::allocate(
            32,
            config.num_key_value_heads() * geometry.head_dim,
            Arc::clone(&stream),
            &pool,
            1,
        )
        .unwrap();
        charges.push(matrix.allocated_bytes());
        assert_eq!(baseline - pool.available(), matrix.allocated_bytes());
        matrix.close().unwrap();
        assert_eq!(pool.available(), baseline);
        if charges.len() == 2 {
            break;
        }
    }
    assert_eq!(charges.len(), 2);
    // Budget refusal, NOT driver OOM: leave exactly one key, or a completed
    // first layer plus the next key. The subsequent value cannot reserve.
    for (available, requested) in [
        (charges[0], charges[0]),
        (2 * charges[0] + charges[1], charges[1]),
    ] {
        let pressure = pool.reserve(baseline - available).unwrap();
        let released = pool.capacity_wait();
        let error = Gemma4CudaDecoder::shared(&model)
            .err()
            .expect("value must not fit");
        let expected = ReserveError::Exhausted {
            requested,
            available: 0,
        };
        assert!(error.to_string().contains(&expected.to_string()), "{error}");
        assert!(
            released.changed(),
            "partial allocation must publish retirement"
        );
        assert_eq!(pool.available(), available);
        model.lock().unwrap().retire_failure().unwrap();
        assert_eq!(pool.available(), available);
        drop(pressure);
        assert_eq!(pool.available(), baseline);
    }
    let mut decoder = Gemma4CudaDecoder::shared(&model).unwrap();
    assert!(pool.available() < baseline);
    let logits = decoder.prefill(&[2]).unwrap();
    assert_eq!(logits.len(), config.vocab_size());
    assert!(logits.iter().all(|bits| bits & 0x7f80 != 0x7f80));
    assert_eq!(decoder.position(), 1);
    model.lock().unwrap().retire_failure().unwrap();
    decoder.close().unwrap();
    assert_eq!(pool.available(), baseline);
    decoder.close().unwrap();
    assert_eq!(pool.available(), baseline);
    drop(decoder);
    drop(model);
    assert_eq!(pool.granted(), 0);
}

fn check_state_only_history(
    decoder: &mut Gemma4CudaDecoder,
    case: &Value,
    first_step: usize,
    prefix: usize,
    expected_logits: &[Vec<u16>],
) {
    decoder.reset();
    for (relative, expected) in expected_logits.iter().enumerate() {
        let step = &case["steps"][first_step + relative];
        let input: Vec<u32> = step[if relative == 0 {
            "history_ids"
        } else {
            "input_ids"
        }]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| u32::try_from(id.as_u64().unwrap()).unwrap())
        .collect();
        let start = if relative == 0 { prefix } else { 0 };
        if start != 0 {
            decoder.advance_prefill(&input[..start]).unwrap();
        }
        let logits = prefill_chunked(decoder, &input[start..], 16, true);
        assert_logits(&logits, expected);
        assert_eq!(
            decoder.position(),
            step["history_ids"].as_array().unwrap().len()
        );
    }
}

fn prefill_chunked(
    decoder: &mut Gemma4CudaDecoder,
    tokens: &[u32],
    chunk_size: usize,
    state_only: bool,
) -> Vec<u16> {
    let final_start = (tokens.len() - 1) / chunk_size * chunk_size;
    let mut position = decoder.position();
    for chunk in tokens[..final_start].chunks(chunk_size) {
        if state_only {
            decoder.advance_prefill(chunk).unwrap();
        } else {
            decoder.prefill(chunk).unwrap();
        }
        position += chunk.len();
        assert_eq!(decoder.position(), position);
    }
    decoder.prefill(&tokens[final_start..]).unwrap()
}

fn assert_logits(actual: &[u16], expected: &[u16]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|bits| bits & 0x7f80 != 0x7f80));
    assert!(
        actual == expected,
        "prefill partition changed logit encodings"
    );
}
