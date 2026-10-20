//! End-to-end native CUDA Gemma forwarding against the archived greedy case.
#![cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use engine_gemma4::{Gemma4Config, Gemma4CpuWeights, Gemma4CudaDecoder, Gemma4Source, argmax};
use ribn_foundation::BytePool;
use ribn_hf::LocalModelPackage;
use serde_json::Value;
use std::path::PathBuf;
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
    let host = Gemma4CpuWeights::load(&source).unwrap();
    assert_eq!(host.config(), &config);
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(20_u64 << 30).shared();
    let started = Instant::now();
    let mut decoder =
        Gemma4CudaDecoder::load(&source, host, stream, &pool, 32, 512).expect("CUDA decoder");
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
        decoder.reset();
        assert_eq!(decoder.position(), 0);
        for chunk in prompt.chunks(chunk_size) {
            logits = decoder.prefill(chunk).unwrap();
        }
        assert_logits(&logits, &expected_logits[0]);
        for (token, expected) in steps.iter().zip(&expected_logits[1..]) {
            logits = decoder.decode(*token).unwrap();
            assert_logits(&logits, expected);
        }
        assert_eq!(decoder.position(), prompt.len() + steps.len() - 1);
    }
    decoder.reset();
    assert_eq!(decoder.position(), 0);
}

fn assert_logits(actual: &[u16], expected: &[u16]) {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|bits| bits & 0x7f80 != 0x7f80));
    assert!(
        actual == expected,
        "prefill partition changed logit encodings"
    );
}
