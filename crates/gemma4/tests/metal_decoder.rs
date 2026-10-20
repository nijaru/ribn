//! End-to-end Metal Gemma forwarding, gated on macOS and the metal feature.
#![cfg(all(target_os = "macos", feature = "metal"))]

use engine_gemma4::{Gemma4Config, Gemma4MetalDecoder, Gemma4Source, metal_argmax};
use ribn_hf::LocalModelPackage;
use serde_json::Value;
use std::path::PathBuf;

fn environment(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name)
            .unwrap_or_else(|| panic!("{name} must name the archived evidence file")),
    )
}

fn finite_logits(logits: &[u16]) {
    assert_eq!(logits.len(), 262_144);
    assert!(
        logits
            .iter()
            .all(|&bits| f32::from_bits(u32::from(bits) << 16).is_finite())
    );
}

fn same_logits(actual: &[u16], expected: &[u16], chunk: usize, step: usize) {
    finite_logits(actual);
    assert_eq!(actual.len(), expected.len());
    let mismatch = actual
        .iter()
        .zip(expected)
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected);
    assert!(
        mismatch.is_none(),
        "chunk {chunk}, step {step}: first logit mismatch {mismatch:?}"
    );
}

#[test]
#[ignore = "loads the 14.9 GB package and runs the Metal decoder; use --release"]
fn metal_decoder_reproduces_the_archived_greedy_tokens() {
    let package = LocalModelPackage::open(environment("RIBN_GEMMA_PACKAGE")).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
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
    let steps = case["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 4);

    assert_eq!(source.config(), &config);
    let mut decoder = Gemma4MetalDecoder::load(&source).expect("Metal decoder");
    let mut logits = decoder.prefill(&prompt).expect("prompt prefill");
    finite_logits(&logits);
    let mut full = vec![logits.clone()];
    let mut chosen = metal_argmax(&logits).unwrap();
    println!("step 0: {chosen} expected {}", steps[0]["chosen"]);
    assert_eq!(
        u64::from(chosen),
        steps[0]["chosen"].as_u64().unwrap(),
        "Metal greedy token at step 0"
    );
    for (index, step) in steps.iter().enumerate().skip(1) {
        logits = decoder.decode(chosen).expect("decode step");
        finite_logits(&logits);
        full.push(logits.clone());
        chosen = metal_argmax(&logits).unwrap();
        println!("step {index}: {chosen} expected {}", step["chosen"]);
        assert_eq!(
            u64::from(chosen),
            step["chosen"].as_u64().unwrap(),
            "Metal greedy token at step {index}"
        );
    }
    assert_eq!(decoder.position(), prompt.len() + steps.len() - 1);
    // Whole-prompt execution selects paired rows; these chunk sizes exercise
    // the original kernel. Compare full rows on identical continuation histories,
    // not just the greedy winners, and keep cache state live through each check.
    for chunk in [1, 7] {
        decoder.reset();
        assert_eq!(decoder.position(), 0);
        for tokens in prompt.chunks(chunk) {
            logits = decoder.prefill(tokens).expect("chunked prefill");
        }
        same_logits(&logits, &full[0], chunk, 0);
        for (index, expected) in full.iter().enumerate().skip(1) {
            let token = u32::try_from(steps[index - 1]["chosen"].as_u64().unwrap()).unwrap();
            logits = decoder.decode(token).expect("common-history decode");
            same_logits(&logits, expected, chunk, index);
        }
        assert_eq!(decoder.position(), prompt.len() + steps.len() - 1);
    }
    decoder.reset();
    assert_eq!(decoder.position(), 0);
}
