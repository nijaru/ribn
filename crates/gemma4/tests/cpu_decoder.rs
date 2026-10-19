//! End-to-end host CPU Gemma forwarding, not a performance claim.

use engine_gemma4::{Gemma4Config, Gemma4CpuDecoder, Gemma4CpuWeights, Gemma4Source, argmax};
use ribn_hf::LocalModelPackage;
use serde_json::Value;
use std::path::PathBuf;

fn environment(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name)
            .unwrap_or_else(|| panic!("{name} must name the archived evidence file")),
    )
}

#[test]
fn argmax_rejects_an_empty_logit_row() {
    assert_eq!(argmax(&[]), None);
}

#[test]
#[ignore = "loads the 14.9 GB package and runs the host CPU decoder; use --release"]
fn greedy_chat_steps_match_the_archived_cpu_reference() {
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
    assert_eq!(reference["checkpoint"]["id"], "google/gemma-4-E4B-it");
    assert_eq!(
        reference["checkpoint"]["revision"],
        "ee0ef6023621cff504d758262d4e04895a5af4a2"
    );
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

    let weights = Gemma4CpuWeights::load(&source).unwrap();
    assert_eq!(weights.config(), &config);
    let mut decoder = Gemma4CpuDecoder::new(weights).unwrap();

    // A chunked prefill must reach the same first token, and a rejected request
    // must leave continuation state intact.
    let split = prompt.len() / 2;
    decoder.prefill(&prompt[..split]).expect("first chunk");
    let chunked = decoder.prefill(&prompt[split..]).expect("second chunk");
    let rejected = decoder.position();
    assert!(decoder.decode(u32::MAX).is_err());
    assert_eq!(decoder.position(), rejected);
    decoder.reset();
    assert_eq!(decoder.position(), 0);

    let mut logits = decoder.prefill(&prompt).expect("prompt prefill");
    let mut chosen = argmax(&logits).expect("nonempty logits");
    assert_eq!(argmax(&chunked), Some(chosen), "chunked prefill token");
    println!("step 0: {chosen} expected {}", steps[0]["chosen"]);
    assert_eq!(
        u64::from(chosen),
        steps[0]["chosen"].as_u64().unwrap(),
        "greedy token at step 0"
    );
    for (index, step) in steps.iter().enumerate().skip(1) {
        logits = decoder.decode(chosen).expect("decode step");
        chosen = argmax(&logits).expect("nonempty logits");
        println!("step {index}: {chosen} expected {}", step["chosen"]);
        assert_eq!(
            u64::from(chosen),
            step["chosen"].as_u64().unwrap(),
            "greedy token at step {index}"
        );
    }
    assert_eq!(decoder.position(), prompt.len() + steps.len() - 1);
    assert_eq!(logits.len(), 262_144);
}
