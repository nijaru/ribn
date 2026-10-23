//! Gemma through the token runtime's engine, gated on macOS and the metal feature.
#![cfg(all(target_os = "macos", feature = "metal"))]

use engine_gemma4::{Gemma4Config, Gemma4Executor, Gemma4MetalModel, Gemma4Source};
use ribn::{Engine, Event, GenerationExecutor, GenerationOptions, TokenRequest};
use ribn_hf::LocalModelPackage;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;

fn environment(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name)
            .unwrap_or_else(|| panic!("{name} must name the archived evidence file")),
    )
}

#[test]
#[ignore = "loads the 14.9 GB package and runs the Metal decoder; use --release"]
fn metal_executor_reproduces_the_archived_tokens_through_the_engine() {
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
    let expected: Vec<u32> = case["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|step| u32::try_from(step["chosen"].as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(expected.len(), 4);

    assert_eq!(source.config(), &config);
    let model =
        Arc::new(Gemma4MetalModel::load(&source, engine_gemma4::Gemma4MetalFfn::Native).unwrap());
    let executor = Gemma4Executor::metal(model).expect("Metal executor");
    let limits = executor.info().limits;
    assert_eq!(limits.max_decode_tokens, 1);
    let mut engine = Engine::with_defaults(executor).expect("engine");
    let request = engine
        .enqueue(TokenRequest::new(
            prompt.clone(),
            GenerationOptions {
                max_output_tokens: 4,
                ..GenerationOptions::default()
            },
        ))
        .expect("enqueue");

    let mut tokens = Vec::new();
    let mut finished = None;
    for _ in 0..64 {
        engine.step().expect("engine step");
        while let Some(event) = engine.pop_event_for(request) {
            match event {
                Event::Token { token, .. } => tokens.push(token),
                Event::Finished { reason, .. } => finished = Some(reason),
            }
        }
        if finished.is_some() {
            break;
        }
    }
    assert!(finished.is_some(), "request never finished");
    assert_eq!(tokens, expected, "engine greedy tokens");
    assert_eq!(engine.status().requests, 0);
    engine.shutdown().expect("shutdown");
}
