//! Gemma text facade over the token runtime, gated on macOS and the metal feature.
#![cfg(all(target_os = "macos", feature = "metal"))]

use engine_gemma4::{
    Gemma4Config, Gemma4CpuWeights, Gemma4Executor, Gemma4MetalModel, Gemma4Source,
};
use ribn::driver::{Driver, DriverConfig};
use ribn::{Engine, EngineConfig, GenerationOptions, SchedulePolicy};
use ribn_hf::LocalModelPackage;
use ribn_text::{GemmaProcessor, ProcessorLimits, TextConfig, TextInput, TextOwner};
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
fn gemma_text_facade_reproduces_the_archived_greedy_tokens() {
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

    let host = Gemma4CpuWeights::load(&source).unwrap();
    assert_eq!(host.config(), &config);
    let model = Arc::new(Gemma4MetalModel::load(host).unwrap());
    let executor = Gemma4Executor::metal(model).expect("Metal executor");
    let engine = Engine::new(
        executor,
        EngineConfig {
            max_active_requests: 2,
            max_queued_requests: 2,
            max_queued_input_tokens: 4096,
            max_buffered_events: 32,
            max_events_per_request: 8,
        },
        SchedulePolicy::default(),
    )
    .expect("engine");
    let driver = DriverConfig::for_engine(&engine);
    let (shutdown, handle) = Driver::spawn(engine, driver).expect("driver");
    let processor = Arc::new(GemmaProcessor::open(&package, ProcessorLimits::default()).unwrap());
    let owner = TextOwner::new(
        processor.clone(),
        shutdown,
        handle,
        TextConfig {
            preprocessing_workers: 1,
            batch_window: 1,
        },
    )
    .expect("text owner");
    let response = owner
        .model()
        .clone()
        .generate_blocking(
            TextInput::Tokens(prompt),
            GenerationOptions {
                max_output_tokens: 4,
                ..GenerationOptions::default()
            },
        )
        .expect("text generation");
    assert_eq!(response.tokens, expected, "text facade greedy tokens");
}
