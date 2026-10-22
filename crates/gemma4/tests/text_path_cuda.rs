//! Gemma text facade over the token runtime on the native CUDA decoder.
#![cfg(feature = "cuda")]

use engine_gemma4::{
    Gemma4Config, Gemma4CpuWeights, Gemma4CudaModel, Gemma4Executor, Gemma4Source,
};
use ribn::driver::{Driver, DriverConfig};
use ribn::{Engine, EngineConfig, GenerationOptions, SchedulePolicy};
use ribn_foundation::BytePool;
use ribn_hf::LocalModelPackage;
use ribn_text::{GemmaProcessor, ProcessorLimits, TextConfig, TextInput, TextOwner};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn environment(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name)
            .unwrap_or_else(|| panic!("{name} must name the archived evidence file")),
    )
}

#[test]
#[ignore = "loads the 16 GB package and runs the CUDA decoder; serialize device runs"]
fn cuda_text_facade_reproduces_the_archived_greedy_tokens() {
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
    let context = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(20_u64 << 30).shared();
    let scratch = 32 * config.num_attention_heads() * 8;
    let model = Gemma4CudaModel::load(&source, host, stream, &pool, 32, scratch).unwrap();
    assert_eq!(model.prefill_rows(), 8, "qualified RTX 4090/E4B prefill");
    let model = Arc::new(Mutex::new(model));
    let executor = Gemma4Executor::cuda(&model).expect("CUDA executor");
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
        processor,
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
