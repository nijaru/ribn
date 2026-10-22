//! Gemma through the token runtime's engine on the native CUDA decoder.
#![cfg(feature = "cuda")]

use engine_gemma4::{
    Gemma4Config, Gemma4CpuWeights, Gemma4CudaModel, Gemma4Executor, Gemma4Source,
};
use ribn::{Engine, EngineError, Event, GenerationExecutor, GenerationOptions, TokenRequest};
use ribn_foundation::BytePool;
use ribn_hf::LocalModelPackage;
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
fn cuda_executor_reproduces_the_archived_tokens_through_the_engine() {
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
    assert_eq!(executor.info().limits.max_decode_tokens, 1);
    assert_eq!(executor.info().limits.context_tokens, 32);
    let oversized = vec![prompt[0]; 32];
    let mut engine = Engine::with_defaults(executor).expect("engine");
    let request = engine
        .enqueue(TokenRequest::new(
            prompt,
            GenerationOptions {
                max_output_tokens: 4,
                ..GenerationOptions::default()
            },
        ))
        .expect("enqueue");
    assert!(matches!(
        engine.enqueue(TokenRequest::new(
            oversized,
            GenerationOptions {
                max_output_tokens: 1,
                ..GenerationOptions::default()
            },
        )),
        Err(EngineError::InvalidRequest)
    ));

    let mut tokens = Vec::new();
    let mut finished = false;
    for _ in 0..16 {
        engine.step().expect("engine step");
        while let Some(event) = engine.pop_event_for(request) {
            match event {
                Event::Token { token, .. } => tokens.push(token),
                Event::Finished { .. } => finished = true,
            }
        }
        if finished {
            break;
        }
    }
    assert!(finished, "request never finished");
    assert_eq!(tokens, expected, "engine greedy tokens");
    assert_eq!(engine.status().requests, 0);
    engine.shutdown().expect("shutdown");
}
