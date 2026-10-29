//! Gemma through the token runtime's engine on the native CUDA decoder.
#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, sys};
use engine_gemma4::{Gemma4Config, Gemma4CudaModel, Gemma4Executor, Gemma4Source};
use ribn::{
    Engine, EngineError, Event, FinishReason, GenerationExecutor, GenerationOptions, TokenRequest,
};
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

    assert_eq!(source.config(), &config);
    let context = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(20_u64 << 30).shared();
    let scratch = 32 * config.num_attention_heads() * 16;
    let model = Gemma4CudaModel::load(&source, stream, &pool, 32, scratch).unwrap();
    assert_eq!(model.prefill_rows(), 16, "qualified RTX 4090/E4B prefill");
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
    let mut finished = None;
    for _ in 0..16 {
        engine.step().expect("engine step");
        while let Some(event) = engine.pop_event_for(request) {
            match event {
                Event::Token { token, .. } => tokens.push(token),
                Event::Finished { reason, usage, .. } => finished = Some((reason, usage)),
            }
        }
        if finished.is_some() {
            break;
        }
    }
    let (reason, usage) = finished.expect("request never finished");
    assert_eq!(reason, FinishReason::Length);
    assert_eq!(usage.prompt_tokens, 18);
    assert_eq!(usage.completion_tokens, 4);
    assert_eq!(tokens, expected, "engine greedy tokens");
    assert_eq!(engine.status().requests, 0);
    engine.shutdown().expect("shutdown");
}

#[test]
#[ignore = "loads the 15 GB package and exercises native stream errors; serialize device runs"]
fn cuda_failed_drain_keeps_owners_until_capture_ends() {
    let package = LocalModelPackage::open(environment("RIBN_GEMMA_PACKAGE")).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let pool = BytePool::new(20_u64 << 30).shared();
    let model = Arc::new(Mutex::new(
        Gemma4CudaModel::load(&source, Arc::clone(&stream), &pool, 32, 4096).unwrap(),
    ));
    assert_eq!(model.lock().unwrap().prefill_rows(), 16);
    let model_charge = pool.granted();
    let mut engine = Engine::with_defaults(Gemma4Executor::cuda(&model).unwrap()).unwrap();
    engine
        .enqueue(TokenRequest::new(
            vec![2; 16],
            GenerationOptions {
                max_output_tokens: 4,
                ..GenerationOptions::default()
            },
        ))
        .unwrap();
    engine.step().unwrap(); // Admission/cache allocation and submission, not execution.
    let admitted_charge = pool.granted();
    assert!(admitted_charge > model_charge);
    // Captured work is not executed. A synchronous decoder's wait is forbidden;
    // this induces a real driver error without illegal accesses or a GPU reset.
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    let execution = engine.step();
    let shutdown = engine.shutdown();
    let retained_charge = pool.granted();
    let status = stream.capture_status();
    let ended = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    let no_graph = matches!(&ended, Ok(None) | Err(_));
    drop(ended); // Destroy any unexpected graph before attempting allocation release.
    let retry = engine.shutdown();
    // End capture and attempt cleanup before assertions, including on unexpected results.
    assert!(execution.is_err());
    assert!(shutdown.is_err());
    assert_eq!(retained_charge, admitted_charge);
    assert_eq!(
        status.unwrap(),
        sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED
    );
    assert!(
        no_graph,
        "invalidated capture must not produce an executable graph"
    );
    retry.unwrap();
    assert_eq!(pool.granted(), model_charge);
    assert!(
        engine.step().is_err(),
        "retirement must not resume execution"
    );
    drop(engine);
    drop(model);
    assert_eq!(pool.granted(), 0);
}
