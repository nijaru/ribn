//! Real cache-grant pressure through the Gemma CUDA executor and token runtime.
#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use engine_gemma4::{Gemma4CudaModel, Gemma4Executor, Gemma4Source};
use ribn::{
    Engine, EngineConfig, Event, FinishReason, GenerationOptions, RequestId, SchedulePolicy,
    TokenRequest, Usage,
};
use ribn_foundation::{BytePool, ReserveError};
use ribn_hf::LocalModelPackage;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn environment(name: &str) -> PathBuf {
    PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("{name} must be set")))
}

fn request(prompt: &[u32], outputs: u32) -> TokenRequest {
    TokenRequest::new(
        prompt.to_vec(),
        GenerationOptions {
            max_output_tokens: outputs,
            ..GenerationOptions::default()
        },
    )
}

fn collect(engine: &mut Engine, request: RequestId) -> (Vec<u32>, FinishReason, Usage) {
    let mut tokens = Vec::new();
    for _ in 0..32 {
        engine.step().expect("healthy owner must keep progressing");
        while let Some(event) = engine.pop_event_for(request) {
            match event {
                Event::Token { token, .. } => tokens.push(token),
                Event::Finished { reason, usage, .. } => {
                    assert_eq!(engine.pop_event_for(request), None);
                    return (tokens, reason, usage);
                }
            }
        }
    }
    panic!("request did not finish within the bounded step count");
}

fn assert_generation(engine: &mut Engine, request: RequestId, expected: &[u32], usage: Usage) {
    let (tokens, reason, actual_usage) = collect(engine, request);
    assert_eq!(tokens, expected);
    assert_eq!(reason, FinishReason::Length);
    assert_eq!(actual_usage, usage);
}

fn assert_pressure_rejection(
    engine: &mut Engine,
    request: RequestId,
    pool: &BytePool,
    key_bytes: u64,
    prompt_tokens: u32,
) -> FinishReason {
    let released = pool.capacity_wait();
    let (tokens, reason, usage) = collect(engine, request);
    assert!(tokens.is_empty());
    let FinishReason::Failed(ref error) = reason else {
        panic!("expected request-local pressure rejection, got {reason:?}");
    };
    let expected = ReserveError::Exhausted {
        requested: key_bytes,
        available: 0,
    };
    assert!(error.to_string().contains(&expected.to_string()), "{error}");
    assert!(
        released.changed(),
        "partial cache retirement must publish readiness"
    );
    assert_eq!(
        usage,
        Usage {
            prompt_tokens,
            completion_tokens: 0
        }
    );
    reason
}

struct Fixture {
    engine: Engine,
    model: Arc<Mutex<Gemma4CudaModel>>,
    pool: Arc<BytePool>,
    prompt: Vec<u32>,
    expected: Vec<u32>,
    cache_bytes: u64,
    first_key_bytes: u64,
}

fn fixture() -> Fixture {
    let package = LocalModelPackage::open(environment("RIBN_GEMMA_PACKAGE")).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(environment("RIBN_GEMMA_REFERENCE")).unwrap())
            .unwrap();
    let case = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "chat")
        .expect("archived chat case");
    let prompt: Vec<u32> = serde_json::from_value(case["prompt_ids"].clone()).unwrap();
    let expected: Vec<u32> = case["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|step| u32::try_from(step["chosen"].as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(expected.len(), 4);
    let config = source.config();
    let capacity = 4096;
    // Only KV-owning layers allocate storage; donor layers share those caches.
    let key_bytes =
        |head_dim| u64::try_from(capacity * config.num_key_value_heads() * head_dim * 2).unwrap();
    let cache_bytes: u64 = config
        .layers()
        .iter()
        .enumerate()
        .filter(|(index, layer)| layer.kv_owner == *index)
        .map(|(_, layer)| 2 * key_bytes(layer.head_dim))
        .sum();
    let first_key_bytes = key_bytes(config.layers()[0].head_dim);
    let pool = BytePool::new(20_u64 << 30).shared();
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let model = Gemma4CudaModel::load(&source, stream, &pool, capacity, 1 << 20).unwrap();
    assert_eq!(model.prefill_rows(), 16);
    let model = Arc::new(Mutex::new(model));
    let engine = Engine::new(
        Gemma4Executor::cuda(&model).unwrap(),
        EngineConfig {
            max_active_requests: 4,
            max_queued_requests: 4,
            max_queued_input_tokens: 4096,
            max_buffered_events: 16,
            max_events_per_request: 2,
        },
        SchedulePolicy::default(),
    )
    .unwrap();
    Fixture {
        engine,
        model,
        pool,
        prompt,
        expected,
        cache_bytes,
        first_key_bytes,
    }
}

#[test]
#[ignore = "loads the 16 GB package and tests real CUDA allocation pressure; serialize device runs"]
fn partial_cache_failure_preserves_peers_and_cancelled_capacity_is_reusable() {
    let Fixture {
        mut engine,
        model,
        pool,
        prompt,
        expected,
        cache_bytes,
        first_key_bytes,
    } = fixture();
    let outputs = u32::try_from(expected.len()).unwrap();
    let usage = Usage {
        prompt_tokens: u32::try_from(prompt.len()).unwrap(),
        completion_tokens: outputs,
    };
    let model_bytes = pool.granted();
    // A competing logical grant, not an allocation of dummy physical VRAM.
    // Two caches fit; the third can allocate its first K but not its first V.
    let pressure = pool
        .reserve(
            pool.available()
                .checked_sub(2 * cache_bytes + first_key_bytes)
                .unwrap(),
        )
        .unwrap();
    let baseline = pool.granted();
    let victim = engine.enqueue(request(&prompt, outputs)).unwrap();
    let peer = engine.enqueue(request(&prompt, outputs)).unwrap();
    for _ in 0..16 {
        engine.step().unwrap();
        if engine.status().buffered_events == 2 && !engine.status().in_flight {
            break;
        }
    }
    // One unread token leaves only terminal credit per mailbox, blocking decode.
    assert_eq!(engine.status().buffered_events, 2);
    assert!(!engine.status().in_flight);
    assert_eq!(engine.status().active_sequences, 2);
    assert_eq!(pool.granted(), baseline + 2 * cache_bytes);

    let rejected = engine.enqueue(request(&prompt, outputs)).unwrap();
    let reason = assert_pressure_rejection(
        &mut engine,
        rejected,
        &pool,
        first_key_bytes,
        usage.prompt_tokens,
    );
    assert!(!engine.status().faulted);
    assert_eq!(engine.status().active_sequences, 2);
    assert_eq!(
        pool.granted(),
        baseline + 2 * cache_bytes,
        "partial K retired"
    );
    eprintln!(
        "cache_bytes={cache_bytes} first_key_bytes={first_key_bytes} model_bytes={model_bytes} rejected={reason:?}"
    );

    assert_eq!(
        engine.pop_event_for(victim),
        Some(Event::Token {
            request: victim,
            token: expected[0]
        })
    );
    assert!(engine.step().unwrap().submitted);
    assert!(engine.status().in_flight);
    engine.cancel(victim).unwrap();
    assert_eq!(
        pool.granted(),
        baseline + 2 * cache_bytes,
        "intent is not retirement"
    );
    assert!(engine.status().in_flight);
    assert!(engine.step().unwrap().completed);
    assert_eq!(
        engine.pop_event_for(victim),
        Some(Event::Finished {
            request: victim,
            reason: FinishReason::Cancelled,
            usage: Usage {
                completion_tokens: 1,
                ..usage
            },
        })
    );
    assert_eq!(engine.pop_event_for(victim), None);
    assert_eq!(pool.granted(), baseline + cache_bytes);

    // The peer's unread token/cache cannot prevent another request from finishing.
    let recovered = engine.enqueue(request(&prompt, outputs)).unwrap();
    assert_generation(&mut engine, recovered, &expected, usage);
    assert_eq!(pool.granted(), baseline + cache_bytes);
    assert_generation(&mut engine, peer, &expected, usage);
    assert_eq!(pool.granted(), baseline);
    drop(pressure);
    assert_eq!(pool.granted(), model_bytes);
    let reused = engine.enqueue(request(&prompt, outputs)).unwrap();
    assert_generation(&mut engine, reused, &expected, usage);
    engine.shutdown().unwrap();
    drop(engine);
    drop(model);
    assert_eq!(pool.granted(), 0);
}
