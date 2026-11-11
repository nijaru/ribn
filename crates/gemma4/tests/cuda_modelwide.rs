//! Real model-wide execution over archived histories and the service context boundary.
#![cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use engine_gemma4::{
    Gemma4CudaDecoder, Gemma4CudaExecutor, Gemma4CudaModel, Gemma4CudaModelWide, Gemma4Source,
    GemmaDecode, argmax,
};
use ribn::{Admission, Engine, Event, FinishReason, GenerationOptions, TokenRequest};
use ribn_foundation::BytePool;
use ribn_hf::LocalModelPackage;
use serde_json::Value;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

struct Case {
    name: String,
    prompt: Vec<u32>,
    chosen: Vec<u32>,
    inputs: Vec<u32>,
}
fn environment(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name).unwrap_or_else(|| panic!("{name} must name pinned local evidence")),
    )
}
fn cases() -> Vec<Case> {
    let document: Value =
        serde_json::from_slice(&std::fs::read(environment("RIBN_GEMMA_REFERENCE")).unwrap())
            .unwrap();
    assert_eq!(document["complete"], true);
    assert_eq!(document["precision"], "bf16");
    document["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| Case {
            name: case["name"].as_str().unwrap().into(),
            prompt: case["prompt_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                .collect(),
            inputs: case["steps"]
                .as_array()
                .unwrap()
                .iter()
                .skip(1)
                .map(|step| {
                    let input = step["input_ids"].as_array().unwrap();
                    assert_eq!(input.len(), 1, "archived teacher-forced continuation");
                    u32::try_from(input[0].as_u64().unwrap()).unwrap()
                })
                .collect(),
            chosen: case["steps"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u32::try_from(v["chosen"].as_u64().unwrap()).unwrap())
                .collect(),
        })
        .collect()
}
#[path = "support/modelwide_ids.rs"]
mod identities;
use identities::issued_ids;

fn exercise_runtime(model: Gemma4CudaModelWide, case: &Case) {
    assert_eq!(case.name, "chat");
    let mut engine = Engine::with_defaults(Gemma4CudaExecutor::new(model).unwrap()).unwrap();
    // Odd population exercises both a real two-member lane and the qualified
    // scalar remainder, with whole-offer original-order completion through Engine.
    let requests: [_; 3] = std::array::from_fn(|_| {
        engine
            .enqueue(TokenRequest::new(
                case.prompt.clone(),
                GenerationOptions {
                    max_output_tokens: 4,
                    ..GenerationOptions::default()
                },
            ))
            .unwrap()
    });
    let mut tokens: [Vec<u32>; 3] = std::array::from_fn(|_| Vec::new());
    let mut finished = [false; 3];
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while !finished.into_iter().all(|done| done) {
        engine.step().unwrap();
        for (index, request) in requests.iter().enumerate() {
            while let Some(event) = engine.pop_event_for(*request) {
                assert!(!finished[index], "no output after terminal");
                match event {
                    Event::Token { token, .. } => tokens[index].push(token),
                    Event::Finished { reason, usage, .. } => {
                        assert_eq!(reason, FinishReason::Length);
                        assert_eq!(usage.prompt_tokens as usize, case.prompt.len());
                        assert_eq!(usage.completion_tokens, 4);
                        assert_eq!(
                            tokens[index], case.chosen,
                            "runtime own-produced greedy history"
                        );
                        finished[index] = true;
                    }
                }
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "runtime completion deadline"
        );
        std::thread::sleep(std::time::Duration::from_micros(100));
    }
    assert_eq!(engine.status().requests, 0);
    engine.shutdown().unwrap();
    engine.shutdown().unwrap();
}

fn finite(logits: &[u16], width: usize) {
    assert_eq!(logits.len(), width);
    assert!(logits.iter().all(|&b| b & 0x7f80 != 0x7f80));
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one serialized control/candidate qualification with common-history routing and teardown"
)]
#[ignore = "loads the real 15 GB package twice sequentially; release-only serialized device gate"]
fn full_model_pairs_preserve_archived_winners_unequal_histories_and_reordered_members() {
    let _device = identities::DEVICE.lock().unwrap();
    let cases = cases();
    assert_eq!(cases.len(), 3);
    assert!(cases.iter().all(|case| case.chosen.len() == 4));
    let package = LocalModelPackage::open(environment("RIBN_GEMMA_PACKAGE")).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    let width = source.config().vocab_size();
    let capacity = cases
        .iter()
        .map(|c| c.prompt.len() + c.chosen.len())
        .max()
        .unwrap();
    let scratch = source.config().num_attention_heads() * capacity * 16;
    let context = CudaContext::new(0).unwrap();
    let pool = BytePool::new(20_u64 << 30).shared();
    // Do not keep two device weight inventories alive on the 4090. Preserve
    // complete scalar/qualified-prefill rows on host, then unload the control.
    let controls: Vec<Vec<Vec<u16>>> = {
        let stream = context.new_stream().unwrap();
        let model = Arc::new(Mutex::new(
            Gemma4CudaModel::load(&source, stream, &pool, capacity, scratch).unwrap(),
        ));
        cases
            .iter()
            .map(|case| {
                let mut decoder = Gemma4CudaDecoder::shared(&model).unwrap();
                let mut rows = vec![decoder.prefill(&case.prompt).unwrap()];
                for &token in &case.inputs {
                    rows.push(decoder.decode(token).unwrap());
                }
                for (row, &expected) in rows.iter().zip(&case.chosen) {
                    finite(row, width);
                    assert_eq!(argmax(row), Some(expected), "control {}", case.name);
                }
                decoder.close().unwrap();
                rows
            })
            .collect()
    };
    assert_eq!(pool.granted(), 0, "control unload before candidate load");
    let mut model = Gemma4CudaModelWide::load(&source, &context, &pool, capacity, scratch).unwrap();
    let model_grant = pool.granted();
    let ids = issued_ids::<2>();
    // Both unequal reaches and a window crossing inside the paired continuation;
    // case 1 participates twice so every archived case's four winners are covered.
    for pair in [[0, 1], [1, 2]] {
        for (member, &case) in pair.iter().enumerate() {
            assert!(matches!(
                model
                    .admit(
                        ids[member],
                        cases[case].prompt.len() + cases[case].chosen.len()
                    )
                    .unwrap(),
                Admission::Ready
            ));
            let logits = model
                .prefill(ids[member], &cases[case].prompt, true)
                .unwrap()
                .unwrap();
            assert_eq!(
                logits, controls[case][0],
                "qualified fallback {}",
                cases[case].name
            );
        }
        let positions = ids.map(|id| model.position(id).unwrap());
        assert!(
            model
                .start_pair([ids[0], ids[0]], [cases[pair[0]].chosen[0]; 2])
                .is_err()
        );
        assert_eq!(ids.map(|id| model.position(id).unwrap()), positions);
        for (step, _) in cases[pair[0]].chosen.iter().enumerate().skip(1) {
            let order = if step % 2 == 1 { [1, 0] } else { [0, 1] };
            let routed = order.map(|member| ids[member]);
            let tokens = order.map(|member| cases[pair[member]].inputs[step - 1]);
            model.start_pair(routed, tokens).unwrap();
            assert_eq!(
                ids.map(|id| model.position(id).unwrap()),
                positions.map(|p| p + step - 1),
                "no position commit before acknowledgement"
            );
            let rows = loop {
                if let Some(rows) = model.poll_pair().unwrap() {
                    break rows;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            };
            for (lane, logits) in rows.iter().enumerate() {
                let case = pair[order[lane]];
                finite(logits, width);
                let control = &controls[case][step];
                let mismatches = logits.iter().zip(control).filter(|(a, b)| a != b).count();
                let max_abs = logits
                    .iter()
                    .zip(control)
                    .map(|(&a, &b)| {
                        (f32::from_bits(u32::from(a) << 16) - f32::from_bits(u32::from(b) << 16))
                            .abs()
                    })
                    .fold(0.0_f32, f32::max);
                println!(
                    "{} step={step} changed={mismatches}/{width} max_abs={max_abs} winner={:?}",
                    cases[case].name,
                    argmax(logits)
                );
                // GEMM reduction order may change. No observed-error tolerance is
                // inferred: independent archived winners remain a strict gate.
                assert_eq!(
                    argmax(logits),
                    Some(cases[case].chosen[step]),
                    "candidate {} step {step}",
                    cases[case].name
                );
            }
        }
        for id in ids {
            model.release(id).unwrap();
        }
        assert_eq!(
            pool.granted(),
            model_grant,
            "all request-reachable caches refunded"
        );
    }
    exercise_runtime(model, &cases[0]);
    assert_eq!(pool.granted(), 0);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "serialized service-boundary control, candidate, rejection and teardown gate"
)]
#[ignore = "real 15 GB package, two near-4K histories; release-only serialized device gate"]
fn paired_decode_reaches_the_service_context_boundary_without_overrunning_either_member() {
    let _device = identities::DEVICE.lock().unwrap();
    let archived = cases();
    let pattern = &archived
        .iter()
        .find(|case| case.name == "window_513")
        .unwrap()
        .prompt;
    // Synthetic long contexts extend access/position qualification, not the
    // publisher's archived predictions or the labeled generation quality screen.
    let capacity = 4096;
    let scratch = 1 << 20; // Actual ordinary CUDA service configuration.
    let prompts = [capacity - 2, capacity - 3].map(|length| {
        pattern
            .iter()
            .copied()
            .cycle()
            .take(length)
            .collect::<Vec<_>>()
    });
    let inputs = [562, 126_641]; // Explicit common history, not autonomous rollout.
    let package = LocalModelPackage::open(environment("RIBN_GEMMA_PACKAGE")).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    let width = source.config().vocab_size();
    let context = CudaContext::new(0).unwrap();
    let pool = BytePool::new(20_u64 << 30).shared();
    let controls: Vec<Vec<Vec<u16>>> = {
        let stream = context.new_stream().unwrap();
        let model = Arc::new(Mutex::new(
            Gemma4CudaModel::load(&source, stream, &pool, capacity, scratch).unwrap(),
        ));
        prompts
            .iter()
            .map(|prompt| {
                let mut decoder = Gemma4CudaDecoder::shared(&model).unwrap();
                let mut rows = vec![decoder.prefill(prompt).unwrap()];
                for token in inputs {
                    rows.push(decoder.decode(token).unwrap());
                }
                for row in &rows {
                    finite(row, width);
                }
                decoder.close().unwrap();
                rows
            })
            .collect()
    };
    assert_eq!(pool.granted(), 0, "control unload before candidate load");
    let mut model = Gemma4CudaModelWide::load(&source, &context, &pool, capacity, scratch).unwrap();
    let model_grant = pool.granted();
    let ids = issued_ids::<2>();
    for (member, prompt) in prompts.iter().enumerate() {
        assert!(matches!(
            model
                .admit(ids[member], prompt.len() + inputs.len())
                .unwrap(),
            Admission::Ready
        ));
        assert_eq!(
            model.prefill(ids[member], prompt, true).unwrap().unwrap(),
            controls[member][0]
        );
    }
    for (step, token) in inputs.into_iter().enumerate() {
        let order = if step == 0 { [1, 0] } else { [0, 1] };
        model
            .start_pair(order.map(|member| ids[member]), [token; 2])
            .unwrap();
        assert_eq!(
            ids.map(|id| model.position(id).unwrap()),
            prompts.each_ref().map(|prompt| prompt.len() + step)
        );
        let rows = loop {
            if let Some(rows) = model.poll_pair().unwrap() {
                break rows;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        };
        for (lane, row) in rows.iter().enumerate() {
            let member = order[lane];
            finite(row, width);
            println!(
                "service boundary member={member} position={} winner={:?} control={:?}",
                model.position(ids[member]).unwrap(),
                argmax(row),
                argmax(&controls[member][step + 1])
            );
            assert_eq!(
                argmax(row),
                argmax(&controls[member][step + 1]),
                "common-history service boundary member {member}, step {step}"
            );
        }
    }
    let positions = ids.map(|id| model.position(id).unwrap());
    assert_eq!(positions, [capacity, capacity - 1]);
    let charged = pool.granted();
    // Exhausted reaches reject in either order without advancing either member.
    for order in [[0, 1], [1, 0]] {
        assert!(
            model
                .start_pair(order.map(|member| ids[member]), [36_501; 2])
                .is_err()
        );
        assert_eq!(ids.map(|id| model.position(id).unwrap()), positions);
        assert_eq!(pool.granted(), charged);
    }
    for id in ids {
        assert!(model.prefill(id, &[36_501], false).is_err());
        model.release(id).unwrap();
    }
    assert_eq!(pool.granted(), model_grant);
    model.close().unwrap();
    assert_eq!(pool.granted(), 0);
}
