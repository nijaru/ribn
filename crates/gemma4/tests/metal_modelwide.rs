//! Synthetic and real-artifact qualification tests for `Gemma4MetalModelWide`.
#![cfg(all(target_os = "macos", feature = "metal"))]

#[path = "common/weights.rs"]
mod weights;

use engine_gemma4::{
    Gemma4CpuDecoder, Gemma4CpuWeights, Gemma4MetalModelWide, Gemma4MetalModelWideConfig,
    Gemma4MetalModelWideExecutor, Gemma4MetalQuant, Gemma4MetalWeights, Gemma4Source,
    Gemma4SourceEncoding, PackedStepItem, argmax,
};
use ribn::driver::{Driver, DriverConfig};
use ribn::{
    Engine, EngineConfig, Event, FinishReason, GenerationOptions, Sampling, SchedulePolicy,
    TokenRequest,
};
use ribn_hf::LocalModelPackage;
use ribn_safetensors::SafeTensorArtifact;
use serde_json::Value;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn f32_from_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[test]
#[expect(
    clippy::too_many_lines,
    clippy::similar_names,
    reason = "end-to-end synthetic parity, continuous batching and sampling test"
)]
fn synthetic_modelwide_matches_cpu_and_batches_concurrently() {
    let dir = weights::Dir::new();
    let pkg = weights::package(&dir, &weights::inventory(), false);
    let source = Gemma4Source::open(&pkg).unwrap();

    // 1. CPU reference on two distinct prompts crossing the sliding window (window = 4)
    let cpu_weights = Arc::new(Gemma4CpuWeights::load(&source).unwrap());
    let prompt_a = vec![2_u32, 3, 4, 5, 6];
    let prompt_b = vec![2_u32, 6, 4];

    let mut cpu_a = Gemma4CpuDecoder::shared(Arc::clone(&cpu_weights)).unwrap();
    let cpu_a_prefill = cpu_a.prefill(&prompt_a).unwrap();
    let tok_a0 = argmax(&cpu_a_prefill).unwrap();
    let cpu_a_dec1 = cpu_a.decode(tok_a0).unwrap();
    let tok_a1 = argmax(&cpu_a_dec1).unwrap();
    let cpu_a_dec2 = cpu_a.decode(tok_a1).unwrap();

    let mut cpu_b = Gemma4CpuDecoder::shared(Arc::clone(&cpu_weights)).unwrap();
    let cpu_b_prefill = cpu_b.prefill(&prompt_b).unwrap();
    let tok_b0 = argmax(&cpu_b_prefill).unwrap();
    let cpu_b_dec1 = cpu_b.decode(tok_b0).unwrap();

    // 2. Metal modelwide single-sequence parity against CPU
    let limits = Gemma4MetalModelWideConfig {
        max_sequences: 4,
        max_positions: 32,
        max_batch_tokens: 32,
        num_kv_blocks: 0,
        quant: Gemma4MetalQuant::Bf16,
    };
    let mut model = Gemma4MetalModelWide::load(&source, limits).unwrap();
    let slot0 = model.allocate_slot().unwrap();
    let m_a_prefill = model.prefill_slot(slot0, &prompt_a, true).unwrap().unwrap();
    assert_close_logits(&m_a_prefill, &cpu_a_prefill, "prompt_a prefill");
    assert_eq!(argmax(&m_a_prefill).unwrap(), tok_a0);
    let m_a_dec1 = model.decode_slot(slot0, tok_a0).unwrap();
    assert_close_logits(&m_a_dec1, &cpu_a_dec1, "prompt_a dec1");
    let m_a_dec2 = model.decode_slot(slot0, tok_a1).unwrap();
    assert_close_logits(&m_a_dec2, &cpu_a_dec2, "prompt_a dec2");

    // 3. Packed mixed prefill + decode in ONE command buffer matches isolated outputs!
    model.reset_slot(slot0).unwrap();
    let slot1 = model.allocate_slot().unwrap();
    let out1 = model
        .step_packed(&[
            PackedStepItem {
                slot: slot0,
                tokens: &prompt_a[..3],
                need_logits: false,
            },
            PackedStepItem {
                slot: slot1,
                tokens: &prompt_b,
                need_logits: true,
            },
        ])
        .unwrap();
    assert_eq!(out1.argmax.len(), 1);
    assert_close_logits(
        out1.logits_row(0),
        &cpu_b_prefill,
        "batched prompt_b prefill",
    );
    let b_tok0 = out1.argmax[0];
    assert_eq!(b_tok0, tok_b0);

    let out2 = model
        .step_packed(&[
            PackedStepItem {
                slot: slot0,
                tokens: &prompt_a[3..],
                need_logits: true,
            },
            PackedStepItem {
                slot: slot1,
                tokens: &[b_tok0],
                need_logits: true,
            },
        ])
        .unwrap();
    assert_eq!(out2.argmax.len(), 2);
    assert_close_logits(out2.logits_row(0), &m_a_prefill, "batched prompt_a prefill");
    assert_close_logits(out2.logits_row(1), &cpu_b_dec1, "batched prompt_b dec1");

    // 4. Release slots and exercise full continuous-batching GenerationExecutor + sampling
    model.free_slot(slot0).unwrap();
    model.free_slot(slot1).unwrap();

    let executor = Gemma4MetalModelWideExecutor::new(model).unwrap();
    let engine_cfg = EngineConfig {
        max_active_requests: 4,
        max_queued_requests: 8,
        ..EngineConfig::default()
    };
    let policy = SchedulePolicy {
        max_batch_tokens: 32,
        prefill_chunk_tokens: 4,
        ..SchedulePolicy::default()
    };
    let engine = Engine::new(executor, engine_cfg, policy).unwrap();
    let driver_cfg = DriverConfig::for_engine(&engine);
    let (mut driver, handle) = Driver::spawn(engine, driver_cfg).unwrap();

    let greedy_req = |prompt: Vec<u32>, max_output_tokens: u32| {
        TokenRequest::new(
            prompt,
            GenerationOptions {
                max_output_tokens,
                ..GenerationOptions::default()
            },
        )
    };
    let sampled_req = |prompt: Vec<u32>, seed: u64| {
        TokenRequest::new(
            prompt,
            GenerationOptions {
                max_output_tokens: 4,
                sampling: Sampling {
                    temperature: 0.8,
                    top_p: 0.9,
                    top_k: 4,
                    seed: Some(seed),
                },
                stop_tokens: Vec::new(),
            },
        )
    };

    let s1 = handle
        .stream_blocking(greedy_req(prompt_a.clone(), 3))
        .unwrap();
    let s2 = handle
        .stream_blocking(sampled_req(prompt_b.clone(), 42))
        .unwrap();
    let s3 = handle.stream_blocking(sampled_req(prompt_b, 42)).unwrap();
    let s4 = handle.stream_blocking(greedy_req(prompt_a, 8)).unwrap();
    s4.cancel();

    let collect_tokens = |mut stream: ribn::driver::GenerationStream| -> (Vec<u32>, FinishReason) {
        let mut tokens = Vec::new();
        let mut reason = None;
        while let Some(event) = stream.next_blocking() {
            match event.unwrap() {
                Event::Token { token, .. } => tokens.push(token),
                Event::Finished { reason: r, .. } => reason = Some(r),
            }
        }
        (tokens, reason.unwrap())
    };

    let (t1, r1) = collect_tokens(s1);
    let (t2, r2) = collect_tokens(s2);
    let (t3, r3) = collect_tokens(s3);
    let (_, r4) = collect_tokens(s4);
    assert_eq!(r1, FinishReason::Length);
    assert_eq!(t1[0], tok_a0);
    assert_eq!(t1[1], tok_a1);
    assert_eq!(r2, FinishReason::Length);
    assert_eq!(r3, FinishReason::Length);
    assert_eq!(
        t2, t3,
        "same seed must yield deterministic stochastic tokens"
    );
    assert_eq!(r4, FinishReason::Cancelled);
    driver.shutdown().unwrap();
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "end-to-end synthetic prefix reuse, bit-exactness, and LRU eviction test"
)]
fn synthetic_prefix_cache_is_bit_exact_and_evicts_lru() {
    let dir = weights::Dir::new();
    let pkg = weights::package(&dir, &weights::inventory(), false);
    let source = Gemma4Source::open(&pkg).unwrap();

    // 40-token prompts sharing a 32-token (2-page) prefix with distinct 8-token suffixes.
    let shared_prefix: Vec<u32> = (0_u32..32).map(|i| 2 + (i % 5)).collect();
    let mut prompt_a = shared_prefix.clone();
    prompt_a.extend([2, 3, 4, 5, 6, 5, 4, 3]);
    let mut prompt_b = shared_prefix;
    prompt_b.extend([6, 5, 4, 3, 2, 3, 4, 5]);
    assert_eq!(prompt_a.len(), 40);
    assert_eq!(prompt_b.len(), 40);

    // 1. Cold baseline for prompt_b on an empty pool.
    let limits = Gemma4MetalModelWideConfig {
        max_sequences: 4,
        max_positions: 64,
        max_batch_tokens: 64,
        num_kv_blocks: 8,
        quant: Gemma4MetalQuant::Bf16,
    };
    let mut baseline = Gemma4MetalModelWide::load(&source, limits).unwrap();
    let base_slot = baseline.allocate_slot().unwrap();
    let cold_b_logits = baseline
        .prefill_slot(base_slot, &prompt_b, true)
        .unwrap()
        .unwrap();
    let cold_b_tok0 = argmax(&cold_b_logits).unwrap();
    let cold_b_dec1 = baseline.decode_slot(base_slot, cold_b_tok0).unwrap();
    let cold_b_tok1 = argmax(&cold_b_dec1).unwrap();

    // 2. Warm prompt_a, then reuse the 32-token cached prefix for prompt_b.
    let mut model = Gemma4MetalModelWide::load(&source, limits).unwrap();
    let (slot_a, matched_a) = model.allocate_slot_with_prefix(&prompt_a).unwrap();
    assert_eq!(matched_a, 0);
    let _ = model.prefill_slot(slot_a, &prompt_a, true).unwrap();
    assert_eq!(model.prefix_cache_stats().cached_blocks, 2);

    // Even while slot_a is still live, slot_b shares the 2 read-only prefix blocks zero-copy.
    let (slot_b, matched_b) = model.allocate_slot_with_prefix(&prompt_b).unwrap();
    assert_eq!(matched_b, 32);
    let cached_b_logits = model
        .prefill_slot(slot_b, &prompt_b[matched_b..], true)
        .unwrap()
        .unwrap();
    assert_eq!(
        cached_b_logits, cold_b_logits,
        "cached-prefix prefill must be bit-for-bit identical to cold prefill"
    );
    let cached_b_dec1 = model.decode_slot(slot_b, cold_b_tok0).unwrap();
    assert_eq!(
        cached_b_dec1, cold_b_dec1,
        "decode following cached-prefix prefill must be bit-for-bit identical"
    );
    model.free_slot(slot_a).unwrap();
    model.free_slot(slot_b).unwrap();

    // 3. Verify Engine + Executor transparently skips the cached prefix on admission.
    let executor = Gemma4MetalModelWideExecutor::new(model).unwrap();
    let engine_cfg = EngineConfig {
        max_active_requests: 4,
        max_queued_requests: 8,
        ..EngineConfig::default()
    };
    let policy = SchedulePolicy {
        max_batch_tokens: 64,
        prefill_chunk_tokens: 16,
        ..SchedulePolicy::default()
    };
    let engine = Engine::new(executor, engine_cfg, policy).unwrap();
    let driver_cfg = DriverConfig::for_engine(&engine);
    let (mut driver, handle) = Driver::spawn(engine, driver_cfg).unwrap();
    let mut stream = handle
        .stream_blocking(TokenRequest::new(
            prompt_b.clone(),
            GenerationOptions {
                max_output_tokens: 2,
                ..GenerationOptions::default()
            },
        ))
        .unwrap();
    let mut tokens = Vec::new();
    while let Some(event) = stream.next_blocking() {
        if let Event::Token { token, .. } = event.unwrap() {
            tokens.push(token);
        }
    }
    assert_eq!(tokens, vec![cold_b_tok0, cold_b_tok1]);
    driver.shutdown().unwrap();

    // 4. Verify tight pool (4 blocks = 64 tokens) evicts unreferenced cached blocks LRU.
    let tight = Gemma4MetalModelWideConfig {
        max_sequences: 2,
        max_positions: 64,
        max_batch_tokens: 64,
        num_kv_blocks: 4,
        quant: Gemma4MetalQuant::Bf16,
    };
    let mut evicted_model = Gemma4MetalModelWide::load(&source, tight).unwrap();
    let p1: Vec<u32> = vec![2; 33]; // uses 3 blocks, caches 2
    let p2: Vec<u32> = vec![3; 33]; // needs 3 blocks -> must evict p1's cached blocks
    let (s1, m1) = evicted_model.allocate_slot_with_prefix(&p1).unwrap();
    assert_eq!(m1, 0);
    let _ = evicted_model.prefill_slot(s1, &p1, true).unwrap();
    evicted_model.free_slot(s1).unwrap();

    let (s2, m2) = evicted_model.allocate_slot_with_prefix(&p2).unwrap();
    assert_eq!(m2, 0);
    let _ = evicted_model.prefill_slot(s2, &p2, true).unwrap();
    evicted_model.free_slot(s2).unwrap();
    assert!(
        evicted_model.prefix_cache_stats().evictions >= 1,
        "tight KV pool must evict unreferenced cached blocks under pressure"
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "end-to-end slot and KV-pool backpressure, deferral wakeup, and permanent-refusal verification"
)]
fn kv_pool_and_slot_pressure_defer_wake_and_reject_impossibly_large_requests() {
    let dir = weights::Dir::new();
    let pkg = weights::package(&dir, &weights::inventory(), false);
    let source = Gemma4Source::open(&pkg).unwrap();
    let pool_cfg = Gemma4MetalModelWideConfig {
        max_sequences: 2,
        max_positions: 96,
        max_batch_tokens: 64,
        num_kv_blocks: 4, // 64 tokens total across the entire pool
        quant: Gemma4MetalQuant::Bf16,
    };
    let model = Gemma4MetalModelWide::load(&source, pool_cfg).unwrap();
    let exec = Gemma4MetalModelWideExecutor::new(model).unwrap();
    let engine_cfg = EngineConfig {
        max_active_requests: 2,
        max_queued_requests: 8,
        ..EngineConfig::default()
    };
    let policy = SchedulePolicy {
        max_batch_tokens: 64,
        prefill_chunk_tokens: 32,
        ..SchedulePolicy::default()
    };
    let engine = Engine::new(exec, engine_cfg, policy).unwrap();
    let driver_cfg = DriverConfig::for_engine(&engine);
    let (mut driver, handle) = Driver::spawn(engine, driver_cfg).unwrap();

    // 1. Permanently impossible request (65 prompt + 1 output = 5 pages > 4 pool blocks)
    //    must reject immediately instead of hanging in Admission::Deferred.
    let impossible = handle
        .stream_blocking(TokenRequest::new(
            vec![2_u32; 65],
            GenerationOptions {
                max_output_tokens: 1,
                ..GenerationOptions::default()
            },
        ))
        .unwrap();
    let mut s = impossible;
    let first_event = s
        .next_blocking()
        .expect("impossible request must emit terminal failure event");
    assert!(
        matches!(
            first_event,
            Err(_)
                | Ok(Event::Finished {
                    reason: ribn::FinishReason::Failed(_),
                    ..
                })
        ),
        "request exceeding total KV pool blocks must fail cleanly, got {first_event:?}"
    );

    // 2. Concurrent KV-pool & slot burst: 4 requests, each needing 3 pages (34 prompt + 2 output).
    //    Only 1 fits at a time in 4 blocks (3 + 3 = 6 > 4), so requests 2..4 must defer and
    //    wake cleanly as peers complete without ever triggering mid-generation KV OOM.
    let mut streams = Vec::new();
    for idx in 0_u32..4 {
        let mut prompt = vec![2 + (idx % 4); 34];
        prompt[33] = 2 + ((idx + 1) % 4);
        streams.push(
            handle
                .stream_blocking(TokenRequest::new(
                    prompt,
                    GenerationOptions {
                        max_output_tokens: 2,
                        ..GenerationOptions::default()
                    },
                ))
                .unwrap(),
        );
    }
    for (idx, mut stream) in streams.into_iter().enumerate() {
        let mut count = 0_usize;
        while let Some(ev) = stream.next_blocking() {
            match ev.unwrap() {
                Event::Token { .. } => count += 1,
                Event::Finished { .. } => break,
            }
        }
        assert_eq!(
            count, 2,
            "deferred burst request {idx} must wake and produce all 2 tokens"
        );
    }

    // 3. Slot saturation with small requests (4 concurrent 4-token prompts on max_sequences=2).
    let mut small_streams = Vec::new();
    for idx in 0_u32..4 {
        small_streams.push(
            handle
                .stream_blocking(TokenRequest::new(
                    vec![2, 3, 4, 2 + (idx % 4)],
                    GenerationOptions {
                        max_output_tokens: 3,
                        ..GenerationOptions::default()
                    },
                ))
                .unwrap(),
        );
    }
    for (idx, mut stream) in small_streams.into_iter().enumerate() {
        let mut count = 0_usize;
        while let Some(ev) = stream.next_blocking() {
            match ev.unwrap() {
                Event::Token { .. } => count += 1,
                Event::Finished { .. } => break,
            }
        }
        assert_eq!(
            count, 3,
            "slot-deferred request {idx} must wake and produce all 3 tokens"
        );
    }

    driver.shutdown().unwrap();
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "end-to-end Metal partial-envelope admission, growth preemption, and bit-exact CPU oracle replay"
)]
fn kv_pool_partial_envelope_batches_preempts_and_replays_bit_identically() {
    let dir = weights::Dir::new();
    let pkg = weights::package(&dir, &weights::inventory(), false);
    let source = Gemma4Source::open(&pkg).unwrap();

    // Two 20-token prompts (2 pages each) generating 22 tokens each (reach = 41 tokens = 3 pages).
    // Pool holds 4 blocks (64 tokens): partial-envelope admission admits BOTH concurrently
    // (2 + 2 = 4 <= 4), batches them at M=2 through position 32, then triggers completion-safe
    // recomputation preemption on page-3 growth and replays the victim with exact CPU parity.
    let prompt_a: Vec<u32> = (0_u32..20).map(|i| 2 + (i % 5)).collect();
    let prompt_b: Vec<u32> = (0_u32..20).map(|i| 2 + ((i + 3) % 5)).collect();
    let cpu_weights = Arc::new(Gemma4CpuWeights::load(&source).unwrap());
    let run_cpu = |prompt: &[u32], count: usize| -> Vec<u32> {
        let mut dec = Gemma4CpuDecoder::shared(Arc::clone(&cpu_weights)).unwrap();
        let mut logits = dec.prefill(prompt).unwrap();
        let mut out = Vec::with_capacity(count);
        for step in 0..count {
            let tok = argmax(&logits).unwrap();
            out.push(tok);
            if step + 1 < count {
                logits = dec.decode(tok).unwrap();
            }
        }
        out
    };
    let expected_a = run_cpu(&prompt_a, 22);
    let expected_b = run_cpu(&prompt_b, 22);

    let pool_cfg = Gemma4MetalModelWideConfig {
        max_sequences: 2,
        max_positions: 64,
        max_batch_tokens: 64,
        num_kv_blocks: 4,
        quant: Gemma4MetalQuant::Bf16,
    };
    let model = Gemma4MetalModelWide::load(&source, pool_cfg).unwrap();
    let exec = Gemma4MetalModelWideExecutor::new(model).unwrap();
    let mut engine = Engine::new(
        exec,
        EngineConfig {
            max_active_requests: 2,
            max_queued_requests: 4,
            max_buffered_events: 128,
            max_events_per_request: 64,
            ..EngineConfig::default()
        },
        SchedulePolicy {
            max_batch_tokens: 64,
            prefill_chunk_tokens: 64,
            ..SchedulePolicy::default()
        },
    )
    .unwrap();

    let req_a = engine
        .enqueue(TokenRequest::new(
            prompt_a,
            GenerationOptions {
                max_output_tokens: 22,
                ..GenerationOptions::default()
            },
        ))
        .unwrap();
    let req_b = engine
        .enqueue(TokenRequest::new(
            prompt_b,
            GenerationOptions {
                max_output_tokens: 22,
                ..GenerationOptions::default()
            },
        ))
        .unwrap();

    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    let mut done_a = None;
    let mut done_b = None;
    let mut saw_both_active = false;
    for _ in 0..256 {
        engine.step().unwrap();
        if engine.status().active_sequences == 2 {
            saw_both_active = true;
        }
        while let Some(ev) = engine.pop_event() {
            match ev {
                Event::Token { request, token } if request == req_a => out_a.push(token),
                Event::Token { request, token } if request == req_b => out_b.push(token),
                Event::Finished {
                    request,
                    reason,
                    usage,
                } if request == req_a => done_a = Some((reason, usage)),
                Event::Finished {
                    request,
                    reason,
                    usage,
                } if request == req_b => done_b = Some((reason, usage)),
                _ => {}
            }
        }
        if done_a.is_some() && done_b.is_some() {
            break;
        }
    }

    assert!(saw_both_active, "both requests must batch concurrently");
    assert!(
        engine.preemptions() >= 1,
        "KV pool saturation must trigger recomputation preemption"
    );
    assert_eq!(out_a, expected_a);
    assert_eq!(out_b, expected_b);
    assert_eq!(
        done_a,
        Some((
            FinishReason::Length,
            ribn::Usage {
                prompt_tokens: 20,
                completion_tokens: 22,
            }
        ))
    );
    assert_eq!(
        done_b,
        Some((
            FinishReason::Length,
            ribn::Usage {
                prompt_tokens: 20,
                completion_tokens: 22,
            }
        ))
    );
    assert_eq!(engine.prefix_cache_stats().unwrap().pinned_blocks, 0);
    engine.shutdown().unwrap();
}

fn assert_close_logits(actual: &[u16], expected: &[u16], label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let fa = f32_from_bf16(a);
        let fe = f32_from_bf16(e);
        assert!(fa.is_finite(), "{label} idx {i} not finite");
        let tol = 0.05_f32.max(fe.abs() * 0.02);
        assert!(
            (fa - fe).abs() <= tol,
            "{label} idx {i}: actual={fa} ({a:#06x}) expected={fe} ({e:#06x})"
        );
    }
}

fn read_bf16_tensor(artifact: &SafeTensorArtifact, name: &str) -> Vec<u16> {
    let tensor = artifact.tensor(name).expect(name);
    let mut bytes = Vec::new();
    tensor.reader().read_to_end(&mut bytes).unwrap();
    let (chunks, rem) = bytes.as_chunks::<2>();
    assert!(rem.is_empty());
    chunks
        .iter()
        .map(|&chunk| u16::from_le_bytes(chunk))
        .collect()
}

fn logit_metrics(actual: &[u16], reference: &[u16]) -> (f64, f64, f64, f64) {
    assert_eq!(actual.len(), reference.len());
    let mut sq_err = 0.0_f64;
    let mut sq_ref = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut max_a = f64::NEG_INFINITY;
    let mut max_r = f64::NEG_INFINITY;
    for (&a, &r) in actual.iter().zip(reference) {
        let fa = f64::from(f32_from_bf16(a));
        let fr = f64::from(f32_from_bf16(r));
        assert!(fa.is_finite(), "non-finite Metal logit");
        let diff = fa - fr;
        sq_err += diff * diff;
        sq_ref += fr * fr;
        max_abs = max_abs.max(diff.abs());
        max_a = max_a.max(fa);
        max_r = max_r.max(fr);
    }
    let lse_a = max_a
        + actual
            .iter()
            .map(|&a| (f64::from(f32_from_bf16(a)) - max_a).exp())
            .sum::<f64>()
            .ln();
    let lse_r = max_r
        + reference
            .iter()
            .map(|&r| (f64::from(f32_from_bf16(r)) - max_r).exp())
            .sum::<f64>()
            .ln();
    let mut kl = 0.0_f64;
    for (&a, &r) in actual.iter().zip(reference) {
        let log_actual = f64::from(f32_from_bf16(a)) - lse_a;
        let log_ref = f64::from(f32_from_bf16(r)) - lse_r;
        let prob_ref = log_ref.exp();
        kl += prob_ref * (log_ref - log_actual);
    }
    #[expect(clippy::cast_precision_loss, reason = "vocab length fits in f64")]
    let rmse = (sq_err / actual.len() as f64).sqrt();
    let rel_l2 = (sq_err / sq_ref).sqrt();
    (rmse, rel_l2, kl, max_abs)
}

#[test]
#[ignore = "loads 14.9 GB Gemma 4 E4B on M3 Max; runs 12-row oracle + speed sweep; use --release -- --nocapture"]
#[expect(
    clippy::too_many_lines,
    reason = "12-row oracle qualification plus direct prefill and batched decode sweep"
)]
fn e4b_modelwide_matches_12_row_bf16_oracle_and_benchmarks() {
    let pkg_path =
        PathBuf::from(std::env::var_os("RIBN_GEMMA_PACKAGE").expect("RIBN_GEMMA_PACKAGE"));
    let ref_path =
        PathBuf::from(std::env::var_os("RIBN_GEMMA_REFERENCE").expect("RIBN_GEMMA_REFERENCE"));
    let tensors_path = ref_path.with_file_name("tensors.safetensors");

    let package = LocalModelPackage::open(&pkg_path).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    let reference: Value = serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
    let ref_tensors = SafeTensorArtifact::open(&tensors_path).unwrap();

    let t_load = Instant::now();
    let limits = Gemma4MetalModelWideConfig {
        max_sequences: 16,
        max_positions: 2048,
        max_batch_tokens: 512,
        num_kv_blocks: 512,
        quant: Gemma4MetalQuant::Bf16,
    };
    let mut model = Gemma4MetalModelWide::load(&source, limits).unwrap();
    println!("load_ms={:.1}", t_load.elapsed().as_secs_f64() * 1e3);

    let slot = model.allocate_slot().unwrap();
    let mut total_rows = 0_usize;
    let mut matched_winners = 0_usize;
    let mut worst_kl = 0.0_f64;
    let mut worst_rel_l2 = 0.0_f64;
    let mut window_513_cold: Option<(Vec<u32>, Vec<u16>, f64)> = None;

    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let prompt: Vec<u32> = case["prompt_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
            .collect();
        let steps = case["steps"].as_array().unwrap();
        model.reset_slot(slot).unwrap();

        let t0 = Instant::now();
        let mut logits = model.prefill_slot(slot, &prompt, true).unwrap().unwrap();
        let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
        if name == "window_513" {
            window_513_cold = Some((prompt.clone(), logits.clone(), prefill_ms));
        }

        for (step_idx, step) in steps.iter().enumerate() {
            let step_ms = if step_idx == 0 {
                prefill_ms
            } else {
                let input_tok = u32::try_from(step["input_ids"][0].as_u64().unwrap()).unwrap();
                let td = Instant::now();
                logits = model.decode_slot(slot, input_tok).unwrap();
                td.elapsed().as_secs_f64() * 1e3
            };
            let expected = u32::try_from(step["chosen"].as_u64().unwrap()).unwrap();
            let actual = argmax(&logits).unwrap();
            total_rows += 1;
            if actual == expected {
                matched_winners += 1;
            }
            let tensor_key = format!("{name}.{step_idx}.logits");
            let ref_logits = read_bf16_tensor(&ref_tensors, &tensor_key);
            let (rmse, rel_l2, kl, max_abs) = logit_metrics(&logits, &ref_logits);
            worst_kl = worst_kl.max(kl);
            worst_rel_l2 = worst_rel_l2.max(rel_l2);
            println!(
                "{name}[{step_idx}] (len={}): actual={actual} expected={expected} ms={step_ms:.2} rmse={rmse:.4} rel_l2={rel_l2:.5} kl={kl:.6} max_abs={max_abs:.3}",
                if step_idx == 0 { prompt.len() } else { 1 }
            );
        }
    }
    println!(
        "ORACLE SUMMARY: matched={matched_winners}/{total_rows}, worst_rel_l2={worst_rel_l2:.5}, worst_kl={worst_kl:.6}"
    );
    assert_eq!(
        matched_winners, total_rows,
        "every archived BF16 oracle row must match winner"
    );
    assert!(worst_kl <= 0.01, "worst KL {worst_kl} exceeded 0.01");
    assert!(
        worst_rel_l2 <= 0.035,
        "worst rel_l2 {worst_rel_l2} exceeded 0.035"
    );

    if let Some((prompt_513, cold_logits, cold_ms)) = window_513_cold {
        let (cached_slot, matched) = model.allocate_slot_with_prefix(&prompt_513).unwrap();
        assert_eq!(
            matched, 512,
            "window_513 must reuse 32 cached 16-token blocks"
        );
        let t_hit = Instant::now();
        let cached_logits = model
            .prefill_slot(cached_slot, &prompt_513[matched..], true)
            .unwrap()
            .unwrap();
        let cached_ms = t_hit.elapsed().as_secs_f64() * 1e3;
        assert_eq!(
            cached_logits, cold_logits,
            "E4B cached-prefix prefill must be bit-for-bit identical to cold prefill"
        );
        let stats = model.prefix_cache_stats();
        println!(
            "E4B PREFIX CACHE (len={}, hit={matched}): cold={cold_ms:.2} ms -> cached={cached_ms:.2} ms ({:.1}x) stats={stats:?}",
            prompt_513.len(),
            cold_ms / cached_ms
        );
        model.free_slot(cached_slot).unwrap();
    }

    // Speed benchmarks: single decode M=1, batched decode M=2,4,8,16, and prefill 18, 64, 128!
    let chat_prompt: Vec<u32> = reference["cases"][0]["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    let long8: Vec<u32> = chat_prompt.iter().copied().cycle().take(8).collect();
    let long32: Vec<u32> = chat_prompt.iter().copied().cycle().take(32).collect();
    let long64: Vec<u32> = chat_prompt.iter().copied().cycle().take(64).collect();
    let long128: Vec<u32> = chat_prompt.iter().copied().cycle().take(128).collect();
    let long256: Vec<u32> = chat_prompt.iter().copied().cycle().take(256).collect();
    let long512: Vec<u32> = chat_prompt.iter().copied().cycle().take(512).collect();

    // Verify chunked (16) vs token-by-token (1) prefill bit-exact invariance on E4B
    let p16 = &chat_prompt[..16];
    model.reset_slot(slot).unwrap();
    let l16_whole = model.prefill_slot(slot, p16, true).unwrap().unwrap();
    model.reset_slot(slot).unwrap();
    let mut l16_tok = Vec::new();
    for (i, &t) in p16.iter().enumerate() {
        let out = model.prefill_slot(slot, &[t], i + 1 == p16.len()).unwrap();
        if let Some(l) = out {
            l16_tok = l;
        }
    }
    assert_eq!(
        l16_whole, l16_tok,
        "16-token prefill must match token-by-token prefill bit-for-bit"
    );

    println!("\n=== DIRECT PREFILL BENCHMARKS (warm, 3 runs) ===");
    for (label, tokens) in [
        ("prefill_8", long8.as_slice()),
        ("prefill_18", chat_prompt.as_slice()),
        ("prefill_32", long32.as_slice()),
        ("prefill_64", long64.as_slice()),
        ("prefill_128", long128.as_slice()),
        ("prefill_256", long256.as_slice()),
        ("prefill_512", long512.as_slice()),
    ] {
        let mut times = Vec::new();
        for _ in 0..3 {
            model.reset_slot(slot).unwrap();
            let t = Instant::now();
            let _ = model.prefill_slot(slot, tokens, true).unwrap();
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let min_ms = times.iter().copied().fold(f64::INFINITY, f64::min);
        #[expect(clippy::cast_precision_loss, reason = "small prompt length")]
        let tok_s = (tokens.len() as f64) / (min_ms / 1e3);
        println!("{label}: min={min_ms:.2} ms ({tok_s:.1} tok/s) runs={times:.2?}");
    }

    println!("\n=== DIRECT BATCHED DECODE BENCHMARKS (warm, 5 steps) ===");
    let mut all_slots = vec![slot];
    for _ in 1..16 {
        all_slots.push(model.allocate_slot().unwrap());
    }
    for &batch_size in &[1_usize, 2, 4, 8, 16] {
        for &s in &all_slots[..batch_size] {
            model.reset_slot(s).unwrap();
            let _ = model.prefill_slot(s, &chat_prompt, false).unwrap();
        }
        let mut step_times = Vec::new();
        let tok_slice = [236_776_u32];
        for _ in 0..5 {
            let items: Vec<PackedStepItem<'_>> = all_slots[..batch_size]
                .iter()
                .map(|&s| PackedStepItem {
                    slot: s,
                    tokens: &tok_slice,
                    need_logits: true,
                })
                .collect();
            let t = Instant::now();
            let _ = model.step_packed(&items).unwrap();
            step_times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let min_ms = step_times.iter().copied().fold(f64::INFINITY, f64::min);
        let mean_ms = step_times.iter().sum::<f64>() / 5.0;
        #[expect(clippy::cast_precision_loss, reason = "small batch size")]
        let agg_tok_s = (batch_size as f64) / (mean_ms / 1e3);
        println!(
            "decode_M={batch_size:2}: min={min_ms:.2} ms mean={mean_ms:.2} ms agg={agg_tok_s:.1} tok/s runs={step_times:.2?}"
        );
    }
}

#[test]
#[ignore = "loads quantized Q8 and Q4 Gemma 4 E4B on M3 Max; checks 12-row BF16 oracle and speed sweep; use --release -- --nocapture"]
#[expect(
    clippy::too_many_lines,
    reason = "Q8 and Q4 real E4B oracle diagnostics, prefix-cache bit-exactness, and direct speed sweep"
)]
fn e4b_modelwide_quantized_q8_and_q4_oracle_and_benchmarks() {
    let pkg_path =
        PathBuf::from(std::env::var_os("RIBN_GEMMA_PACKAGE").expect("RIBN_GEMMA_PACKAGE"));
    let ref_path =
        PathBuf::from(std::env::var_os("RIBN_GEMMA_REFERENCE").expect("RIBN_GEMMA_REFERENCE"));
    let tensors_path = ref_path.with_file_name("tensors.safetensors");

    let package = LocalModelPackage::open(&pkg_path).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    let reference: Value = serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
    let ref_tensors = SafeTensorArtifact::open(&tensors_path).unwrap();

    let chat_prompt: Vec<u32> = reference["cases"][0]["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    let long64: Vec<u32> = chat_prompt.iter().copied().cycle().take(64).collect();
    let long128: Vec<u32> = chat_prompt.iter().copied().cycle().take(128).collect();
    let long512: Vec<u32> = chat_prompt.iter().copied().cycle().take(512).collect();

    for quant in [Gemma4MetalQuant::Q8, Gemma4MetalQuant::Q4] {
        println!("\n================ E4B METAL QUANTIZED: {quant:?} ================");
        let t_load = Instant::now();
        let limits = Gemma4MetalModelWideConfig {
            max_sequences: 16,
            max_positions: 2048,
            max_batch_tokens: 512,
            num_kv_blocks: 512,
            quant,
        };
        let mut model = Gemma4MetalModelWide::load(&source, limits).unwrap();
        #[expect(clippy::cast_precision_loss, reason = "display GiB")]
        let weight_gib = model.weights_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "load_ms={:.1} weight_bytes={} ({weight_gib:.2} GiB)",
            t_load.elapsed().as_secs_f64() * 1e3,
            model.weights_bytes()
        );

        let slot = model.allocate_slot().unwrap();
        let mut total_rows = 0_usize;
        let mut matched_winners = 0_usize;
        let mut worst_kl = 0.0_f64;
        let mut worst_rel_l2 = 0.0_f64;
        let mut window_513_cold: Option<(Vec<u32>, Vec<u16>, f64)> = None;

        for case in reference["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let prompt: Vec<u32> = case["prompt_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                .collect();
            let steps = case["steps"].as_array().unwrap();
            model.reset_slot(slot).unwrap();

            let t0 = Instant::now();
            let mut logits = model.prefill_slot(slot, &prompt, true).unwrap().unwrap();
            let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
            if name == "window_513" {
                window_513_cold = Some((prompt.clone(), logits.clone(), prefill_ms));
            }
            for (step_idx, step) in steps.iter().enumerate() {
                if step_idx > 0 {
                    let input_tok = u32::try_from(step["input_ids"][0].as_u64().unwrap()).unwrap();
                    logits = model.decode_slot(slot, input_tok).unwrap();
                }
                let expected = u32::try_from(step["chosen"].as_u64().unwrap()).unwrap();
                let actual = argmax(&logits).unwrap();
                total_rows += 1;
                if actual == expected {
                    matched_winners += 1;
                }
                let tensor_key = format!("{name}.{step_idx}.logits");
                let ref_logits = read_bf16_tensor(&ref_tensors, &tensor_key);
                let (_rmse, rel_l2, kl, max_abs) = logit_metrics(&logits, &ref_logits);
                worst_kl = worst_kl.max(kl);
                worst_rel_l2 = worst_rel_l2.max(rel_l2);
                println!(
                    "{quant:?} {name}[{step_idx}]: actual={actual} expected={expected} rel_l2={rel_l2:.5} kl={kl:.6} max_abs={max_abs:.3}"
                );
            }
        }
        println!(
            "{quant:?} ORACLE SUMMARY: matched={matched_winners}/{total_rows}, worst_rel_l2={worst_rel_l2:.5}, worst_kl={worst_kl:.6}"
        );
        if quant == Gemma4MetalQuant::Q8 {
            assert_eq!(
                matched_winners, total_rows,
                "Q8 must match all 12 BF16 oracle winners"
            );
            assert!(worst_kl <= 0.02, "Q8 worst KL {worst_kl} exceeded 0.02");
        } else {
            assert!(
                matched_winners >= 10,
                "Q4 matched only {matched_winners}/{total_rows} BF16 oracle winners"
            );
        }

        if let Some((prompt_513, cold_logits, cold_ms)) = window_513_cold {
            let (cached_slot, matched) = model.allocate_slot_with_prefix(&prompt_513).unwrap();
            assert_eq!(matched, 512);
            let t_hit = Instant::now();
            let cached_logits = model
                .prefill_slot(cached_slot, &prompt_513[matched..], true)
                .unwrap()
                .unwrap();
            let cached_ms = t_hit.elapsed().as_secs_f64() * 1e3;
            assert_eq!(
                cached_logits, cold_logits,
                "{quant:?} cached-prefix prefill must be bit-identical to cold prefill"
            );
            println!(
                "{quant:?} PREFIX CACHE (len=513, hit=512): cold={cold_ms:.2} ms -> cached={cached_ms:.2} ms ({:.1}x)",
                cold_ms / cached_ms
            );
            model.free_slot(cached_slot).unwrap();
        }

        for (label, tokens) in [
            ("prefill_18", chat_prompt.as_slice()),
            ("prefill_64", long64.as_slice()),
            ("prefill_128", long128.as_slice()),
            ("prefill_512", long512.as_slice()),
        ] {
            let mut times = Vec::new();
            for _ in 0..3 {
                model.reset_slot(slot).unwrap();
                let t = Instant::now();
                let _ = model.prefill_slot(slot, tokens, true).unwrap();
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let min_ms = times.iter().copied().fold(f64::INFINITY, f64::min);
            #[expect(clippy::cast_precision_loss, reason = "small prompt length")]
            let tok_s = (tokens.len() as f64) / (min_ms / 1e3);
            println!("{quant:?} {label}: min={min_ms:.2} ms ({tok_s:.1} tok/s)");
        }

        let mut all_slots = vec![slot];
        for _ in 1..16 {
            all_slots.push(model.allocate_slot().unwrap());
        }
        for &batch_size in &[1_usize, 2, 4, 8, 16] {
            for &s in &all_slots[..batch_size] {
                model.reset_slot(s).unwrap();
                let _ = model.prefill_slot(s, &chat_prompt, false).unwrap();
            }
            let mut step_times = Vec::new();
            let tok_slice = [236_776_u32];
            for _ in 0..5 {
                let items: Vec<PackedStepItem<'_>> = all_slots[..batch_size]
                    .iter()
                    .map(|&s| PackedStepItem {
                        slot: s,
                        tokens: &tok_slice,
                        need_logits: true,
                    })
                    .collect();
                let t = Instant::now();
                let _ = model.step_packed(&items).unwrap();
                step_times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let min_ms = step_times.iter().copied().fold(f64::INFINITY, f64::min);
            let mean_ms = step_times.iter().sum::<f64>() / 5.0;
            #[expect(clippy::cast_precision_loss, reason = "small batch size")]
            let agg_tok_s = (batch_size as f64) / (mean_ms / 1e3);
            println!(
                "{quant:?} decode_M={batch_size:2}: min={min_ms:.2} ms mean={mean_ms:.2} ms agg={agg_tok_s:.1} tok/s"
            );
        }
    }
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let mut bits = value.to_bits();
    if (bits & 0x7f80_0000) == 0x7f80_0000 {
        #[expect(clippy::cast_possible_truncation, reason = "top 16 bits fit in u16")]
        return if (bits & 0x007f_ffff) != 0 {
            ((bits >> 16) | 0x0040) as u16
        } else {
            (bits >> 16) as u16
        };
    }
    bits = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (bits >> 16) as u16
}

fn write_safetensors_raw(path: &std::path::Path, entries: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, dtype, shape, bytes) in entries {
        let start = payload.len();
        payload.extend_from_slice(bytes);
        let end = payload.len();
        header.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }
    let mut header_bytes = serde_json::to_vec(&Value::Object(header)).unwrap();
    while !header_bytes.len().is_multiple_of(8) {
        header_bytes.push(b' ');
    }
    let mut artifact = Vec::with_capacity(8 + header_bytes.len() + payload.len());
    artifact.extend_from_slice(&u64::try_from(header_bytes.len()).unwrap().to_le_bytes());
    artifact.extend_from_slice(&header_bytes);
    artifact.extend_from_slice(&payload);
    std::fs::write(path, artifact).unwrap();
}

fn quantize_bf16_matrix_for_mlx(
    rows: usize,
    cols: usize,
    bits: u8,
    bf16_vals: &[u16],
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let groups = cols / 64;
    let q_max = if bits == 4 { 15.0_f32 } else { 255.0_f32 };
    let mut packed_u32 = Vec::with_capacity(rows * (cols / usize::from(32 / bits)));
    let mut scales_u16 = Vec::with_capacity(rows * groups);
    let mut biases_u16 = Vec::with_capacity(rows * groups);
    for r in 0..rows {
        for g in 0..groups {
            let base = r * cols + g * 64;
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for i in 0..64 {
                let v = f32_from_bf16(bf16_vals[base + i]);
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let span = hi - lo;
            let s_bits = f32_to_bf16_rne(if span == 0.0 { 0.0 } else { span / q_max });
            let b_bits = f32_to_bf16_rne(lo);
            scales_u16.push(s_bits);
            biases_u16.push(b_bits);
            let s = f32_from_bf16(s_bits);
            let b = f32_from_bf16(b_bits);
            let inv_s = if s == 0.0 { 0.0 } else { 1.0 / s };
            if bits == 4 {
                for w in 0..8 {
                    let mut word = 0_u32;
                    for nib in 0..8 {
                        let v = f32_from_bf16(bf16_vals[base + w * 8 + nib]);
                        #[expect(
                            clippy::cast_possible_truncation,
                            clippy::cast_sign_loss,
                            reason = "clamped to [0, 15]"
                        )]
                        let q = ((v - b) * inv_s).round().clamp(0.0, 15.0) as u32;
                        word |= q << (nib * 4);
                    }
                    packed_u32.push(word);
                }
            } else {
                for w in 0..16 {
                    let mut word = 0_u32;
                    for byte in 0..4 {
                        let v = f32_from_bf16(bf16_vals[base + w * 4 + byte]);
                        #[expect(
                            clippy::cast_possible_truncation,
                            clippy::cast_sign_loss,
                            reason = "clamped to [0, 255]"
                        )]
                        let q = ((v - b) * inv_s).round().clamp(0.0, 255.0) as u32;
                        word |= q << (byte * 8);
                    }
                    packed_u32.push(word);
                }
            }
        }
    }
    let packed_bytes = packed_u32.iter().flat_map(|w| w.to_le_bytes()).collect();
    let scale_bytes = scales_u16.iter().flat_map(|w| w.to_le_bytes()).collect();
    let bias_bytes = biases_u16.iter().flat_map(|w| w.to_le_bytes()).collect();
    (packed_bytes, scale_bytes, bias_bytes)
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "builds synthetic 64-aligned BF16 and MLX pre-quantized Q4/Q8 packages and verifies parity and rejections"
)]
fn synthetic_prequantized_mlx_q4_and_q8_match_online_quant_and_reject_mismatches() {
    let mut base_config = weights::tiny();
    let t = &mut base_config["text_config"];
    for (key, val) in [
        ("vocab_size", 32),
        ("vocab_size_per_layer_input", 32),
        ("hidden_size", 64),
        ("hidden_size_per_layer_input", 64),
        ("intermediate_size", 128),
        ("head_dim", 64),
        ("global_head_dim", 64),
    ] {
        t[key] = serde_json::json!(val);
    }

    // Build deterministic BF16 parameters for all specs
    let parsed_cfg = engine_gemma4::Gemma4Config::from_json(&base_config).unwrap();
    let mut specs_and_vals: Vec<(String, Vec<usize>, Vec<u16>)> = Vec::new();
    let add = |list: &mut Vec<(String, Vec<usize>, Vec<u16>)>, suffix: &str, shape: Vec<usize>| {
        let count: usize = shape.iter().product();
        let seed = suffix.bytes().fold(17_usize, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(usize::from(b))
        });
        let vals: Vec<u16> = if shape.len() == 1 && count == 1 {
            vec![f32_to_bf16_rne(0.95)]
        } else if shape.len() == 1 {
            (0..count)
                .map(|i| {
                    #[expect(clippy::cast_precision_loss, reason = "small fixture index")]
                    let v = 0.8 + ((seed + i * 7) % 19) as f32 * 0.02;
                    f32_to_bf16_rne(v)
                })
                .collect()
        } else {
            (0..count)
                .map(|i| {
                    #[expect(clippy::cast_precision_loss, reason = "small fixture index")]
                    let v = ((seed + i * 13) % 89) as f32 * 0.01 - 0.44;
                    f32_to_bf16_rne(v)
                })
                .collect()
        };
        list.push((suffix.to_owned(), shape, vals));
    };

    let h = parsed_cfg.hidden_size();
    let ple = 64_usize;
    let packed_ple = parsed_cfg.layers().len() * ple;
    add(&mut specs_and_vals, "embed_tokens.weight", vec![32, h]);
    add(
        &mut specs_and_vals,
        "embed_tokens_per_layer.weight",
        vec![32, packed_ple],
    );
    add(
        &mut specs_and_vals,
        "per_layer_model_projection.weight",
        vec![packed_ple, h],
    );
    add(
        &mut specs_and_vals,
        "per_layer_projection_norm.weight",
        vec![ple],
    );
    add(&mut specs_and_vals, "norm.weight", vec![h]);
    for layer in 0..4 {
        let base = format!("layers.{layer}");
        add(
            &mut specs_and_vals,
            &format!("{base}.self_attn.q_proj.weight"),
            vec![128, h],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.self_attn.k_proj.weight"),
            vec![64, h],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.self_attn.v_proj.weight"),
            vec![64, h],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.self_attn.o_proj.weight"),
            vec![h, 128],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.self_attn.q_norm.weight"),
            vec![64],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.self_attn.k_norm.weight"),
            vec![64],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.input_layernorm.weight"),
            vec![h],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.post_attention_layernorm.weight"),
            vec![h],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.pre_feedforward_layernorm.weight"),
            vec![h],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.post_feedforward_layernorm.weight"),
            vec![h],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.mlp.gate_proj.weight"),
            vec![128, h],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.mlp.up_proj.weight"),
            vec![128, h],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.mlp.down_proj.weight"),
            vec![h, 128],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.per_layer_input_gate.weight"),
            vec![ple, h],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.per_layer_projection.weight"),
            vec![h, ple],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.post_per_layer_input_norm.weight"),
            vec![h],
        );
        add(
            &mut specs_and_vals,
            &format!("{base}.layer_scalar"),
            vec![1],
        );
    }

    // 1. Write standard BF16 HF package
    let bf16_dir = weights::Dir::new();
    std::fs::write(
        bf16_dir.0.join("config.json"),
        serde_json::to_vec(&base_config).unwrap(),
    )
    .unwrap();
    let bf16_entries: Vec<(String, Vec<usize>, Vec<u8>)> = specs_and_vals
        .iter()
        .map(|(suffix, shape, vals)| {
            (
                format!("model.language_model.{suffix}"),
                shape.clone(),
                vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
        })
        .collect();
    let bf16_refs: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = bf16_entries
        .iter()
        .map(|(n, s, b)| (n.as_str(), "BF16", s.clone(), b.clone()))
        .collect();
    write_safetensors_raw(&bf16_dir.0.join("model.safetensors"), &bf16_refs);
    let bf16_pkg = LocalModelPackage::open(&bf16_dir.0).unwrap();
    let bf16_source = Gemma4Source::open(&bf16_pkg).unwrap();
    assert_eq!(bf16_source.encoding(), Gemma4SourceEncoding::Bf16);

    let prompt = vec![2_u32, 5, 9, 13, 17, 21];

    // 2. For both Q4 and Q8, write MLX-format pre-quantized package (with index + single shard +
    //    language_model.model.* prefix + inactive shared-tail quantized tensors + vision exclusion)
    for (bits, quant_enum, expected_enc, wrong_quant) in [
        (
            4_u8,
            Gemma4MetalQuant::Q4,
            Gemma4SourceEncoding::AffineQ4G64,
            Gemma4MetalQuant::Q8,
        ),
        (
            8_u8,
            Gemma4MetalQuant::Q8,
            Gemma4SourceEncoding::AffineQ8G64,
            Gemma4MetalQuant::Q4,
        ),
    ] {
        let mlx_dir = weights::Dir::new();
        let mut mlx_cfg = base_config.clone();
        let q_obj = serde_json::json!({"group_size": 64, "bits": bits, "mode": "affine"});
        mlx_cfg["quantization"] = q_obj.clone();
        mlx_cfg["quantization_config"] = q_obj;
        std::fs::write(
            mlx_dir.0.join("config.json"),
            serde_json::to_vec(&mlx_cfg).unwrap(),
        )
        .unwrap();

        let mut owned_entries: Vec<(String, String, Vec<usize>, Vec<u8>)> = Vec::new();
        for (suffix, shape, vals) in &specs_and_vals {
            let full_name = format!("language_model.model.{suffix}");
            if shape.len() == 2 && shape[1].is_multiple_of(64) && full_name.ends_with(".weight") {
                let (packed, scales, biases) =
                    quantize_bf16_matrix_for_mlx(shape[0], shape[1], bits, vals);
                let stem = full_name.strip_suffix(".weight").unwrap();
                let pack_factor = usize::from(32 / bits);
                owned_entries.push((
                    full_name.clone(),
                    "U32".to_owned(),
                    vec![shape[0], shape[1] / pack_factor],
                    packed,
                ));
                owned_entries.push((
                    format!("{stem}.scales"),
                    "BF16".to_owned(),
                    vec![shape[0], shape[1] / 64],
                    scales,
                ));
                owned_entries.push((
                    format!("{stem}.biases"),
                    "BF16".to_owned(),
                    vec![shape[0], shape[1] / 64],
                    biases,
                ));
            } else {
                owned_entries.push((
                    full_name,
                    "BF16".to_owned(),
                    shape.clone(),
                    vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ));
            }
        }
        // Add an excluded MLX vision tensor to verify non-text exclusion
        owned_entries.push((
            "vision_tower.dummy.weight".to_owned(),
            "BF16".to_owned(),
            vec![2],
            vec![0_u8; 4],
        ));

        let mut weight_map = serde_json::Map::new();
        for (name, _, _, _) in &owned_entries {
            weight_map.insert(name.clone(), Value::String("model.safetensors".to_owned()));
        }
        std::fs::write(
            mlx_dir.0.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();
        let entry_refs: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = owned_entries
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), d.as_str(), s.clone(), b.clone()))
            .collect();
        write_safetensors_raw(&mlx_dir.0.join("model.safetensors"), &entry_refs);

        let mlx_pkg = LocalModelPackage::open(&mlx_dir.0).unwrap();
        let mlx_source = Gemma4Source::open(&mlx_pkg).unwrap();
        assert_eq!(mlx_source.encoding(), expected_enc);

        // BF16-only loaders reject pre-quantized sources cleanly
        assert!(Gemma4CpuWeights::load(&mlx_source).is_err());
        let metal_exec = engine_metal::MetalBf16Executor::new().unwrap();
        assert!(Gemma4MetalWeights::load(&mlx_source, &metal_exec).is_err());

        // Mismatched quant policy rejects cleanly
        for bad_q in [Gemma4MetalQuant::Bf16, wrong_quant] {
            let bad_cfg = Gemma4MetalModelWideConfig {
                max_sequences: 4,
                max_positions: 32,
                max_batch_tokens: 32,
                num_kv_blocks: 16,
                quant: bad_q,
            };
            assert!(
                Gemma4MetalModelWide::load(&mlx_source, bad_cfg).is_err(),
                "must reject {bad_q:?} on {expected_enc:?} package"
            );
        }

        // Native pre-quantized load matches online-quantized BF16 load BIT-FOR-BIT!
        let resolved = Gemma4MetalQuant::resolve_for_source(None, mlx_source.encoding()).unwrap();
        assert_eq!(resolved, quant_enum);
        let cfg = Gemma4MetalModelWideConfig {
            max_sequences: 4,
            max_positions: 32,
            max_batch_tokens: 32,
            num_kv_blocks: 16,
            quant: resolved,
        };
        let mut online_model = Gemma4MetalModelWide::load(&bf16_source, cfg).unwrap();
        let mut preq_model = Gemma4MetalModelWide::load(&mlx_source, cfg).unwrap();
        assert_eq!(preq_model.weights_bytes(), online_model.weights_bytes());

        let s_on = online_model.allocate_slot().unwrap();
        let s_pre = preq_model.allocate_slot().unwrap();
        let l_on = online_model
            .prefill_slot(s_on, &prompt, true)
            .unwrap()
            .unwrap();
        let l_pre = preq_model
            .prefill_slot(s_pre, &prompt, true)
            .unwrap()
            .unwrap();
        assert_eq!(
            l_pre, l_on,
            "{quant_enum:?} prefill logits must be bit-identical"
        );

        let next_tok = argmax(&l_on).unwrap();
        let d_on = online_model.decode_slot(s_on, next_tok).unwrap();
        let d_pre = preq_model.decode_slot(s_pre, next_tok).unwrap();
        assert_eq!(
            d_pre, d_on,
            "{quant_enum:?} decode logits must be bit-identical"
        );
    }

    // 3. Reject unsupported quantization configs (group_size=128, bits=3, quant_method=awq, disagree)
    for bad_q in [
        serde_json::json!({"group_size": 128, "bits": 4, "mode": "affine"}),
        serde_json::json!({"group_size": 64, "bits": 3, "mode": "affine"}),
        serde_json::json!({"group_size": 64, "bits": 4, "mode": "mxfp4"}),
        serde_json::json!({"quant_method": "awq", "group_size": 64, "bits": 4}),
    ] {
        let dir = weights::Dir::new();
        let mut cfg = base_config.clone();
        cfg["quantization"] = bad_q;
        std::fs::write(dir.0.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
        write_safetensors_raw(&dir.0.join("model.safetensors"), &bf16_refs);
        let pkg = LocalModelPackage::open(&dir.0).unwrap();
        assert!(Gemma4Source::open(&pkg).is_err());
    }
}

#[test]
#[ignore = "loads a real pre-quantized MLX affine group-64 Gemma 4 E4B package from RIBN_GEMMA_PREQUANT_PACKAGE; use --release -- --nocapture"]
#[expect(
    clippy::too_many_lines,
    reason = "real-model pre-quantized oracle, load-time, prefix-cache, and rejection gate"
)]
fn e4b_prequantized_mlx_oracle_and_prefix_gate() {
    let pkg_path = PathBuf::from(
        std::env::var_os("RIBN_GEMMA_PREQUANT_PACKAGE").expect("RIBN_GEMMA_PREQUANT_PACKAGE"),
    );
    let ref_path =
        PathBuf::from(std::env::var_os("RIBN_GEMMA_REFERENCE").expect("RIBN_GEMMA_REFERENCE"));
    let tensors_path = ref_path.with_file_name("tensors.safetensors");

    let t_open = Instant::now();
    let package = LocalModelPackage::open(&pkg_path).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    let open_ms = t_open.elapsed().as_secs_f64() * 1e3;
    assert_ne!(
        source.encoding(),
        Gemma4SourceEncoding::Bf16,
        "RIBN_GEMMA_PREQUANT_PACKAGE must be pre-quantized"
    );
    let quant = Gemma4MetalQuant::resolve_for_source(None, source.encoding()).unwrap();
    assert!(
        Gemma4MetalQuant::resolve_for_source(Some(Gemma4MetalQuant::Bf16), source.encoding())
            .is_err()
    );

    let t_load = Instant::now();
    let limits = Gemma4MetalModelWideConfig {
        max_sequences: 16,
        max_positions: 2048,
        max_batch_tokens: 512,
        num_kv_blocks: 512,
        quant,
    };
    let mut model = Gemma4MetalModelWide::load(&source, limits).unwrap();
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    #[expect(clippy::cast_precision_loss, reason = "display GiB")]
    let weight_gib = model.weights_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "\n================ PRE-QUANTIZED E4B ({:?}) ================\nopen_ms={open_ms:.1} load_ms={load_ms:.1} total_ms={:.1} weight_bytes={} ({weight_gib:.2} GiB)",
        source.encoding(),
        open_ms + load_ms,
        model.weights_bytes(),
    );

    let reference: Value = serde_json::from_slice(&std::fs::read(&ref_path).unwrap()).unwrap();
    let ref_tensors = SafeTensorArtifact::open(&tensors_path).unwrap();
    let slot = model.allocate_slot().unwrap();
    let mut total_rows = 0_usize;
    let mut matched_winners = 0_usize;
    let mut worst_kl = 0.0_f64;
    let mut worst_rel_l2 = 0.0_f64;
    let mut window_513_cold: Option<(Vec<u32>, Vec<u16>, f64)> = None;

    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let prompt: Vec<u32> = case["prompt_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
            .collect();
        let steps = case["steps"].as_array().unwrap();
        model.reset_slot(slot).unwrap();

        let t0 = Instant::now();
        let mut logits = model.prefill_slot(slot, &prompt, true).unwrap().unwrap();
        let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
        if name == "window_513" {
            window_513_cold = Some((prompt.clone(), logits.clone(), prefill_ms));
        }
        for (step_idx, step) in steps.iter().enumerate() {
            if step_idx > 0 {
                let input_tok = u32::try_from(step["input_ids"][0].as_u64().unwrap()).unwrap();
                logits = model.decode_slot(slot, input_tok).unwrap();
            }
            let expected = u32::try_from(step["chosen"].as_u64().unwrap()).unwrap();
            let actual = argmax(&logits).unwrap();
            total_rows += 1;
            if actual == expected {
                matched_winners += 1;
            }
            let tensor_key = format!("{name}.{step_idx}.logits");
            let ref_logits = read_bf16_tensor(&ref_tensors, &tensor_key);
            let (_rmse, rel_l2, kl, max_abs) = logit_metrics(&logits, &ref_logits);
            worst_kl = worst_kl.max(kl);
            worst_rel_l2 = worst_rel_l2.max(rel_l2);
            println!(
                "PREQUANT-{quant:?} {name}[{step_idx}]: actual={actual} expected={expected} rel_l2={rel_l2:.5} kl={kl:.6} max_abs={max_abs:.3}"
            );
        }
    }
    println!(
        "PREQUANT-{quant:?} ORACLE SUMMARY: matched={matched_winners}/{total_rows}, worst_rel_l2={worst_rel_l2:.5}, worst_kl={worst_kl:.6}"
    );
    if quant == Gemma4MetalQuant::Q8 {
        assert_eq!(matched_winners, total_rows);
        assert!(worst_kl <= 0.02, "worst_kl {worst_kl} exceeded 0.02");
    } else {
        assert!(
            matched_winners >= 10,
            "matched only {matched_winners}/{total_rows} BF16 oracle winners"
        );
    }

    if let Some((prompt_513, cold_logits, cold_ms)) = window_513_cold {
        let (cached_slot, matched) = model.allocate_slot_with_prefix(&prompt_513).unwrap();
        assert_eq!(matched, 512);
        let t_hit = Instant::now();
        let cached_logits = model
            .prefill_slot(cached_slot, &prompt_513[matched..], true)
            .unwrap()
            .unwrap();
        let cached_ms = t_hit.elapsed().as_secs_f64() * 1e3;
        assert_eq!(cached_logits, cold_logits);
        println!(
            "PREQUANT-{quant:?} PREFIX CACHE (len=513, hit=512): cold={cold_ms:.2} ms -> cached={cached_ms:.2} ms ({:.1}x)",
            cold_ms / cached_ms
        );
        model.free_slot(cached_slot).unwrap();
    }
}
