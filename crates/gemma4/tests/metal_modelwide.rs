//! Synthetic and real-artifact qualification tests for `Gemma4MetalModelWide`.
#![cfg(all(target_os = "macos", feature = "metal"))]

#[path = "common/weights.rs"]
mod weights;

use engine_gemma4::{
    Gemma4CpuDecoder, Gemma4CpuWeights, Gemma4MetalModelWide, Gemma4MetalModelWideConfig,
    Gemma4MetalModelWideExecutor, Gemma4Source, PackedStepItem, argmax,
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
        max_batch_tokens: 256,
    };
    let mut model = Gemma4MetalModelWide::load(&source, limits).unwrap();
    println!("load_ms={:.1}", t_load.elapsed().as_secs_f64() * 1e3);

    let slot = model.allocate_slot().unwrap();
    let mut total_rows = 0_usize;
    let mut matched_winners = 0_usize;
    let mut worst_kl = 0.0_f64;
    let mut worst_rel_l2 = 0.0_f64;

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
