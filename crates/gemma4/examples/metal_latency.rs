//! Break down one Gemma Metal request into loading, upload, prefill and decode.
//!
//! Uses the archived chat prompt so the work matches the integration test. This
//! measures the correctness-first orchestration, not a competitive engine path.

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    use engine_gemma4::{Gemma4Config, Gemma4CpuWeights, Gemma4MetalDecoder, Gemma4Source, argmax};
    use ribn_hf::LocalModelPackage;
    use serde_json::Value;
    use std::time::Instant;

    let package = LocalModelPackage::open(
        std::env::var_os("RIBN_GEMMA_PACKAGE").expect("RIBN_GEMMA_PACKAGE"),
    )
    .unwrap();
    let reference: Value = serde_json::from_slice(
        &std::fs::read(std::env::var_os("RIBN_GEMMA_REFERENCE").expect("RIBN_GEMMA_REFERENCE"))
            .unwrap(),
    )
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
    let config = Gemma4Config::from_json(
        &serde_json::from_slice(&std::fs::read(package.root().join("config.json")).unwrap())
            .unwrap(),
    )
    .unwrap();

    let load = Instant::now();
    let source = Gemma4Source::open(&package).unwrap();
    let host = Gemma4CpuWeights::load(&source).unwrap();
    assert_eq!(host.config(), &config);
    let loaded = load.elapsed();

    let upload = Instant::now();
    let mut decoder = Gemma4MetalDecoder::load(host).expect("Metal decoder");
    let uploaded = upload.elapsed();

    let prefill = Instant::now();
    let mut logits = decoder.prefill(&prompt).expect("prefill");
    let prefilled = prefill.elapsed();

    let mut steps = Vec::new();
    for _ in 0..4 {
        let token = argmax(&logits).expect("logits");
        let decode = Instant::now();
        logits = decoder.decode(token).expect("decode");
        steps.push(decode.elapsed());
    }
    println!("prompt_tokens={}", prompt.len());
    println!("host_load_ms={:.1}", loaded.as_secs_f64() * 1e3);
    println!(
        "upload_including_load_ms={:.1}",
        uploaded.as_secs_f64() * 1e3
    );
    println!("prefill_ms={:.1}", prefilled.as_secs_f64() * 1e3);
    for (index, step) in steps.iter().enumerate() {
        println!("decode_{index}_ms={:.1}", step.as_secs_f64() * 1e3);
    }
    println!(
        "decode_mean_ms={:.1}",
        steps
            .iter()
            .map(std::time::Duration::as_secs_f64)
            .sum::<f64>()
            / f64::from(u32::try_from(steps.len()).unwrap())
            * 1e3
    );
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn main() {
    eprintln!("metal_latency requires macOS and --features metal");
}
