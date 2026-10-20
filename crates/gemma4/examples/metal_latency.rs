//! Break down one Gemma Metal request into loading, upload, prefill, decode and RSS.
//!
//! Uses the archived chat prompt so the work matches the integration test. This
//! measures the correctness-first orchestration, not a competitive engine path.

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() {
    use engine_gemma4::{Gemma4Config, Gemma4MetalDecoder, Gemma4Source, argmax};
    use ribn_hf::LocalModelPackage;
    use serde_json::Value;
    use std::process::Command;
    use std::time::Instant;

    fn rss_kib() -> u64 {
        let output = Command::new("/bin/ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .expect("sample this process's RSS");
        assert!(output.status.success(), "ps RSS sample failed");
        String::from_utf8(output.stdout)
            .expect("RSS is UTF-8")
            .trim()
            .parse()
            .expect("RSS is a KiB integer")
    }

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
    assert_eq!(source.config(), &config);
    let loaded = load.elapsed();
    let preflight_rss = rss_kib();

    let prepare = Instant::now();
    let mut decoder = Gemma4MetalDecoder::load(&source).expect("Metal decoder");
    let prepared = prepare.elapsed();
    let prepared_rss = rss_kib();

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
    let executed_rss = rss_kib();
    println!("prompt_tokens={}", prompt.len());
    println!("preflight_rss_kib={preflight_rss}");
    println!("prepared_rss_kib={prepared_rss}");
    println!("executed_rss_kib={executed_rss}");
    println!("source_preflight_ms={:.1}", loaded.as_secs_f64() * 1e3);
    println!("prepare_ms={:.1}", prepared.as_secs_f64() * 1e3);
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
