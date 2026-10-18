use engine_gemma4::{AttentionKind, Gemma4Config};
use serde_json::Value;

fn config() -> Gemma4Config {
    Gemma4Config::from_json(
        &serde_json::from_str(include_str!(
            "../../../benchmarks/gemma4-reference/config.json"
        ))
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn persistent_suffix_and_donor_spans_are_distinct_across_window_and_chunk_boundaries() {
    let c = config();
    for (prefix, chunk, attention, retained) in [
        (0, 511, 0..511, 0..511),
        (0, 512, 0..512, 1..512),
        (0, 513, 0..513, 2..513),
        (511, 1, 0..512, 1..512),
        (512, 1, 1..513, 2..513),
        (511, 513, 0..1024, 513..1024),
    ] {
        let step = c.kv_step(prefix, chunk, 4096).unwrap();
        assert_eq!(step.input(), prefix..prefix + chunk);
        assert_eq!(step.owners().len(), 24);
        let mut local = 0;
        for owner in step.owners() {
            let full = owner.layer() % 6 == 5;
            if full {
                assert_eq!(owner.kind(), AttentionKind::Full);
                assert_eq!(owner.attention().range(), 0..prefix + chunk);
                assert_eq!(owner.after().range(), 0..prefix + chunk);
                assert_eq!(owner.bytes_per_token(), 4096);
            } else {
                local += 1;
                assert_eq!(owner.kind(), AttentionKind::Sliding);
                assert_eq!(owner.attention().range(), attention);
                assert_eq!(owner.after().range(), retained);
                assert_eq!(owner.bytes_per_token(), 2048);
            }
            assert_eq!(
                owner.last_consumer(),
                match owner.layer() {
                    22 => 40,
                    23 => 41,
                    i => i,
                }
            );
            if prefix > 0 {
                assert!(owner.visible(prefix - 1).is_err());
            }
            assert!(owner.visible(prefix + chunk).is_err());
            for query in prefix..prefix + chunk {
                let visible = owner.visible(query).unwrap();
                assert_eq!(
                    visible,
                    if full {
                        0..query + 1
                    } else {
                        query.saturating_sub(511)..query + 1
                    }
                );
                assert!(owner.attention().range().start <= visible.start);
                assert!(owner.attention().range().end >= visible.end);
            }
        }
        assert_eq!(local, 20);
    }
}

#[test]
fn logical_payload_counts_only_owner_kv_and_is_not_a_peak_allocation_estimate() {
    let c = config();
    let prefill = c.kv_step(0, 513, 4096).unwrap();
    assert_eq!(prefill.retained_payload_bytes(), 29_335_552);
    assert_eq!(prefill.donor_payload_bytes(), 3_151_872);
    assert_eq!(
        c.kv_step(0, 4096, 4096).unwrap().retained_payload_bytes(),
        88_039_424
    );
    let split = c.kv_step(511, 513, 4096).unwrap();
    let whole = c.kv_step(0, 1024, 4096).unwrap();
    for (a, b) in split.owners().iter().zip(whole.owners()) {
        assert_eq!(
            a.after(),
            b.after(),
            "same final coverage, not a claim of bit-identical arithmetic"
        );
    }
    assert_eq!(split.owners()[22].before().range(), 0..511);
    assert_eq!(split.owners()[22].attention().range(), 0..1024);
    assert_eq!(split.owners()[22].after().range(), 513..1024);
}

#[test]
fn invalid_progress_capacity_and_payload_overflow_are_rejected() {
    let c = config();
    for (prefix, tokens, capacity) in [
        (0, 0, 4096),
        (0, 1, 0),
        (4096, 1, 4096),
        (0, 1, 131_073),
        (usize::MAX, 1, 4096),
    ] {
        assert!(c.kv_step(prefix, tokens, capacity).is_err());
    }
    let mut value: Value = serde_json::from_str(include_str!(
        "../../../benchmarks/gemma4-reference/config.json"
    ))
    .unwrap();
    value["text_config"]["max_position_embeddings"] = serde_json::json!(usize::MAX);
    let large = Gemma4Config::from_json(&value).unwrap();
    assert!(
        large.kv_step(0, usize::MAX, usize::MAX).is_err(),
        "logical byte sum must not wrap"
    );
    value["text_config"]["sliding_window"] = serde_json::json!(1);
    let singleton = Gemma4Config::from_json(&value)
        .unwrap()
        .kv_step(5, 3, 100)
        .unwrap();
    assert_eq!(singleton.owners()[0].before().range(), 5..5);
    assert_eq!(singleton.owners()[0].attention().range(), 5..8);
    assert_eq!(singleton.owners()[0].after().range(), 8..8);
    assert_eq!(singleton.owners()[0].visible(6).unwrap(), 6..7);
}
