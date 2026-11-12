//! Exact-row working sets, real continuation progress and late load retirement.
use super::{
    Admission, BytePool, CudaContext, Gemma4CudaModelWide, Gemma4Source, LocalModelPackage, cases,
    environment, finite, identities, issued_ids,
};
use engine_gemma4::{AttentionKind, Gemma4Config};

const CAPACITY: usize = 4096;
const SCRATCH: usize = 1 << 20;

fn workspace_bytes(config: &Gemma4Config, rows: usize) -> u64 {
    let h = config.hidden_size();
    let p = config.ple_dim();
    let shared = 2 * h + 3 * config.packed_ple_dim() + p;
    let geometry: usize = [AttentionKind::Sliding, AttentionKind::Full]
        .into_iter()
        .map(|kind| {
            let width = config
                .layers()
                .iter()
                .find(|layer| layer.kind == kind)
                .unwrap()
                .head_dim;
            2 * width
                + 4 * config.num_attention_heads() * width
                + 5 * config.num_key_value_heads() * width
                + 8 * h
                + 4 * config.intermediate_size()
                + 3 * p
        })
        .sum();
    // Pinned columns are multiples of 256; allocation alignment adds no bytes.
    u64::try_from(2 * (rows * (shared + geometry) + 2 * h + config.vocab_size())).unwrap()
}

#[test]
#[expect(
    clippy::assertions_on_constants,
    reason = "compile in debug but require release device execution"
)]
#[ignore = "real 15 GB package, every exact row workspace and late load refusal; serialized release CUDA gate"]
fn all_prefill_widths_preserve_charges_progress_and_late_construction_retirement() {
    assert!(!cfg!(debug_assertions), "device gate requires --release");
    let _device = identities::DEVICE.lock().unwrap();
    let package = LocalModelPackage::open(environment("RIBN_GEMMA_PACKAGE")).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    let context = CudaContext::new(0).unwrap();
    assert!(Gemma4CudaModelWide::matches_qualification_scope(
        source.config(),
        &context.name().unwrap(),
        CAPACITY,
    ));
    let pool = BytePool::new(20 << 30).shared();
    let mut model = Gemma4CudaModelWide::load(&source, &context, &pool, CAPACITY, SCRATCH).unwrap();
    let baseline = pool.granted();
    let added: u64 = (2..=16)
        .filter(|rows| ![8, 16].contains(rows))
        .map(|rows| workspace_bytes(source.config(), rows))
        .sum();
    // Independently archived fixed-width charge plus the new buffer layouts.
    // This protects actual allocation, not just a planner that could fall back.
    assert_eq!(baseline, 14_949_422_368 + added);
    assert_eq!(added, 50_425_344);
    let sequence = issued_ids::<1>()[0];
    let key_bytes = u64::try_from(
        CAPACITY * source.config().num_key_value_heads() * source.config().layers()[0].head_dim * 2,
    )
    .unwrap();
    let pressure = pool.reserve(pool.available() - key_bytes).unwrap();
    let pressured = pool.granted();
    assert!(matches!(
        model.admit(sequence, CAPACITY).unwrap(),
        Admission::Rejected(_)
    ));
    assert_eq!(model.position(sequence), None);
    assert_eq!(
        pool.granted(),
        pressured,
        "first key retires when next value refuses"
    );
    drop(pressure);
    assert_eq!(pool.granted(), baseline);
    assert!(matches!(
        model.admit(sequence, CAPACITY).unwrap(),
        Admission::Ready
    ));
    let history = cases()
        .into_iter()
        .find(|case| case.name == "window_513")
        .unwrap()
        .prompt;
    model.prefill(sequence, &history[..16], false).unwrap();
    let with_cache = pool.granted();
    let mut position = 16;
    for rows in 2..=16 {
        let logits = model
            .prefill(sequence, &history[position..position + rows], true)
            .unwrap()
            .unwrap();
        position += rows;
        finite(&logits, source.config().vocab_size());
        assert_eq!(model.position(sequence), Some(position));
        assert_eq!(pool.granted(), with_cache);
    }
    assert_eq!(position, 151);
    model.release(sequence).unwrap();
    assert_eq!(pool.granted(), baseline);
    model.close().unwrap();
    model.close().unwrap();
    drop(model);
    assert_eq!(pool.granted(), 0);

    let limited = BytePool::new(baseline - 1).shared();
    let mut failure =
        match Gemma4CudaModelWide::load(&source, &context, &limited, CAPACITY, SCRATCH) {
            Ok(mut unexpected) => {
                unexpected.close().unwrap();
                panic!("late load must refuse");
            }
            Err(failure) => failure,
        };
    // Last allocation: separate packed-M2 logits, after all prefill workspaces.
    assert_eq!(
        baseline - limited.granted(),
        u64::try_from(4 * source.config().vocab_size()).unwrap()
    );
    assert!(failure.retirement_pending());
    failure.retire().unwrap();
    failure.retire().unwrap();
    assert_eq!(limited.granted(), 0);
}
