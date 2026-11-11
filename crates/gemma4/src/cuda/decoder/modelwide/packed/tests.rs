//! Release-only gates over the real package, opaque stream and physical owner.
//! No serial-KV equality is assumed: M2 GEMM may reorder arithmetic upstream.
use super::{
    AttentionGeometry, AttentionKind, AttentionSpan, CudaBf16Commands, CudaBf16Error,
    CudaBf16Matrix, PhysicalSequence, layer_forward,
};
use crate::{
    Gemma4Source, ParameterId,
    cuda::{
        Gemma4CudaWeights,
        decoder::{
            DeviceCache, GEMM_WORKSPACE, WEIGHT_STAGING,
            modelwide::{Gemma4CudaModelWide, Phase, preparation::Prepared},
        },
    },
    cuda_test_support::{DEVICE, issued_ids},
};
use cudarc::driver::CudaContext;
use engine_nvidia::CudaBf16Stream;
use ribn::Admission;
use ribn_foundation::BytePool;
use ribn_hf::LocalModelPackage;
use serde_json::Value;
use std::{ops::Range, path::PathBuf};

struct History {
    prompt: Vec<u32>,
    continuation: Vec<u32>,
}
fn environment(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name).unwrap_or_else(|| panic!("{name} must name pinned evidence")),
    )
}
#[expect(
    clippy::assertions_on_constants,
    reason = "compile ignored gates in debug, but refuse accidental debug device execution"
)]
fn source() -> Gemma4Source {
    assert!(!cfg!(debug_assertions), "device gates require --release");
    let package = LocalModelPackage::open(environment("RIBN_GEMMA_PACKAGE")).unwrap();
    let source = Gemma4Source::open(&package).unwrap();
    assert_eq!(source.config().layers().len(), 42, "pinned E4B stack");
    assert_eq!(source.config().sliding_window(), 512);
    source
}
fn histories() -> [History; 2] {
    let document: Value =
        serde_json::from_slice(&std::fs::read(environment("RIBN_GEMMA_REFERENCE")).unwrap())
            .unwrap();
    assert_eq!(document["complete"], true);
    assert_eq!(document["precision"], "bf16");
    let cases = document["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    let ids = |values: &Value| {
        values
            .as_array()
            .unwrap()
            .iter()
            .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
            .collect::<Vec<_>>()
    };
    assert!(
        cases
            .iter()
            .any(|case| ids(&case["prompt_ids"]).len() == 18)
    );
    [511, 513].map(|length| {
        let case = cases
            .iter()
            .find(|case| ids(&case["prompt_ids"]).len() == length)
            .expect("both pinned window-boundary histories");
        let continuation = case["steps"]
            .as_array()
            .unwrap()
            .iter()
            .skip(1)
            .map(|step| {
                let input = ids(&step["input_ids"]);
                assert_eq!(input.len(), 1, "teacher-forced common history");
                input[0]
            })
            .collect::<Vec<_>>();
        assert_eq!(continuation.len(), 3);
        History {
            prompt: ids(&case["prompt_ids"]),
            continuation,
        }
    })
}
fn scratch(source: &Gemma4Source, capacity: usize) -> usize {
    source.config().num_attention_heads() * capacity * 16
}
fn matrix(model: &Prepared, rows: usize, columns: usize) -> CudaBf16Matrix {
    model
        .stream
        .matrix(rows, columns, &model.pool, rows * columns * 2)
        .unwrap()
}
fn cache(model: &Prepared, rows: usize, columns: usize) -> DeviceCache {
    DeviceCache {
        keys: matrix(model, rows, columns),
        values: matrix(model, rows, columns),
    }
}
fn copy_cache(
    commands: &mut CudaBf16Commands,
    destination: &mut DeviceCache,
    source: &DeviceCache,
    rows: Range<usize>,
) {
    commands
        .copy_rows(&mut destination.keys, 0, &source.keys, rows.clone())
        .unwrap();
    commands
        .copy_rows(&mut destination.values, 0, &source.values, rows)
        .unwrap();
}
fn bytes(matrix: &mut CudaBf16Matrix) -> Vec<u8> {
    let (rows, columns) = matrix.shape();
    let bytes = matrix.read_chunk(0, rows * columns * 2).unwrap();
    assert!(
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|&pair| u16::from_le_bytes(pair) & 0x7f80 != 0x7f80),
        "real KV/attention captures must be finite"
    );
    bytes
}

struct OwnerCapture {
    layer: usize,
    produced: DeviceCache,
    before: [DeviceCache; 2],
    after: [DeviceCache; 2],
}
impl OwnerCapture {
    fn new(model: &Prepared, layer: usize, positions: [usize; 2]) -> Self {
        let width =
            model.host.config.layers()[layer].head_dim * model.host.config.num_key_value_heads();
        Self {
            layer,
            produced: cache(model, 2, width),
            before: positions.map(|position| cache(model, position, width)),
            after: positions.map(|position| cache(model, position + 1, width)),
        }
    }
    fn close(&mut self) {
        self.produced.close().unwrap();
        for snapshot in self.before.iter_mut().chain(&mut self.after) {
            snapshot.close().unwrap();
        }
    }
}
struct AttentionCapture {
    actual: CudaBf16Matrix,
    suffix: CudaBf16Matrix,
}
impl AttentionCapture {
    fn close(&mut self) {
        self.actual.close().unwrap();
        self.suffix.close().unwrap();
    }
}
struct WindowScratch {
    cache: DeviceCache,
    query: CudaBf16Matrix,
    output: CudaBf16Matrix,
}
impl WindowScratch {
    fn new(model: &Prepared) -> Self {
        let config = &model.host.config;
        let width = config
            .layers()
            .iter()
            .find(|geometry| geometry.kind == AttentionKind::Sliding)
            .unwrap()
            .head_dim;
        Self {
            cache: cache(
                model,
                config.sliding_window(),
                width * config.num_key_value_heads(),
            ),
            query: matrix(model, 1, width * config.num_attention_heads()),
            output: matrix(model, 1, width * config.num_attention_heads()),
        }
    }
    fn close(&mut self) {
        self.cache.close().unwrap();
        self.query.close().unwrap();
        self.output.close().unwrap();
    }
}

// Called only after the production helper has consumed the actual cache and
// generated buffers.attended. Re-lower its window as an explicit suffix, leaving
// window=None. The visible operands and ascending accumulation order are identical,
// so this is a bitwise span/routing gate, not a new numerical error allowance.
fn capture_window(
    model: &mut Prepared,
    members: &[&mut PhysicalSequence; 2],
    commands: &mut CudaBf16Commands,
    layer: usize,
    capture: &mut AttentionCapture,
    scratch: &mut WindowScratch,
) {
    let config = &model.host.config;
    let geometry = &config.layers()[layer];
    let buffers = if geometry.head_dim == model.packed.buffers.local.width {
        &model.packed.buffers.local
    } else {
        &model.packed.buffers.global
    };
    commands
        .copy_rows(&mut capture.actual, 0, &buffers.attended, 0..2)
        .unwrap();
    for (lane, state) in members.iter().enumerate() {
        let end = state.position + 1;
        let first = end.saturating_sub(config.sliding_window());
        // Both histories have at least one complete window throughout this gate.
        assert_eq!(end - first, config.sliding_window());
        copy_cache(
            commands,
            &mut scratch.cache,
            state.caches[geometry.kv_owner].as_ref().unwrap(),
            first..end,
        );
        commands
            .copy_rows(&mut scratch.query, 0, &buffers.query_rotary, lane..lane + 1)
            .unwrap();
        commands
            .attend_into(
                &mut model.executors.attention,
                &scratch.query,
                &scratch.cache.keys,
                &scratch.cache.values,
                AttentionGeometry {
                    heads: config.num_attention_heads(),
                    kv_heads: config.num_key_value_heads(),
                    width: geometry.head_dim,
                    window: None,
                },
                AttentionSpan {
                    query_start: state.position,
                    key_start: first,
                    key_sequence: end - first,
                },
                &mut scratch.output,
            )
            .unwrap();
        commands
            .copy_rows(&mut capture.suffix, lane, &scratch.output, 0..1)
            .unwrap();
    }
}

fn assert_append(
    capture: &mut OwnerCapture,
    states: &mut [&mut PhysicalSequence; 2],
    positions: [usize; 2],
) -> [bool; 2] {
    let mut distinguished = [false; 2];
    for (index, (kind, produced)) in [
        ("rotated K", bytes(&mut capture.produced.keys)),
        ("normalized V", bytes(&mut capture.produced.values)),
    ]
    .into_iter()
    .enumerate()
    {
        let row_bytes = produced.len() / 2;
        // A first-layer V can legitimately coincide when both input tokens do.
        // Require discriminating rows across the stack, not at every producer.
        distinguished[index] = produced[..row_bytes] != produced[row_bytes..];
        for (lane, state) in states.iter_mut().enumerate() {
            let (before, after, actual) = if kind == "rotated K" {
                (
                    &mut capture.before[lane].keys,
                    &mut capture.after[lane].keys,
                    &mut state.caches[capture.layer].as_mut().unwrap().keys,
                )
            } else {
                (
                    &mut capture.before[lane].values,
                    &mut capture.after[lane].values,
                    &mut state.caches[capture.layer].as_mut().unwrap().values,
                )
            };
            let prefix = bytes(before);
            let appended = bytes(after);
            assert_eq!(
                appended[..prefix.len()],
                prefix,
                "prefix changed: layer={} lane={lane} position={} {kind}",
                capture.layer,
                positions[lane]
            );
            assert_eq!(
                appended[prefix.len()..],
                produced[lane * row_bytes..(lane + 1) * row_bytes],
                "wrong producer row: layer={} lane={lane} {kind}",
                capture.layer
            );
            let end = positions[lane] + 1;
            assert_eq!(actual.shape(), (state.capacity, row_bytes / 2));
            assert_eq!(actual.ready(), end == state.capacity);
            // Production caches deliberately stage only one byte. The test-owned
            // full copies above verify every byte; these probes verify the actual
            // cache's initialized boundary without widening its staging envelope.
            assert_eq!(
                actual
                    .read_chunk(u64::try_from(appended.len() - 1).unwrap(), 1)
                    .unwrap(),
                appended[appended.len() - 1..]
            );
            assert!(matches!(
                actual.read_chunk(u64::try_from(appended.len()).unwrap(), 1),
                Err(CudaBf16Error::Invalid(
                    "matrix read exceeds initialized bytes"
                ))
            ));
        }
    }
    distinguished
}

#[test]
#[ignore = "real 15 GB E4B package; release-only serialized CUDA gate"]
#[expect(
    clippy::too_many_lines,
    reason = "one depth-one full-stack capture and acknowledgement lifecycle"
)]
fn own_produced_kv_preserves_prefixes_bounds_windows_and_reordered_members() {
    let _device = DEVICE.lock().unwrap();
    let source = source();
    let histories = histories();
    assert_ne!(histories[0].prompt, histories[1].prompt);
    let capacity = histories[1].prompt.len() + histories[1].continuation.len();
    let context = CudaContext::new(0).unwrap();
    let pool = BytePool::new(20_u64 << 30).shared();
    let mut model = Gemma4CudaModelWide::load(
        &source,
        &context,
        &pool,
        capacity,
        scratch(&source, capacity),
    )
    .unwrap();
    let ids = issued_ids::<2>();
    for (member, history) in histories.iter().enumerate() {
        let reach = history.prompt.len() + history.continuation.len();
        assert!(matches!(
            model.admit(ids[member], reach).unwrap(),
            Admission::Ready
        ));
        assert!(
            model
                .prefill(ids[member], &history.prompt, false)
                .unwrap()
                .is_none()
        );
    }
    let resident = pool.granted();
    let mut window = WindowScratch::new(&model.owner().model);
    for step in 0..3 {
        let order = if step % 2 == 0 { [1, 0] } else { [0, 1] };
        let routed = order.map(|member| ids[member]);
        let positions = routed.map(|id| model.position(id).unwrap());
        let tokens = order.map(|member| histories[member].continuation[step]);
        let owner = model.owner_mut();
        let mut owners = owner
            .model
            .host
            .config
            .layers()
            .iter()
            .enumerate()
            .filter(|(layer, geometry)| geometry.kv_owner == *layer)
            .map(|(layer, _)| OwnerCapture::new(&owner.model, layer, positions))
            .collect::<Vec<_>>();
        let mut attention = owner
            .model
            .host
            .config
            .layers()
            .iter()
            .map(|geometry| {
                (geometry.kind == AttentionKind::Sliding).then(|| {
                    let columns = owner.model.host.config.num_attention_heads() * geometry.head_dim;
                    AttentionCapture {
                        actual: matrix(&owner.model, 2, columns),
                        suffix: matrix(&owner.model, 2, columns),
                    }
                })
            })
            .collect::<Vec<_>>();
        owner.phase = Phase::Failed;
        let packed_positions = positions.map(|position| u32::try_from(position).unwrap());
        owner
            .model
            .packed
            .prepare(
                &owner.model.host,
                &mut owner.model.cuda,
                &mut owner.model.executors,
                &mut owner.model.factors,
                tokens,
                packed_positions,
            )
            .unwrap();
        // Production's 64/layer bound plus test copies and suffix attention.
        owner.commands = Some(CudaBf16Commands::new(&owner.model.stream, 42 * 96).unwrap());
        let commands = owner.commands.as_mut().unwrap();
        let [Some(first), Some(second)] = owner.states.get_disjoint_mut([&routed[0], &routed[1]])
        else {
            panic!("admitted independent histories");
        };
        let mut members = [first, second];
        for capture in &mut owners {
            for (lane, member) in members.iter().enumerate() {
                copy_cache(
                    commands,
                    &mut capture.before[lane],
                    member.caches[capture.layer].as_ref().unwrap(),
                    0..positions[lane],
                );
            }
        }
        for (layer, attention_capture) in attention.iter_mut().enumerate() {
            let prepared = &mut owner.model;
            let geometry = &prepared.host.config.layers()[layer];
            let (buffers, member_buffers) =
                if geometry.head_dim == prepared.packed.buffers.local.width {
                    (
                        &mut prepared.packed.buffers.local,
                        &mut prepared.packed.local,
                    )
                } else {
                    (
                        &mut prepared.packed.buffers.global,
                        &mut prepared.packed.global,
                    )
                };
            // The SAME private function as Packed::enqueue, through all 42 layers.
            // No waits, reads or host staging occur inside this dependent stack.
            layer_forward(
                &prepared.host.config,
                &prepared.host,
                &mut prepared.cuda,
                &mut prepared.executors,
                &mut members,
                commands,
                &mut prepared.packed.buffers.hidden,
                &mut prepared.packed.buffers.hidden_next,
                &mut prepared.packed.layers[layer],
                buffers,
                member_buffers,
                layer,
                geometry,
            )
            .unwrap();
            if let Some(capture) = owners.iter_mut().find(|capture| capture.layer == layer) {
                commands
                    .copy_rows(&mut capture.produced.keys, 0, &buffers.key_rotary, 0..2)
                    .unwrap();
                commands
                    .copy_rows(&mut capture.produced.values, 0, &buffers.value_norm, 0..2)
                    .unwrap();
                for (lane, member) in members.iter().enumerate() {
                    copy_cache(
                        commands,
                        &mut capture.after[lane],
                        member.caches[layer].as_ref().unwrap(),
                        0..positions[lane] + 1,
                    );
                }
            }
            if let Some(capture) = attention_capture {
                capture_window(prepared, &members, commands, layer, capture, &mut window);
            }
        }
        assert_eq!(members.each_ref().map(|member| member.position), positions);
        owner.pending = Some(owner.commands.take().unwrap().submit().unwrap());
        owner.pending.as_mut().unwrap().retire().unwrap();
        owner.pending = None;
        // One whole-stack acknowledgement, then comparisons. This test-only
        // orchestration stops before the head; the public start_pair numerical
        // integration gate separately covers head/readout and logical commitment.
        let mut distinguished = [false; 2];
        for capture in &mut owners {
            let different = assert_append(capture, &mut members, positions);
            for kind in 0..2 {
                distinguished[kind] |= different[kind];
            }
            capture.close();
        }
        assert!(
            distinguished.into_iter().all(|different| different),
            "both K and V must discriminate member routing"
        );
        for (layer, capture) in attention.iter_mut().enumerate() {
            if let Some(capture) = capture {
                assert_eq!(
                    bytes(&mut capture.actual),
                    bytes(&mut capture.suffix),
                    "window span/routing layer={layer} step={step}"
                );
                capture.close();
            }
        }
        assert_eq!(members.each_ref().map(|member| member.position), positions);
        for member in members {
            member.position += 1;
        }
        owner.phase = Phase::Idle;
    }
    window.close();
    assert_eq!(pool.granted(), resident, "all test captures refunded");
    for (member, id) in ids.into_iter().enumerate() {
        assert_eq!(model.position(id), Some(histories[member].prompt.len() + 3));
        model.release(id).unwrap();
    }
    model.close().unwrap();
    model.close().unwrap();
    assert_eq!(pool.granted(), 0);
}

#[test]
#[ignore = "real 15 GB E4B package; release-only serialized CUDA gate"]
fn late_head_binding_failure_retains_the_actual_enqueued_stack_until_close() {
    let _device = DEVICE.lock().unwrap();
    let source = source();
    let histories = histories();
    let capacity = histories
        .iter()
        .map(|history| history.prompt.len() + 1)
        .max()
        .unwrap();
    let context = CudaContext::new(0).unwrap();
    let pool = BytePool::new(20_u64 << 30).shared();
    let mut model = Gemma4CudaModelWide::load(
        &source,
        &context,
        &pool,
        capacity,
        scratch(&source, capacity),
    )
    .unwrap();
    let [first, second, later] = issued_ids::<3>();
    for (id, history) in [first, second].into_iter().zip(&histories) {
        assert!(matches!(
            model.admit(id, capacity).unwrap(),
            Admission::Ready
        ));
        model.prefill(id, &history.prompt, false).unwrap();
    }
    let ids = [first, second];
    let positions = ids.map(|id| model.position(id).unwrap());
    let charged = pool.granted();
    assert!(charged > model.owner().model.cuda.allocated_bytes());
    // PLE preparation uses host lookups, not this role. The sole remaining use
    // in Packed::enqueue is the tied output projection AFTER the entire stack.
    assert!(
        model
            .owner_mut()
            .model
            .cuda
            .slots
            .remove(&ParameterId::TokenEmbedding)
            .is_some()
    );
    let tokens = histories.each_ref().map(|history| history.continuation[0]);
    let error = model.start_pair(ids, tokens).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("logical projection has no active allocation")
    );
    let owner = model.owner_mut();
    assert!(matches!(owner.phase, Phase::Failed));
    assert!(owner.commands.is_some(), "positive recording remains owned");
    assert!(owner.pending.is_none());
    assert!(owner.members.is_none());
    assert!(matches!(
        owner.failure.as_ref().unwrap().cause(),
        CudaBf16Error::Invalid("logical projection has no active allocation")
    ));
    // prepare() uploaded these contexts without command guards. Each must now
    // be retained by its own real layer's multiply, including layer 41. The
    // final norm guard additionally proves progress reached the output head.
    for inputs in &mut owner.model.packed.layers {
        assert!(matches!(
            inputs.context.read_chunk(0, 1),
            Err(CudaBf16Error::InUse)
        ));
    }
    assert!(matches!(
        owner.model.packed.buffers.final_norm.read_chunk(0, 1),
        Err(CudaBf16Error::InUse)
    ));
    assert_eq!(ids.map(|id| model.position(id).unwrap()), positions);
    assert_eq!(pool.granted(), charged, "no refund on uncertain enqueue");
    assert!(model.start_pair(ids, tokens).is_err());
    assert!(model.prefill(first, &[tokens[0]], false).is_err());
    assert!(model.admit(later, capacity).is_err());
    assert!(model.release(first).is_err());
    assert!(model.poll_pair().is_err());
    assert!(matches!(model.owner().phase, Phase::Failed));
    assert_eq!(ids.map(|id| model.position(id).unwrap()), positions);
    assert_eq!(
        pool.granted(),
        charged,
        "rejections retain the complete owner"
    );
    model.close().unwrap();
    assert!(matches!(model.owner().phase, Phase::Closed));
    assert!(model.owner().commands.is_none());
    assert_eq!(pool.granted(), 0);
    assert!(
        model.start_pair(ids, tokens).is_err(),
        "close never recovers execution"
    );
    model.close().unwrap();
    assert_eq!(pool.granted(), 0);
}

#[test]
#[ignore = "loads the real 15 GB inventory twice sequentially; release-only serialized CUDA gate"]
fn partial_initialization_owns_weights_when_gemm_workspace_is_refused() {
    let _device = DEVICE.lock().unwrap();
    let source = source();
    let context = CudaContext::new(0).unwrap();
    let measurement = BytePool::new(20_u64 << 30).shared();
    // Derive the real backend charge, including its padding, rather than sum
    // logical tensor sizes. WEIGHT_STAGING is retained host memory, not a
    // BytePool grant; assert that distinction against the actual loaded owner.
    let charged = {
        let stream = CudaBf16Stream::new(&context).unwrap();
        let mut weights =
            Gemma4CudaWeights::load_on(&source, &stream, &measurement, WEIGHT_STAGING).unwrap();
        let charged = measurement.granted();
        assert_eq!(charged, weights.allocated_bytes());
        assert!(charged > 0);
        weights.close().unwrap();
        weights.close().unwrap();
        assert_eq!(measurement.granted(), 0, "no concurrent weight inventory");
        charged
    };
    let pool = BytePool::new(charged).shared();
    let capacity = 2;
    let mut error = match Gemma4CudaModelWide::load(
        &source,
        &context,
        &pool,
        capacity,
        scratch(&source, capacity),
    ) {
        Ok(mut model) => {
            model.close().unwrap();
            panic!("exact weight-only pool must refuse the GEMM workspace");
        }
        Err(error) => error,
    };
    assert!(
        error.retirement_pending(),
        "owning partial preparation survives load failure"
    );
    assert_eq!(
        pool.granted(),
        charged,
        "fully uploaded weights stay owned before any sequence exists"
    );
    let workspace_refusal = pool
        .reserve(u64::try_from(GEMM_WORKSPACE).unwrap())
        .unwrap_err();
    let expected = super::super::backend(&CudaBf16Error::Reserve(workspace_refusal).into());
    assert_eq!(
        error.cause().to_string(),
        expected.to_string(),
        "the next actual allocation is GEMM workspace"
    );
    error.retire().unwrap();
    assert!(!error.retirement_pending());
    assert_eq!(pool.granted(), 0);
    error.retire().unwrap();
    assert_eq!(pool.granted(), 0);
}
