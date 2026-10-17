//! Capacity/addressing parity, not a new cross-engine numerical tolerance.
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use engine_core::{
    ConvolutionStateShape, DataType, KvStateSpec, RecurrentMatrixShape, RecurrentStateSpec,
};
use engine_nvidia::{CudaHybridState, CudaQwen35BatchDecode, CudaQwen35Decode, QwenLayerKind};
use engine_qwen::{QwenGguf, QwenLayerKind as ModelLayerKind};

const TOKENS: [u32; 8] = [760, 6511, 314, 9338, 369, 12_675, 1017, 283];

fn token(member: usize, position: u32) -> u32 {
    TOKENS[(member + usize::try_from(position).unwrap()) % TOKENS.len()]
}

fn fresh_state(
    stream: &Arc<CudaStream>,
    initializer: &mut CudaQwen35Decode,
    capacity: u32,
    position: u32,
    member: usize,
) -> CudaHybridState {
    let kv = KvStateSpec::new(16, 4, 256, capacity, DataType::F16).unwrap();
    let recurrent = RecurrentStateSpec::new(
        48,
        RecurrentMatrixShape::new(48, 128, 128).unwrap(),
        ConvolutionStateShape::new(10_240, 3).unwrap(),
        DataType::F32,
        DataType::F32,
    )
    .unwrap();
    let mut state =
        CudaHybridState::from_specs(Arc::clone(stream), Some(kv), Some(recurrent)).unwrap();
    state.zero().unwrap();
    // Build real, distinct histories. Never claim an uncomputed prefix merely
    // to place a write near an allocation boundary.
    for offset in 0..position {
        initializer
            .prefill_step(&mut state, token(member, offset), offset)
            .unwrap();
    }
    stream.synchronize().unwrap();
    state.advance_to(position).unwrap();
    state
}

fn assert_finite_bits(expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len());
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        assert!(
            expected.is_finite() && actual.is_finite(),
            "element {index}"
        );
        assert_eq!(expected.to_bits(), actual.to_bits(), "element {index}");
    }
}

fn assert_state_bits(
    stream: &Arc<CudaStream>,
    expected: &CudaHybridState,
    actual: &CudaHybridState,
) {
    let expected_kv = expected.kv().unwrap();
    let actual_kv = actual.kv().unwrap();
    for layer in 0..16 {
        for (expected, actual) in [
            (expected_kv.layer_keys(layer), actual_kv.layer_keys(layer)),
            (
                expected_kv.layer_values(layer),
                actual_kv.layer_values(layer),
            ),
        ] {
            let actual = actual.unwrap().as_f16().unwrap();
            let expected = expected
                .unwrap()
                .as_f16()
                .unwrap()
                .try_slice(..actual.len())
                .unwrap();
            let expected = stream.clone_dtoh(&expected).unwrap();
            let actual = stream.clone_dtoh(actual).unwrap();
            assert!(actual.iter().all(|bits| bits & 0x7c00 != 0x7c00));
            assert_eq!(expected.len(), actual.len());
            assert_eq!(
                expected.iter().zip(&actual).position(|(a, b)| a != b),
                None,
                "first differing KV element in layer {layer}"
            );
        }
    }
    let expected = expected.recurrent().unwrap();
    let actual = actual.recurrent().unwrap();
    for layer in 0..48 {
        for (expected, actual) in [
            (expected.layer_matrix(layer), actual.layer_matrix(layer)),
            (
                expected.layer_convolution(layer),
                actual.layer_convolution(layer),
            ),
        ] {
            assert_finite_bits(
                &stream
                    .clone_dtoh(expected.unwrap().as_f32().unwrap())
                    .unwrap(),
                &stream
                    .clone_dtoh(actual.unwrap().as_f32().unwrap())
                    .unwrap(),
            );
        }
    }
}

#[test]
#[ignore = "requires an idle CUDA GPU and the pinned ENGINE_QWEN_GGUF"]
fn different_capacities_preserve_hidden_and_all_continuation_bits() {
    let path = std::env::var("ENGINE_QWEN_GGUF").expect("pinned GGUF path");
    let provider = QwenGguf::open(path).unwrap();
    assert_eq!(
        provider.artifact_id(),
        "gguf:sha256:322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482"
    );
    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let weights = super::stage_full_text_path(&provider, &context, &stream);
    let kinds: Vec<_> = (0..64)
        .map(|layer| match provider.layer_kind(layer).unwrap() {
            ModelLayerKind::Recurrent => QwenLayerKind::Recurrent,
            ModelLayerKind::FullAttention => QwenLayerKind::FullAttention,
        })
        .collect();
    let epsilon = provider.config().rms_norm_epsilon;
    let mut initializer = CudaQwen35Decode::new(
        &context,
        Arc::clone(&stream),
        Arc::clone(&weights),
        kinds.clone(),
        epsilon,
    )
    .unwrap();
    for members in [1, 2, 4, 8] {
        let build_lane = || {
            CudaQwen35BatchDecode::new(
                &context,
                Arc::clone(&stream),
                Arc::clone(&weights),
                &kinds,
                epsilon,
                members,
            )
            .unwrap()
        };
        let mut control = build_lane();
        let mut candidate = build_lane();
        // Reuse the same lanes across growth and smaller subsequent capacities.
        // Small cases fill each cache exactly; large cases exercise score strides
        // through 4096 with real short histories and nonadjacent row capacities.
        for (capacities, starts) in [
            ([2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7]),
            (
                [4096, 17, 129, 512, 31, 64, 257, 1024],
                [3, 6, 12, 1, 9, 0, 5, 15],
            ),
            ([9, 8, 7, 6, 5, 4, 3, 2], [7, 6, 5, 4, 3, 2, 1, 0]),
        ] {
            let maximum = *capacities[..members].iter().max().unwrap();
            let mut reference: Vec<_> = (0..members)
                .map(|member| {
                    fresh_state(&stream, &mut initializer, maximum, starts[member], member)
                })
                .collect();
            let mut actual: Vec<_> = (0..members)
                .map(|member| {
                    fresh_state(
                        &stream,
                        &mut initializer,
                        capacities[member],
                        starts[member],
                        member,
                    )
                })
                .collect();
            for step in 0..2 {
                let positions: Vec<_> =
                    starts[..members].iter().map(|start| start + step).collect();
                let tokens: Vec<_> = positions
                    .iter()
                    .enumerate()
                    .map(|(member, &position)| token(member, position))
                    .collect();
                let expected = control
                    .decode_step_batch(
                        &mut reference.iter_mut().collect::<Vec<_>>(),
                        &tokens,
                        &positions,
                    )
                    .unwrap();
                let selected = candidate
                    .decode_step_batch(
                        &mut actual.iter_mut().collect::<Vec<_>>(),
                        &tokens,
                        &positions,
                    )
                    .expect("heterogeneous batch");
                assert_eq!(expected, selected);
                for member in 0..members {
                    assert_finite_bits(
                        &control.copy_hidden_member(member).unwrap(),
                        &candidate.copy_hidden_member(member).unwrap(),
                    );
                    reference[member].advance_to(positions[member] + 1).unwrap();
                    actual[member].advance_to(positions[member] + 1).unwrap();
                    assert_state_bits(&stream, &reference[member], &actual[member]);
                }
            }
        }
    }
    stream.synchronize().unwrap();
}

/// Cover active attention extents through the pinned model's 262144-token
/// ceiling without pretending a sparse primitive input is a real model history.
/// The slice entrypoint is the unchanged per-member operator reference. Queue
/// all overlapping score views before readback to exercise stream ordering.
#[test]
#[ignore = "requires an idle CUDA GPU"]
fn shared_scores_preserve_attention_bits_through_model_context_limit() {
    use engine_nvidia::{ATTN_HEAD_DIM, ATTN_KV_HEADS, ATTN_Q_HEADS, CudaQwen35Ops};

    let context = CudaContext::new(0).unwrap();
    let stream = context.default_stream();
    let ops = CudaQwen35Ops::from_context(&context, Arc::clone(&stream)).unwrap();
    let row = ATTN_Q_HEADS * ATTN_HEAD_DIM;
    let kv_row = ATTN_KV_HEADS * ATTN_HEAD_DIM;
    let q = stream.clone_htod(&vec![0.125_f32; row]).unwrap();
    let gate = stream.alloc_zeros::<f32>(row * 2).unwrap();
    let capacities = [1, 2, 4096, 262_144];
    let mut cases = Vec::new();
    for (&capacity, value) in capacities.iter().zip([0.125_f32, 0.25, 0.375, 0.5]) {
        let mut keys = stream.alloc_zeros::<u16>(capacity * kv_row).unwrap();
        let mut values = stream.alloc_zeros::<u16>(capacity * kv_row).unwrap();
        let k = stream.clone_htod(&vec![value; kv_row]).unwrap();
        let v = stream.clone_htod(&vec![value * 2.0; kv_row]).unwrap();
        ops.kv_append_f16(
            &k,
            &v,
            &mut keys,
            &mut values,
            capacity - 1,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
        )
        .unwrap();
        let mut scores = stream.alloc_zeros::<f32>(ATTN_Q_HEADS * capacity).unwrap();
        let mut output = stream.alloc_zeros::<f32>(row).unwrap();
        ops.attn_score_gqa(
            &q,
            &keys,
            &values,
            &gate,
            &mut scores,
            &mut output,
            capacity,
            1,
            capacity,
            ATTN_Q_HEADS,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
        )
        .unwrap();
        let expected = stream.clone_dtoh(&output).unwrap();
        assert!(expected.iter().any(|&x| x != 0.0));
        cases.push((keys, values, expected));
    }
    let maximum = *capacities.iter().max().unwrap();
    let mut scores = stream.alloc_zeros::<f32>(ATTN_Q_HEADS * maximum).unwrap();
    let mut output = stream.alloc_zeros::<f32>(row * cases.len()).unwrap();
    for (member, (&capacity, (keys, values, _))) in capacities.iter().zip(&cases).enumerate() {
        ops.attn_score_gqa_views(
            &q.try_slice(..).unwrap(),
            keys,
            values,
            &gate.try_slice(..).unwrap(),
            &mut scores.try_slice_mut(..ATTN_Q_HEADS * capacity).unwrap(),
            &mut output
                .try_slice_mut(member * row..(member + 1) * row)
                .unwrap(),
            capacity,
            1,
            capacity,
            ATTN_Q_HEADS,
            ATTN_KV_HEADS,
            ATTN_HEAD_DIM,
        )
        .unwrap();
    }
    let actual = stream.clone_dtoh(&output).unwrap();
    for ((_, _, expected), actual) in cases.iter().zip(actual.chunks_exact(row)) {
        assert_finite_bits(expected, actual);
    }
}
