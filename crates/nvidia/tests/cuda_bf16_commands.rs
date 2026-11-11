//! Same-stream dataflow and retirement gates, not full-model promotion.
#![cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, sys};
use engine_core::{DataType, WeightTensorSpec};
use engine_nvidia::{
    AttentionGeometry, AttentionSpan, CudaBf16Commands, CudaBf16Error, CudaBf16Matrix,
    CudaBf16Stream,
};
use ribn_foundation::BytePool;
use std::{io::Cursor, sync::Arc};

const WIDTH: usize = 16;
fn matrix(stream: &CudaBf16Stream, pool: &Arc<BytePool>, rows: usize) -> CudaBf16Matrix {
    stream.matrix(rows, WIDTH, pool, rows * WIDTH * 2).unwrap()
}
fn bytes(rows: usize, shift: usize) -> Vec<u8> {
    (0..rows * WIDTH)
        .flat_map(|index| {
            let numerator = i16::try_from((index * 7 + shift) % 19).unwrap() - 9;
            let bits = (f32::from(numerator) / 8.0).to_bits();
            // These dyadic test inputs are exactly BF16-representable.
            assert_eq!(bits & 0xffff, 0);
            u16::try_from(bits >> 16).unwrap().to_le_bytes()
        })
        .collect()
}
struct Buffers {
    projected: CudaBf16Matrix,
    normed: CudaBf16Matrix,
    rotated: CudaBf16Matrix,
    query: CudaBf16Matrix,
    keys: [CudaBf16Matrix; 2],
    values: [CudaBf16Matrix; 2],
    attended: CudaBf16Matrix,
    packed: CudaBf16Matrix,
    activated: CudaBf16Matrix,
    product: CudaBf16Matrix,
    sum: CudaBf16Matrix,
    scaled: CudaBf16Matrix,
    output: CudaBf16Matrix,
    selected: CudaBf16Matrix,
}
impl Buffers {
    fn new(stream: &CudaBf16Stream, pool: &Arc<BytePool>, prefix: &CudaBf16Matrix) -> Self {
        let mut result = Self {
            projected: matrix(stream, pool, 2),
            normed: matrix(stream, pool, 2),
            rotated: matrix(stream, pool, 2),
            query: matrix(stream, pool, 1),
            keys: [matrix(stream, pool, 4), matrix(stream, pool, 4)],
            values: [matrix(stream, pool, 4), matrix(stream, pool, 4)],
            attended: matrix(stream, pool, 1),
            packed: matrix(stream, pool, 2),
            activated: matrix(stream, pool, 2),
            product: matrix(stream, pool, 2),
            sum: matrix(stream, pool, 2),
            scaled: matrix(stream, pool, 2),
            output: matrix(stream, pool, 2),
            selected: matrix(stream, pool, 1),
        };
        for (member, reach) in [1, 3].into_iter().enumerate() {
            result.keys[member]
                .copy_rows_from(0, prefix, 0..reach)
                .unwrap();
            result.values[member]
                .copy_rows_from(0, prefix, 0..reach)
                .unwrap();
        }
        result
    }
    fn close(&mut self) {
        for matrix in [
            &mut self.projected,
            &mut self.normed,
            &mut self.rotated,
            &mut self.query,
            &mut self.attended,
            &mut self.packed,
            &mut self.activated,
            &mut self.product,
            &mut self.sum,
            &mut self.scaled,
            &mut self.output,
            &mut self.selected,
        ] {
            matrix.close().unwrap();
        }
        for matrix in self.keys.iter_mut().chain(&mut self.values) {
            matrix.close().unwrap();
        }
    }
}
fn geometry() -> AttentionGeometry {
    AttentionGeometry {
        heads: 1,
        kv_heads: 1,
        width: WIDTH,
        window: Some(2),
    }
}
fn span(prefix: usize) -> AttentionSpan {
    AttentionSpan {
        query_start: prefix,
        key_start: 0,
        key_sequence: prefix + 1,
    }
}

#[test]
#[ignore = "requires CUDA; serialize device qualification"]
#[expect(
    clippy::too_many_lines,
    reason = "paired actual blocking and queued paths with unequal member histories"
)]
fn dependent_commands_match_blocking_bits_and_preserve_member_histories() {
    let context = CudaContext::new(0).unwrap();
    let stream = CudaBf16Stream::new(&context).unwrap();
    let pool = BytePool::new(64 * 1024 * 1024).shared();
    let specs = vec![
        WeightTensorSpec::new("q", vec![16, 16], DataType::BF16).unwrap(),
        WeightTensorSpec::new("out", vec![16, 16], DataType::BF16).unwrap(),
        WeightTensorSpec::new("norm", vec![16], DataType::BF16).unwrap(),
    ];
    let mut weights = stream.weights(specs, &pool, 512).unwrap();
    weights.upload(0, &mut Cursor::new(bytes(16, 1))).unwrap();
    weights.upload(1, &mut Cursor::new(bytes(16, 3))).unwrap();
    weights.upload(2, &mut Cursor::new(bytes(1, 5))).unwrap();
    let mut gemm = stream.gemm(&pool, 1 << 20).unwrap();
    let mut rms = stream.rms_norm().unwrap();
    let mut rotary = stream.rotary().unwrap();
    let mut attention = stream.attention(&pool, 4).unwrap();
    let mut elementwise = stream.elementwise().unwrap();
    let mut factors = stream
        .rotary_factors(
            &pool,
            &[1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0, 0.0],
            2,
        )
        .unwrap();
    let mut cos = matrix(&stream, &pool, 2);
    let mut sin = matrix(&stream, &pool, 2);
    factors.prepare_into(&[1, 3], &mut cos, &mut sin).unwrap();
    let mut input = matrix(&stream, &pool, 2);
    input.upload(&bytes(2, 2)).unwrap();
    let mut prefix = matrix(&stream, &pool, 3);
    prefix.upload(&bytes(3, 7)).unwrap();
    let mut baseline = Buffers::new(&stream, &pool, &prefix);
    let mut candidate = Buffers::new(&stream, &pool, &prefix);
    let grant = pool.granted();

    gemm.linear_into(&mut weights, 0, &mut input, &mut baseline.projected)
        .unwrap();
    rms.normalize_into(
        Some((&mut weights, 2)),
        WIDTH,
        1e-6,
        &mut baseline.projected,
        &mut baseline.normed,
    )
    .unwrap();
    rotary
        .apply_into(
            &mut cos,
            &mut sin,
            &mut baseline.normed,
            &mut baseline.rotated,
        )
        .unwrap();
    for (member, reach) in [1, 3].into_iter().enumerate() {
        baseline.keys[member]
            .copy_rows_from(reach, &baseline.rotated, member..member + 1)
            .unwrap();
        baseline.values[member]
            .copy_rows_from(reach, &input, member..member + 1)
            .unwrap();
        baseline
            .query
            .copy_rows_from(0, &baseline.rotated, member..member + 1)
            .unwrap();
        attention
            .attend_into(
                &baseline.query,
                &baseline.keys[member],
                &baseline.values[member],
                geometry(),
                span(reach),
                &mut baseline.attended,
            )
            .unwrap();
        baseline
            .packed
            .copy_rows_from(member, &baseline.attended, 0..1)
            .unwrap();
    }
    elementwise
        .gelu_tanh_into(&baseline.packed, &mut baseline.activated)
        .unwrap();
    elementwise
        .multiply_into(&baseline.activated, &baseline.normed, &mut baseline.product)
        .unwrap();
    elementwise
        .add_into(&baseline.product, &input, &mut baseline.sum)
        .unwrap();
    elementwise
        .scale_into(&baseline.sum, 0.5, &mut baseline.scaled)
        .unwrap();
    gemm.linear_into(&mut weights, 1, &mut baseline.scaled, &mut baseline.output)
        .unwrap();
    baseline
        .selected
        .copy_rows_from(0, &baseline.output, 1..2)
        .unwrap();

    // Nineteen explicit operations, including two distinct KV owners/reaches.
    let mut commands = CudaBf16Commands::new(&stream, 19).unwrap();
    commands
        .linear_into(
            &mut gemm,
            &mut weights,
            0,
            &mut input,
            &mut candidate.projected,
        )
        .unwrap();
    assert!(!candidate.projected.ready());
    assert!(matches!(
        candidate.projected.read_chunk(0, 2),
        Err(CudaBf16Error::InUse)
    ));
    assert!(matches!(
        input.upload(&bytes(2, 2)),
        Err(CudaBf16Error::InUse)
    ));
    assert!(matches!(
        weights.read_chunk(0, 0, 2),
        Err(CudaBf16Error::InUse)
    ));
    let foreign = gemm
        .linear_into(&mut weights, 0, &mut input, &mut candidate.output)
        .unwrap_err();
    assert!(matches!(foreign.cause(), CudaBf16Error::InUse));
    assert!(!foreign.retirement_pending());
    commands
        .normalize_into(
            &mut rms,
            Some((&mut weights, 2)),
            WIDTH,
            1e-6,
            &mut candidate.projected,
            &mut candidate.normed,
        )
        .unwrap();
    commands
        .rotary_into(
            &mut rotary,
            &mut cos,
            &mut sin,
            &mut candidate.normed,
            &mut candidate.rotated,
        )
        .unwrap();
    for (member, reach) in [1, 3].into_iter().enumerate() {
        commands
            .copy_rows(
                &mut candidate.keys[member],
                reach,
                &candidate.rotated,
                member..member + 1,
            )
            .unwrap();
        commands
            .copy_rows(
                &mut candidate.values[member],
                reach,
                &input,
                member..member + 1,
            )
            .unwrap();
        commands
            .copy_rows(
                &mut candidate.query,
                0,
                &candidate.rotated,
                member..member + 1,
            )
            .unwrap();
        commands
            .attend_into(
                &mut attention,
                &candidate.query,
                &candidate.keys[member],
                &candidate.values[member],
                geometry(),
                span(reach),
                &mut candidate.attended,
            )
            .unwrap();
        commands
            .copy_rows(&mut candidate.packed, member, &candidate.attended, 0..1)
            .unwrap();
    }
    commands
        .gelu_tanh_into(
            &mut elementwise,
            &candidate.packed,
            &mut candidate.activated,
        )
        .unwrap();
    commands
        .multiply_into(
            &mut elementwise,
            &candidate.activated,
            &candidate.normed,
            &mut candidate.product,
        )
        .unwrap();
    commands
        .add_into(
            &mut elementwise,
            &candidate.product,
            &input,
            &mut candidate.sum,
        )
        .unwrap();
    commands
        .scale_into(&mut elementwise, &candidate.sum, 0.5, &mut candidate.scaled)
        .unwrap();
    commands
        .linear_into(
            &mut gemm,
            &mut weights,
            1,
            &mut candidate.scaled,
            &mut candidate.output,
        )
        .unwrap();
    commands
        .copy_rows(&mut candidate.selected, 0, &candidate.output, 1..2)
        .unwrap();
    let mut submitted = commands.submit().unwrap();
    assert!(
        !candidate.output.ready(),
        "event record is not completed publication"
    );
    assert!(matches!(
        candidate.output.close(),
        Err(CudaBf16Error::InUse)
    ));
    while !submitted.poll().unwrap() {
        std::thread::yield_now();
    }
    assert!(
        !candidate.output.ready(),
        "a terminal query must not drain or publish accesses"
    );
    submitted.retire().unwrap();
    assert_eq!(
        pool.granted(),
        grant,
        "commands allocate no device workspace"
    );
    for (actual, expected) in [
        (&mut candidate.packed, &mut baseline.packed),
        (&mut candidate.output, &mut baseline.output),
        (&mut candidate.selected, &mut baseline.selected),
    ] {
        let len = actual.shape().0 * WIDTH * 2;
        let got = actual.read_chunk(0, len).unwrap();
        assert_eq!(got, expected.read_chunk(0, len).unwrap());
        for pair in got.as_chunks::<2>().0 {
            assert_ne!(u16::from_le_bytes([pair[0], pair[1]]) & 0x7f80, 0x7f80);
        }
    }
    for (member, reach) in [2, 4].into_iter().enumerate() {
        assert_eq!(candidate.keys[member].ready(), reach == 4);
        assert_eq!(
            candidate.keys[member]
                .read_chunk(0, reach * WIDTH * 2)
                .unwrap(),
            baseline.keys[member]
                .read_chunk(0, reach * WIDTH * 2)
                .unwrap()
        );
        assert_eq!(
            candidate.values[member]
                .read_chunk(0, reach * WIDTH * 2)
                .unwrap(),
            baseline.values[member]
                .read_chunk(0, reach * WIDTH * 2)
                .unwrap()
        );
    }
    // Completion unblocks ordinary operations; it does not silently poison peers.
    input.upload(&bytes(2, 8)).unwrap();
    gemm.linear_into(&mut weights, 0, &mut input, &mut candidate.projected)
        .unwrap();
    baseline.close();
    candidate.close();
    for matrix in [&mut input, &mut prefix, &mut cos, &mut sin] {
        matrix.close().unwrap();
    }
    attention.close().unwrap();
    rotary.close().unwrap();
    rms.close().unwrap();
    elementwise.close().unwrap();
    factors.close().unwrap();
    gemm.close().unwrap();
    weights.close().unwrap();
    assert_eq!(pool.granted(), 0);
}

#[test]
#[ignore = "requires CUDA; serialize device qualification"]
fn recording_retirement_abandons_outputs_and_releases_dropped_projection_owners() {
    let context = CudaContext::new(0).unwrap();
    let stream = CudaBf16Stream::new(&context).unwrap();
    let pool = BytePool::new(2 * 1024 * 1024).shared();
    let spec = WeightTensorSpec::new("projection", vec![16, 16], DataType::BF16).unwrap();
    let mut weights = stream.weights(vec![spec], &pool, 512).unwrap();
    weights.upload(0, &mut Cursor::new(bytes(16, 1))).unwrap();
    let mut gemm = stream.gemm(&pool, 1 << 20).unwrap();
    let mut input = matrix(&stream, &pool, 2);
    input.upload(&bytes(2, 2)).unwrap();
    let mut output = matrix(&stream, &pool, 2);
    output.upload(&bytes(2, 3)).unwrap();
    let output_charge = output.allocated_bytes();
    let mut commands = CudaBf16Commands::new(&stream, 1).unwrap();
    commands
        .linear_into(&mut gemm, &mut weights, 0, &mut input, &mut output)
        .unwrap();
    assert!(!output.ready());
    assert!(matches!(output.read_chunk(0, 2), Err(CudaBf16Error::InUse)));
    assert!(matches!(input.close(), Err(CudaBf16Error::InUse)));
    assert!(matches!(weights.close(), Err(CudaBf16Error::InUse)));
    assert!(matches!(gemm.close(), Err(CudaBf16Error::InUse)));
    let grant = pool.granted();
    assert!(grant > output_charge);
    drop(input);
    drop(weights);
    drop(gemm);
    assert_eq!(pool.granted(), grant, "recording retains the actual owners");

    // No submit or terminal event record: the model abandons positive work.
    commands.retire().unwrap();
    assert_eq!(
        pool.granted(),
        output_charge,
        "drain releases operand, weight and cuBLAS-program workspace charges"
    );
    commands.retire().unwrap();
    assert_eq!(pool.granted(), output_charge, "retirement is idempotent");
    assert!(!output.ready());
    assert!(matches!(
        output.read_chunk(0, 2),
        Err(CudaBf16Error::Poisoned)
    ));
    assert!(matches!(
        output.upload(&bytes(2, 4)),
        Err(CudaBf16Error::Poisoned)
    ));
    let failure = commands
        .submit()
        .err()
        .expect("retired recording cannot submit");
    assert!(matches!(failure.cause(), CudaBf16Error::Poisoned));
    assert!(!failure.retirement_pending());
    output.close().unwrap();
    assert_eq!(pool.granted(), 0);
}

#[test]
#[ignore = "requires CUDA; serialize device qualification"]
fn later_validation_failure_retains_prior_accesses_and_never_publishes() {
    let context = CudaContext::new(0).unwrap();
    let stream = CudaBf16Stream::new(&context).unwrap();
    let pool = BytePool::new(1024 * 1024).shared();
    let mut input = matrix(&stream, &pool, 2);
    input.upload(&bytes(2, 1)).unwrap();
    let mut first = matrix(&stream, &pool, 2);
    let mut untouched = matrix(&stream, &pool, 2);
    untouched.upload(&bytes(2, 3)).unwrap();
    let mut commands = CudaBf16Commands::new(&stream, 1).unwrap();
    commands.copy_rows(&mut first, 0, &input, 0..2).unwrap();
    let mut failure = commands
        .copy_rows(&mut untouched, 0, &input, 0..2)
        .unwrap_err();
    assert!(matches!(failure.cause(), CudaBf16Error::Invalid(_)));
    assert!(failure.retirement_pending());
    assert!(!first.ready());
    assert!(matches!(
        first.read_chunk(0, 2),
        Err(CudaBf16Error::Poisoned)
    ));
    assert_eq!(untouched.read_chunk(0, 64).unwrap(), bytes(2, 3));
    assert!(matches!(input.close(), Err(CudaBf16Error::InUse)));
    let grant = pool.granted();
    drop(commands);
    drop(input);
    assert_eq!(
        pool.granted(),
        grant,
        "error retains a dropped input's actual allocation"
    );
    failure.retire().unwrap();
    failure.retire().unwrap();
    assert!(
        !first.ready(),
        "successful cleanup cannot revive a failed transaction"
    );
    let mut retry = CudaBf16Commands::new(&stream, 1).unwrap();
    let mut poisoned = retry
        .copy_rows(&mut untouched, 0, &first, 0..2)
        .unwrap_err();
    assert!(matches!(poisoned.cause(), CudaBf16Error::Poisoned));
    poisoned.retire().unwrap();
    first.close().unwrap();
    untouched.close().unwrap();
    assert_eq!(pool.granted(), 0);
}

#[test]
#[ignore = "requires CUDA; serialize device qualification"]
fn closed_sources_and_destinations_reject_before_transport_access() {
    let context = CudaContext::new(0).unwrap();
    let stream = CudaBf16Stream::new(&context).unwrap();
    let pool = BytePool::new(1024 * 1024).shared();
    for closed_source in [false, true] {
        let mut source = matrix(&stream, &pool, 2);
        source.upload(&bytes(2, 1)).unwrap();
        let mut destination = matrix(&stream, &pool, 2);
        destination.upload(&bytes(2, 3)).unwrap();
        if closed_source {
            source.close().unwrap();
        } else {
            destination.close().unwrap();
        }
        let mut commands = CudaBf16Commands::new(&stream, 1).unwrap();
        let mut failure = commands
            .copy_rows(&mut destination, 0, &source, 0..2)
            .unwrap_err();
        assert!(matches!(failure.cause(), CudaBf16Error::Closed));
        failure.retire().unwrap();
        if closed_source {
            assert_eq!(destination.read_chunk(0, 64).unwrap(), bytes(2, 3));
        } else {
            assert_eq!(source.read_chunk(0, 64).unwrap(), bytes(2, 1));
        }
        source.close().unwrap();
        destination.close().unwrap();
        assert_eq!(pool.granted(), 0);
    }
}

#[test]
#[ignore = "requires CUDA; serialize device qualification"]
fn raw_stream_operands_cannot_enter_the_private_domain_even_during_capture() {
    let context = CudaContext::new(0).unwrap();
    let stream = CudaBf16Stream::new(&context).unwrap();
    let raw = context.new_stream().unwrap();
    let pool = BytePool::new(1024 * 1024).shared();
    let mut input = CudaBf16Matrix::allocate(2, WIDTH, Arc::clone(&raw), &pool, 64).unwrap();
    input.upload(&bytes(2, 1)).unwrap();
    let mut output = matrix(&stream, &pool, 2);
    output.upload(&bytes(2, 3)).unwrap();
    let mut commands = CudaBf16Commands::new(&stream, 1).unwrap();
    raw.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .unwrap();
    let operation = commands.copy_rows(&mut output, 0, &input, 0..2);
    let ended = raw.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    drop(ended);
    let mut failure = operation.unwrap_err();
    assert!(matches!(failure.cause(), CudaBf16Error::Invalid(_)));
    failure.retire().unwrap();
    assert_eq!(output.read_chunk(0, 64).unwrap(), bytes(2, 3));
    input.close().unwrap();
    output.close().unwrap();
    assert_eq!(pool.granted(), 0);
}
