#![cfg(feature = "cuda")]

use cudarc::driver::CudaContext;
use engine_core::{DataType, WeightTensorSpec};
use engine_nvidia::CudaBf16Weights;
use ribn_foundation::BytePool;

#[test]
#[ignore = "requires CUDA GPU; native BF16 byte storage/readback/retirement qualification"]
fn native_bf16_bits_alignment_unload_reload() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(32).shared();
    let specs = vec![
        WeightTensorSpec::new("embedding", vec![1, 5], DataType::BF16).unwrap(),
        WeightTensorSpec::new("norm", vec![3], DataType::BF16).unwrap(),
    ];
    // Storage preserves signed zero, infinities and arbitrary NaN payloads;
    // no BF16 arithmetic or host F32 promotion is involved.
    let bits = [
        vec![0x0000_u16, 0x8000, 0x7f80, 0xff80, 0x7fc1],
        vec![0x3f80_u16, 0x7f81, 0xffff],
    ];
    let bytes = bits.map(|words| {
        words
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>()
    });
    let mut previous_id = None;
    for staging in [3, 5, 1] {
        let mut weights =
            CudaBf16Weights::allocate(specs.clone(), stream.clone(), &pool, staging).unwrap();
        let id = weights.allocation_id();
        assert_ne!(Some(id), previous_id);
        previous_id = Some(id);
        assert_eq!(weights.allocated_bytes(), 32);
        assert_eq!(weights.tensor_range(0), Some(0..10));
        assert_eq!(weights.tensor_range(1), Some(16..22));
        assert_eq!(weights.spec(0), Some(&specs[0]));
        assert_eq!(pool.granted(), 32);
        assert!(!weights.ready());
        weights.upload(1, &mut bytes[1].as_slice()).unwrap();
        assert!(!weights.ready());
        weights.upload(0, &mut bytes[0].as_slice()).unwrap();
        assert!(weights.ready());
        for (slot, expected) in bytes.iter().enumerate() {
            let mut actual = Vec::new();
            for offset in (0..expected.len()).step_by(staging) {
                actual.extend(
                    weights
                        .read_chunk(slot, offset as u64, staging.min(expected.len() - offset))
                        .unwrap(),
                );
            }
            assert_eq!(&actual, expected);
        }
        weights.close().unwrap();
        weights.close().unwrap();
        assert!(!weights.ready());
        assert_eq!(pool.granted(), 0);
        drop(weights);
        assert_eq!(pool.available(), 32);
    }
}
