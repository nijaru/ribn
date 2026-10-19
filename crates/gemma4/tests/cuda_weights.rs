//! Native weight storage only, not a forward pass or GPU-wide peak guarantee.
#![cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use engine_gemma4::{Gemma4CudaWeights, Gemma4Source, ParameterId};
use ribn_foundation::BytePool;
use std::io::Read;
#[path = "common/weights.rs"]
mod weights;
use weights::{Dir, inventory, package};

fn allocation_bytes(source: &Gemma4Source) -> u64 {
    source
        .parameters()
        .values()
        .map(|p| {
            let n = u64::try_from(p.spec().byte_len().unwrap()).unwrap();
            n.div_ceil(16) * 16
        })
        .sum()
}

#[test]
#[ignore = "requires an idle CUDA GPU; native BF16 loading, not Gemma execution"]
fn materializes_pinned_readers_bitwise_with_tied_identity_and_retirement() {
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&package).unwrap();
    drop(package);
    // Replacing the pathname must not redirect already selected BF16 readers.
    std::fs::rename(
        dir.0.join("model.safetensors"),
        dir.0.join("selected.safetensors"),
    )
    .unwrap();
    std::fs::write(dir.0.join("model.safetensors"), b"not the selected file").unwrap();
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let bytes = allocation_bytes(&source);
    let pool = BytePool::new(bytes).shared();
    let mut previous = None;
    for _ in 0..2 {
        let mut materialized = Gemma4CudaWeights::load(&source, stream.clone(), &pool, 5).unwrap();
        assert_eq!(materialized.config(), source.config());
        assert_eq!(materialized.allocated_bytes(), bytes);
        assert_eq!(pool.granted(), bytes);
        assert_eq!(
            materialized.token_embedding_slot(),
            materialized.output_projection_slot()
        );
        assert_eq!(
            materialized.parameter_slot(ParameterId::TokenEmbedding),
            Some(materialized.output_projection_slot())
        );
        let allocation = materialized.allocation_id();
        assert_ne!(previous, Some(allocation));
        previous = Some(allocation);
        for (&id, parameter) in source.parameters() {
            let mut reader = parameter.reader();
            let mut expected = [0; 5];
            let mut offset = 0;
            while offset < parameter.spec().byte_len().unwrap() {
                let len = expected
                    .len()
                    .min(parameter.spec().byte_len().unwrap() - offset);
                reader.read_exact(&mut expected[..len]).unwrap();
                assert_eq!(
                    materialized
                        .read_chunk(id, u64::try_from(offset).unwrap(), len)
                        .unwrap(),
                    expected[..len]
                );
                offset += len;
            }
        }
        materialized.close().unwrap();
        materialized.close().unwrap();
        assert_eq!(pool.granted(), 0);
        assert!(
            materialized
                .read_chunk(ParameterId::TokenEmbedding, 0, 1)
                .is_err()
        );
    }
}

#[test]
#[ignore = "requires an idle CUDA GPU; injects source IO failure after preflight"]
fn failed_partial_upload_retains_model_cleanup_owner() {
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&package).unwrap();
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let bytes = allocation_bytes(&source);
    let too_small = BytePool::new(bytes - 1).shared();
    let refused = Gemma4CudaWeights::load(&source, stream.clone(), &too_small, 5)
        .err()
        .unwrap();
    assert!(!refused.retirement_pending());
    assert_eq!(too_small.granted(), 0);
    // Deliberately violate the no-in-place-mutation precondition to inject a late
    // reader EOF, not to claim mutable-package support. The first tensor starts
    // at payload offset zero: three chunks complete before the next read fails.
    let path = dir.0.join("model.safetensors");
    let mut file = std::fs::File::open(&path).unwrap();
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix).unwrap();
    let payload = 8 + u64::from_le_bytes(prefix);
    std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_len(payload + 17)
        .unwrap();
    let pool = BytePool::new(bytes).shared();
    let mut failure = Gemma4CudaWeights::load(&source, stream, &pool, 5)
        .err()
        .unwrap();
    assert!(failure.retirement_pending());
    assert_eq!(pool.granted(), bytes);
    failure.retire().unwrap();
    failure.retire().unwrap();
    assert!(!failure.retirement_pending());
    assert_eq!(pool.granted(), 0);
}
