//! Bit-preserving cache copies and initialized-span validation; no arithmetic tolerance.
#![cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use engine_nvidia::{
    AttentionGeometry, AttentionSpan, CudaBf16Attention, CudaBf16Error, CudaBf16Matrix,
};
use ribn_foundation::BytePool;
use std::sync::Arc;

#[test]
#[ignore = "requires CUDA; serialize device qualification"]
fn row_copies_preserve_bits_and_untouched_rows() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(1024 * 1024).shared();
    for cols in [7, 256, 512] {
        for rows in [1, 3] {
            // Raw encodings deliberately include signed zero, infinity and NaN
            // payloads: a copy must preserve bits, not perform BF16 arithmetic.
            let patterns = [0x0000_u16, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0x7fc1, 0xffff];
            let source_bytes: Vec<u8> = (0..rows * cols)
                .flat_map(|i| patterns[i % patterns.len()].to_le_bytes())
                .collect();
            let mut source = CudaBf16Matrix::allocate(
                rows,
                cols,
                Arc::clone(&stream),
                &pool,
                source_bytes.len(),
            )
            .unwrap();
            source.upload(&source_bytes).unwrap();
            let mut cache =
                CudaBf16Matrix::allocate(7, cols, Arc::clone(&stream), &pool, 14 * cols).unwrap();
            let mut expected = vec![0x55; 14 * cols];
            cache.upload(&expected).unwrap();
            for start in [0, 1, 7 - rows] {
                cache.copy_rows_from(start, &source).unwrap();
                expected[start * cols * 2..(start + rows) * cols * 2]
                    .copy_from_slice(&source_bytes);
                assert_eq!(cache.read_chunk(0, expected.len()).unwrap(), expected);
                assert_eq!(
                    source.read_chunk(0, source_bytes.len()).unwrap(),
                    source_bytes
                );
            }
            for start in [8 - rows, usize::MAX] {
                let error = cache.copy_rows_from(start, &source).unwrap_err();
                assert!(!error.retirement_pending());
                assert_eq!(cache.read_chunk(0, expected.len()).unwrap(), expected);
            }
            let mut foreign =
                CudaBf16Matrix::allocate(rows, cols, context.new_stream().unwrap(), &pool, 1)
                    .unwrap();
            foreign.upload(&source_bytes).unwrap();
            let error = cache.copy_rows_from(0, &foreign).unwrap_err();
            assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
            assert!(!error.retirement_pending());
            assert_eq!(cache.read_chunk(0, expected.len()).unwrap(), expected);
            foreign.close().unwrap();
            source.close().unwrap();
            cache.close().unwrap();
            assert_eq!(pool.granted(), 0);
        }
    }
}

#[test]
#[ignore = "requires CUDA; serialize device qualification"]
fn partial_cache_cannot_publish_or_read_uninitialized_rows() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(1024 * 1024).shared();
    let mut source = CudaBf16Matrix::allocate(1, 7, Arc::clone(&stream), &pool, 14).unwrap();
    let mut keys = CudaBf16Matrix::allocate(3, 7, Arc::clone(&stream), &pool, 42).unwrap();
    let mut values = CudaBf16Matrix::allocate(3, 7, Arc::clone(&stream), &pool, 42).unwrap();
    let error = keys.copy_rows_from(0, &source).unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::NotReady));
    source.upload(&[0; 14]).unwrap();
    let error = keys.copy_rows_from(1, &source).unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
    for cache in [&mut keys, &mut values] {
        cache.copy_rows_from(0, &source).unwrap();
        assert!(!cache.ready(), "one row is not the complete cache");
        assert_eq!(cache.read_chunk(0, 14).unwrap(), [0; 14]);
        assert!(cache.read_chunk(14, 2).is_err());
        assert!(
            cache.copy_rows_from(2, &source).is_err(),
            "cannot leave a hole"
        );
    }
    // A partial cache is not a valid complete source, even though its first row
    // is readable. Nor may attention consume a later uninitialized row.
    let mut other = CudaBf16Matrix::allocate(3, 7, Arc::clone(&stream), &pool, 42).unwrap();
    let error = other.copy_rows_from(0, &keys).unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::NotReady));
    let mut output = CudaBf16Matrix::allocate(1, 7, Arc::clone(&stream), &pool, 14).unwrap();
    output.upload(&[0x55; 14]).unwrap();
    let mut attention = CudaBf16Attention::new(Arc::clone(&stream), &pool, 3).unwrap();
    let geometry = AttentionGeometry {
        heads: 1,
        kv_heads: 1,
        width: 7,
        window: None,
    };
    let error = attention
        .attend_into(
            &source,
            &keys,
            &values,
            geometry,
            AttentionSpan {
                query_start: 1,
                key_start: 0,
                key_sequence: 2,
            },
            &mut output,
        )
        .unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::NotReady));
    assert!(!error.retirement_pending());
    assert_eq!(output.read_chunk(0, 14).unwrap(), [0x55; 14]);
    attention
        .attend_into(
            &source,
            &keys,
            &values,
            geometry,
            AttentionSpan {
                query_start: 0,
                key_start: 0,
                key_sequence: 1,
            },
            &mut output,
        )
        .unwrap();
    assert_eq!(output.read_chunk(0, 14).unwrap(), [0; 14]);
    for cache in [&mut keys, &mut values] {
        for start in 1..3 {
            cache.copy_rows_from(start, &source).unwrap();
        }
        assert!(cache.ready());
        assert_eq!(cache.read_chunk(0, 42).unwrap(), [0; 42]);
        cache.close().unwrap();
        assert!(cache.copy_rows_from(0, &source).is_err());
    }
    attention.close().unwrap();
    source.close().unwrap();
    other.close().unwrap();
    output.close().unwrap();
    assert_eq!(pool.granted(), 0);
}
