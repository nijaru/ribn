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
    for cols in [7, 256, 512, 2560] {
        for rows in [1, 3, 8] {
            // Raw encodings deliberately include signed zero, infinity and NaN
            // payloads: a copy must preserve bits, not perform BF16 arithmetic.
            let patterns = [0x0000_u16, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0x7fc1, 0xffff];
            let source_bytes: Vec<u8> = (0..rows * cols)
                .flat_map(|i| {
                    let row = i / cols;
                    let bits = if row > 0 && i % cols == 0 {
                        0x3f00 + u16::try_from(row).unwrap()
                    } else {
                        patterns[i % patterns.len()]
                    };
                    bits.to_le_bytes()
                })
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
                CudaBf16Matrix::allocate(12, cols, Arc::clone(&stream), &pool, 24 * cols).unwrap();
            let mut expected = vec![0x55; 24 * cols];
            cache.upload(&expected).unwrap();
            let ranges = [0..rows, rows - 1..rows, 1..rows];
            let range_count = if rows == 1 { 1 } else { ranges.len() };
            for range in &ranges[..range_count] {
                let count = range.len();
                for start in [0, 1, 12 - count] {
                    cache.copy_rows_from(start, &source, range.clone()).unwrap();
                    expected[start * cols * 2..(start + count) * cols * 2].copy_from_slice(
                        &source_bytes[range.start * cols * 2..range.end * cols * 2],
                    );
                    assert_eq!(cache.read_chunk(0, expected.len()).unwrap(), expected);
                    assert_eq!(
                        source.read_chunk(0, source_bytes.len()).unwrap(),
                        source_bytes
                    );
                }
            }
            let mut selected =
                CudaBf16Matrix::allocate(1, cols, Arc::clone(&stream), &pool, cols * 2).unwrap();
            selected.copy_rows_from(0, &source, rows - 1..rows).unwrap();
            assert!(selected.ready());
            assert_eq!(
                selected.read_chunk(0, cols * 2).unwrap(),
                source_bytes[(rows - 1) * cols * 2..]
            );
            selected.close().unwrap();
            for (begin, end, start) in [
                (0, rows, 13 - rows),
                (0, rows, usize::MAX),
                (0, 0, 0),
                (1, 0, 0),
                (rows, rows + 1, 0),
                (0, usize::MAX, 0),
            ] {
                let error = cache
                    .copy_rows_from(start, &source, begin..end)
                    .unwrap_err();
                assert!(!error.retirement_pending());
                assert_eq!(cache.read_chunk(0, expected.len()).unwrap(), expected);
            }
            let mut foreign =
                CudaBf16Matrix::allocate(rows, cols, context.new_stream().unwrap(), &pool, 1)
                    .unwrap();
            foreign.upload(&source_bytes).unwrap();
            let error = cache.copy_rows_from(0, &foreign, 0..rows).unwrap_err();
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
    let error = keys.copy_rows_from(0, &source, 0..1).unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::NotReady));
    source.upload(&[0; 14]).unwrap();
    let error = keys.copy_rows_from(1, &source, 0..1).unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::Invalid(_)));
    for cache in [&mut keys, &mut values] {
        cache.copy_rows_from(0, &source, 0..1).unwrap();
        assert!(!cache.ready(), "one row is not the complete cache");
        assert_eq!(cache.read_chunk(0, 14).unwrap(), [0; 14]);
        assert!(cache.read_chunk(14, 2).is_err());
        assert!(
            cache.copy_rows_from(2, &source, 0..1).is_err(),
            "cannot leave a hole"
        );
    }
    // A partial source may supply its initialized row, but neither a copy nor
    // attention may consume the uninitialized tail.
    let mut other = CudaBf16Matrix::allocate(3, 7, Arc::clone(&stream), &pool, 42).unwrap();
    let error = other.copy_rows_from(0, &keys, 0..3).unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::NotReady));
    assert!(!other.ready());
    other.copy_rows_from(0, &keys, 0..1).unwrap();
    assert!(!other.ready());
    assert_eq!(other.read_chunk(0, 14).unwrap(), [0; 14]);
    let error = other.copy_rows_from(0, &keys, 1..2).unwrap_err();
    assert!(matches!(error.cause(), CudaBf16Error::NotReady));
    assert!(!error.retirement_pending());
    assert_eq!(other.read_chunk(0, 14).unwrap(), [0; 14]);
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
            cache.copy_rows_from(start, &source, 0..1).unwrap();
        }
        assert!(cache.ready());
        assert_eq!(cache.read_chunk(0, 42).unwrap(), [0; 42]);
        cache.close().unwrap();
        assert!(cache.copy_rows_from(0, &source, 0..1).is_err());
    }
    attention.close().unwrap();
    source.close().unwrap();
    other.close().unwrap();
    output.close().unwrap();
    assert_eq!(pool.granted(), 0);
}
