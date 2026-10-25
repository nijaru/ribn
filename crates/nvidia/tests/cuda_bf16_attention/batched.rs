//! Same-operand row batching preserves per-query arithmetic and visible-key order.
use super::*;

#[test]
#[ignore = "requires CUDA; eight/sixteen-row attention through 4K reach; serialize device runs"]
fn prepared_rows_match_scalar_queries_at_window_and_capacity_boundaries() {
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let pool = BytePool::new(64 << 20).shared();
    let mut attention = CudaBf16Attention::new(Arc::clone(&stream), &pool, 16 * 8 * 4096).unwrap();
    for rows in [8, 16] {
        for (width, window) in [(256, Some(512)), (512, None)] {
            for keys in [rows, 513, 4096] {
                let geometry = AttentionGeometry {
                    heads: 8,
                    kv_heads: 2,
                    width,
                    window,
                };
                let columns = 8 * width;
                let upload_full = |values: Vec<u16>, rows, cols| {
                    let bytes = encode(&values);
                    let mut matrix = CudaBf16Matrix::allocate(
                        rows,
                        cols,
                        Arc::clone(&stream),
                        &pool,
                        bytes.len(),
                    )
                    .unwrap();
                    matrix.upload(&bytes).unwrap();
                    matrix
                };
                let mut q = upload_full(operands(11, rows * columns), rows, columns);
                let mut k = upload_full(operands(29, keys * 2 * width), keys, 2 * width);
                let mut v = upload_full(operands(47, keys * 2 * width), keys, 2 * width);
                let mut batch = CudaBf16Matrix::allocate(
                    rows,
                    columns,
                    Arc::clone(&stream),
                    &pool,
                    rows * columns * 2,
                )
                .unwrap();
                let mut one_query =
                    CudaBf16Matrix::allocate(1, columns, Arc::clone(&stream), &pool, 1).unwrap();
                let mut one_output =
                    CudaBf16Matrix::allocate(1, columns, Arc::clone(&stream), &pool, columns * 2)
                        .unwrap();
                attention
                    .attend_into(
                        &q,
                        &k,
                        &v,
                        geometry,
                        AttentionSpan {
                            query_start: keys - rows,
                            key_start: 0,
                            key_sequence: keys,
                        },
                        &mut batch,
                    )
                    .unwrap();
                let actual = batch.read_chunk(0, rows * columns * 2).unwrap();
                for row in 0..rows {
                    one_query.copy_rows_from(0, &q, row..row + 1).unwrap();
                    let position = keys - rows + row;
                    attention
                        .attend_into(
                            &one_query,
                            &k,
                            &v,
                            geometry,
                            AttentionSpan {
                                query_start: position,
                                key_start: 0,
                                key_sequence: position + 1,
                            },
                            &mut one_output,
                        )
                        .unwrap();
                    let expected = one_output.read_chunk(0, columns * 2).unwrap();
                    assert_eq!(
                        &actual[row * columns * 2..(row + 1) * columns * 2],
                        expected,
                        "width={width} keys={keys} row={row}"
                    );
                    assert!(
                        expected
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .all(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 != 0x7f80)
                    );
                }
                q.close().unwrap();
                k.close().unwrap();
                v.close().unwrap();
                batch.close().unwrap();
                one_query.close().unwrap();
                one_output.close().unwrap();
            }
        }
    }
    attention.close().unwrap();
    assert_eq!(pool.granted(), 0);
}
