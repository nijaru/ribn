//! Logical-role/backend integration; no full Gemma forward-pass claim.
#![cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
use engine_gemma4::{Gemma4CudaWeights, Gemma4Source, LayerParameter, ParameterId};
use engine_nvidia::{CudaBf16Gemm, CudaBf16Matrix};
use ribn_foundation::BytePool;
use std::{io::Read, sync::Arc};
#[path = "common/weights.rs"]
mod weights;
use weights::{Dir, inventory, package};

fn round_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}
#[test]
#[ignore = "requires CUDA; serialize native BF16 logical projection qualification"]
fn logical_projections_use_selected_native_weights_including_the_tied_head() {
    let dir = Dir::new();
    let package = package(&dir, &inventory(), false);
    let source = Gemma4Source::open(&package).unwrap();
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let pool = BytePool::new(32 * 1024 * 1024).shared();
    let mut model = Gemma4CudaWeights::load(&source, Arc::clone(&stream), &pool, 5).unwrap();
    let weight_allocation = model.allocation_id();
    let mut execution = CudaBf16Gemm::new(Arc::clone(&stream), &pool, 8 * 1024 * 1024).unwrap();
    for role in [
        ParameterId::TokenEmbedding,
        ParameterId::PleProjection,
        ParameterId::Layer {
            layer: 0,
            parameter: LayerParameter::Query,
        },
        ParameterId::Layer {
            layer: 3,
            parameter: LayerParameter::Down,
        },
    ] {
        let selected = &source.parameters()[&role];
        let shape = &selected.spec().shape;
        let (n, k) = (shape[0], shape[1]);
        let mut encoded = Vec::new();
        selected.reader().read_to_end(&mut encoded).unwrap();
        let reference: Vec<_> = encoded
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect();
        let rows = 3;
        let input: Vec<_> = (0..rows)
            .flat_map(|row| {
                (0..k).map(move |j| f32::from(u8::try_from((row * 3 + j) % 9).unwrap()) / 8.0 - 0.5)
            })
            .collect();
        let input_bytes: Vec<_> = input
            .iter()
            .flat_map(|&v| round_bf16(v).to_le_bytes())
            .collect();
        let mut x = CudaBf16Matrix::allocate(rows, k, Arc::clone(&stream), &pool, 5).unwrap();
        let mut y = CudaBf16Matrix::allocate(rows, n, Arc::clone(&stream), &pool, 5).unwrap();
        x.upload(&input_bytes).unwrap();
        model
            .linear_into(role, &mut execution, &mut x, &mut y)
            .unwrap();
        let mut output = Vec::new();
        for offset in (0..rows * n * 2).step_by(5) {
            output.extend(
                y.read_chunk(offset as u64, 5.min(rows * n * 2 - offset))
                    .unwrap(),
            );
        }
        for row in 0..rows {
            for out in 0..n {
                // Tiny fixture weights and inputs are dyadic rationals whose
                // products/partial sums fit FP32 exactly. FP64 independently
                // sums the selected source, then applies one BF16 RNE conversion.
                let value: f64 = (0..k)
                    .map(|j| f64::from(input[row * k + j]) * f64::from(reference[out * k + j]))
                    .sum();
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "tiny fixture dot products are exactly representable FP32 dyadic rationals"
                )]
                let exact = value as f32;
                assert_eq!(f64::from(exact).to_bits(), value.to_bits());
                let expected = round_bf16(exact);
                let index = 2 * (row * n + out);
                let got = u16::from_le_bytes([output[index], output[index + 1]]);
                assert_ne!(got & 0x7f80, 0x7f80, "nonfinite projection");
                assert_eq!(got, expected, "{role:?} row={row} out={out}");
            }
        }
        x.close().unwrap();
        y.close().unwrap();
        assert_eq!(
            model.allocation_id(),
            weight_allocation,
            "projection must not repack weights"
        );
        assert_eq!(model.output_projection_slot(), model.token_embedding_slot());
    }
    execution.close().unwrap();
    model.close().unwrap();
    assert_eq!(pool.granted(), 0);
}
