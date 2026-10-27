//! Fixed native RMS → Gate/Up → exact lookup/product → Down/RMS chain.
use super::{
    LinearGeometry, MetalLinearShape, MetalProjectionNorm, ProjectionGeometry, RmsGeometry,
};
use crate::bf16::{MetalBf16Buffer, MetalBf16Error, MetalBf16Executor, THREADS};
use metal::{BufferRef, ComputeCommandEncoderRef, MTLSize};

/// Immutable bindings for a native gated FFN; activation semantics belong to the table's model.
#[derive(Clone, Copy)]
pub struct MetalFfnWeights<'a> {
    pub gate: &'a MetalBf16Buffer,
    pub up: &'a MetalBf16Buffer,
    pub down: &'a MetalBf16Buffer,
    pub table: &'a MetalBf16Buffer,
    pub input_scale: &'a MetalBf16Buffer,
    pub output_scale: &'a MetalBf16Buffer,
}

impl MetalBf16Executor {
    /// Qualified automatic model selection, not a general memory grant.
    #[must_use]
    pub fn native_ffn_supported(&self, rows: usize, hidden: usize, intermediate: usize) -> bool {
        self.device.name() == "Apple M3 Max"
            && self.ffn_shape.is_none()
            && (1..=64).contains(&rows)
            && hidden == 2560
            && intermediate == 10240
    }

    /// Preserve all six native dispatches and BF16 boundaries in one serial command.
    ///
    /// Shape/scale/table validation precedes every allocation. Inputs, immutable
    /// table and all private allocations remain retained through terminal completion.
    /// The model supplies all 65536 table entries; non-finite gate encodings reject
    /// after completion rather than indexing the model's unused non-finite slots.
    /// # Errors
    /// Rejects invalid complete geometry/table, allocation failure, unsuccessful
    /// execution or a non-finite gate. No partial output is published.
    pub fn native_ffn(
        &self,
        input: &[u16],
        weights: MetalFfnWeights<'_>,
        shape: MetalLinearShape,
        epsilon: f32,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        let gate = LinearGeometry::new(
            shape,
            input.len(),
            weights.gate.elements,
            self.block_linear_rows,
        )?;
        let up = LinearGeometry::new(
            shape,
            input.len(),
            weights.up.elements,
            self.block_linear_rows,
        )?;
        let rms = RmsGeometry::new(
            input.len(),
            shape.rows,
            shape.inner,
            epsilon,
            Some(weights.input_scale),
        )?;
        if weights.table.elements != 1 << 16 {
            return Err(MetalBf16Error::Shape(
                "FFN activation table must have 65536 entries".into(),
            ));
        }
        let down = ProjectionGeometry::new(
            MetalLinearShape {
                rows: shape.rows,
                columns: shape.inner,
                inner: shape.columns,
            },
            gate.elements(),
            weights.down.elements,
            MetalProjectionNorm {
                width: shape.inner,
                epsilon,
                scale: Some(weights.output_scale),
                rotary: None,
            },
            self.block_linear_rows,
        )?;
        let input = self.upload(input)?;
        let normalized = self.allocate_shared(input.elements * 2)?;
        let gate_output = self.allocate_shared(gate.elements() * 2)?;
        let up_output = self.allocate_shared(up.elements() * 2)?;
        let product = self.allocate_shared(gate.elements() * 2)?;
        let down = down.allocate(self)?;
        let invalid = self.upload(&[0, 0])?;
        let mut command = self.command()?;
        let encoder = Self::compute_encoder(&command)?;
        self.encode_rms(
            &encoder,
            &input.buffer,
            &normalized,
            Some(weights.input_scale),
            &rms,
        );
        self.encode_linear(
            &encoder,
            &normalized,
            &weights.gate.buffer,
            &gate_output,
            &gate,
        );
        self.encode_linear(&encoder, &normalized, &weights.up.buffer, &up_output, &up);
        self.encode_lookup_product(
            &encoder,
            &gate_output,
            &up_output,
            &weights.table.buffer,
            &product,
            &invalid.buffer,
            gate.total,
        );
        down.encode(self, &encoder, &product, &weights.down.buffer);
        encoder.end_encoding();
        command.submit("native gated FFN")?;
        if invalid.as_slice().iter().any(|&word| word != 0) {
            return Err(MetalBf16Error::Invalid("FFN gate input must be finite"));
        }
        Ok(down.into_output())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "one fixed pointwise dispatch's explicit bindings"
    )]
    fn encode_lookup_product(
        &self,
        encoder: &ComputeCommandEncoderRef,
        gate: &BufferRef,
        up: &BufferRef,
        table: &BufferRef,
        output: &BufferRef,
        invalid: &BufferRef,
        total: u32,
    ) {
        encoder.set_compute_pipeline_state(&self.lookup_product);
        for (slot, buffer) in [gate, up, table, output, invalid].into_iter().enumerate() {
            encoder.set_buffer(slot as u64, Some(buffer), 0);
        }
        encoder.set_bytes(5, 4, (&raw const total).cast());
        let grid = u64::from(total);
        encoder.dispatch_threads(
            MTLSize::new(grid, 1, 1),
            MTLSize::new(grid.min(THREADS), 1, 1),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bf16(value: f32) -> u16 {
        let b = value.to_bits();
        if b & 0x7fff_ffff > 0x7f80_0000 {
            0x7fc0
        } else {
            ((b.wrapping_add(0x7fff).wrapping_add((b >> 16) & 1)) >> 16) as u16
        }
    }
    #[test]
    fn lookup_product_preserves_cpu_rounding_including_subnormals_and_gate_rejection() {
        let executor = match MetalBf16Executor::new() {
            Ok(e) => e,
            Err(MetalBf16Error::Device) => return,
            Err(e) => panic!("{e}"),
        };
        let table: Vec<u16> = (0..=u16::MAX).collect();
        let table = executor.upload(&table).unwrap();
        let input: Vec<u16> = (0..=u16::MAX).filter(|&b| b & 0x7f80 != 0x7f80).collect();
        let gate = executor.upload(&input).unwrap();
        for up in [
            0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x0080, 0x8080, 0x0081, 0x3f00, 0xbf00,
            0x3f80, 0xbf80, 0x3f81, 0x3fc0, 0xbfc0, 0x3fff, 0x4000, 0xc000, 0x7f7f, 0xff7f, 0x7f80,
            0xff80, 0x7fc1,
        ] {
            let other = executor.upload(&vec![up; input.len()]).unwrap();
            let output = executor.allocate_shared(input.len() * 2).unwrap();
            let invalid = executor.upload(&[0, 0]).unwrap();
            let mut command = executor.command().unwrap();
            let encoder = MetalBf16Executor::compute_encoder(&command).unwrap();
            executor.encode_lookup_product(
                &encoder,
                &gate.buffer,
                &other.buffer,
                &table.buffer,
                &output,
                &invalid.buffer,
                u32::try_from(input.len()).unwrap(),
            );
            encoder.end_encoding();
            command.submit("exact lookup/product test").unwrap();
            let output = MetalBf16Buffer {
                buffer: output,
                elements: input.len(),
            };
            assert!(invalid.as_slice().iter().all(|&v| v == 0));
            for (&input, &actual) in input.iter().zip(output.as_slice()) {
                assert_eq!(
                    actual,
                    bf16(
                        f32::from_bits(u32::from(input) << 16)
                            * f32::from_bits(u32::from(up) << 16)
                    ),
                    "{input:04x} × {up:04x}"
                );
            }
        }
        for bits in [0x7f80, 0xff80, 0x7fc0, 0xffc0] {
            let bad = executor.upload(&[bits]).unwrap();
            let other = executor.upload(&[0x3f80]).unwrap();
            let output = executor.allocate_shared(2).unwrap();
            let invalid = executor.upload(&[0, 0]).unwrap();
            let mut command = executor.command().unwrap();
            let encoder = MetalBf16Executor::compute_encoder(&command).unwrap();
            executor.encode_lookup_product(
                &encoder,
                &bad.buffer,
                &other.buffer,
                &table.buffer,
                &output,
                &invalid.buffer,
                1,
            );
            encoder.end_encoding();
            command.submit("nonfinite gate test").unwrap();
            assert!(invalid.as_slice().iter().any(|&v| v != 0));
        }
        executor.check_healthy().unwrap();
    }

    #[test]
    fn native_chain_matches_unfused_calls_across_row_kernels() {
        let e = match MetalBf16Executor::new() {
            Ok(e) => e,
            Err(MetalBf16Error::Device) => return,
            Err(e) => panic!("{e}"),
        };
        let gate = e
            .upload(
                &(0..32)
                    .map(|i| bf16((f32::from(u16::try_from(i % 7).unwrap()) - 3.0) * 0.125))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let up = e
            .upload(
                &(0..32)
                    .map(|i| bf16((f32::from(u16::try_from(i % 5).unwrap()) - 2.0) * 0.25))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let down = e
            .upload(
                &(0..32)
                    .map(|i| bf16((f32::from(u16::try_from(i % 3).unwrap()) - 1.0) * 0.125))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let scale = e.upload(&[0x3f80; 4]).unwrap();
        let table = e.upload(&(0..=u16::MAX).collect::<Vec<_>>()).unwrap();
        for rows in [1, 7, 8, 16, 32, 64, 65] {
            let input = e
                .upload(
                    &(0..rows * 4)
                        .map(|i| [0x3f00, 0x3f80, 0xbe80, 0xbf80][i % 4])
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let shape = MetalLinearShape {
                rows,
                columns: 8,
                inner: 4,
            };
            let (a, b) = e
                .rms_linear_pair(&input, &gate, &up, shape, 1e-6, Some(&scale))
                .unwrap();
            let product: Vec<_> = a
                .as_slice()
                .iter()
                .zip(b.as_slice())
                .map(|(&a, &b)| {
                    bf16(f32::from_bits(u32::from(a) << 16) * f32::from_bits(u32::from(b) << 16))
                })
                .collect();
            let product = e.upload(&product).unwrap();
            let expected = e
                .project_normalized(
                    &product,
                    &down,
                    MetalLinearShape {
                        rows,
                        columns: 4,
                        inner: 8,
                    },
                    MetalProjectionNorm {
                        width: 4,
                        epsilon: 1e-6,
                        scale: Some(&scale),
                        rotary: None,
                    },
                )
                .unwrap();
            let actual = e
                .native_ffn(
                    input.as_slice(),
                    MetalFfnWeights {
                        gate: &gate,
                        up: &up,
                        down: &down,
                        table: &table,
                        input_scale: &scale,
                        output_scale: &scale,
                    },
                    shape,
                    1e-6,
                )
                .unwrap();
            assert_eq!(actual.as_slice(), expected.as_slice(), "{rows} rows");
        }
        e.check_healthy().unwrap();
    }

    #[test]
    fn native_chain_rejects_invalid_input_without_poisoning_owner() {
        let e = match MetalBf16Executor::new() {
            Ok(e) => e,
            Err(MetalBf16Error::Device) => return,
            Err(e) => panic!("{e}"),
        };
        let table = e.upload(&(0..=u16::MAX).collect::<Vec<_>>()).unwrap();
        let one = e.upload(&[0x3f80]).unwrap();
        let inf = e.upload(&[0x7f80]).unwrap();
        let shape = MetalLinearShape {
            rows: 1,
            columns: 1,
            inner: 1,
        };
        let invalid = e.native_ffn(
            one.as_slice(),
            MetalFfnWeights {
                gate: &inf,
                up: &one,
                down: &one,
                table: &table,
                input_scale: &one,
                output_scale: &one,
            },
            shape,
            1e-6,
        );
        assert!(matches!(invalid, Err(MetalBf16Error::Invalid(_))));
        let bad_table = e.native_ffn(
            one.as_slice(),
            MetalFfnWeights {
                gate: &one,
                up: &one,
                down: &one,
                table: &one,
                input_scale: &one,
                output_scale: &one,
            },
            shape,
            1e-6,
        );
        assert!(matches!(bad_table, Err(MetalBf16Error::Shape(_))));
        let wrong_down = e.upload(&[0x3f80; 2]).unwrap();
        let invalid = e.native_ffn(
            one.as_slice(),
            MetalFfnWeights {
                gate: &one,
                up: &one,
                down: &wrong_down,
                table: &table,
                input_scale: &one,
                output_scale: &one,
            },
            shape,
            1e-6,
        );
        assert!(matches!(invalid, Err(MetalBf16Error::Shape(_))));
        e.native_ffn(
            one.as_slice(),
            MetalFfnWeights {
                gate: &one,
                up: &one,
                down: &one,
                table: &table,
                input_scale: &one,
                output_scale: &one,
            },
            shape,
            1e-6,
        )
        .unwrap();
        assert!(!e.native_ffn_supported(0, 2560, 10240));
        assert!(!e.native_ffn_supported(65, 2560, 10240));
        assert!(!e.native_ffn_supported(16, 2561, 10240));
        assert!(!e.native_ffn_supported(16, 2560, 10241));
        e.check_healthy().unwrap();
    }
}
