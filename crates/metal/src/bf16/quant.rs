//! Affine group-64 quantized Metal weights (Q4 & Q8) and single-command-buffer kernels.
//!
//! Matches MLX's standard affine group-64 bit layout (`w = q * scale + bias` with
//! per-64-element BF16 `scale` and `bias`), streaming directly from BF16 readers
//! into immutable shared Metal storage with bounded ~8 MiB staging scratch.

use super::fast::{MetalDeviceBf16, MetalDeviceU32, MetalFastSession};
use super::{MetalBf16Buffer, MetalBf16Error, MetalBf16Executor};
use metal::{Buffer, CompileOptions, ComputePipelineState, Device, MTLLanguageVersion, MTLSize};
use std::io::Read;

/// Group size along the inner (`K`) dimension for affine Metal quantization.
pub const METAL_QUANT_GROUP_SIZE: usize = 64;

const QUANT_SOURCE: &str = include_str!("../quant_fast.metal");
const STAGING_TARGET_BYTES: usize = 8 * 1024 * 1024;

/// Affine group-64 quantized weight precision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalQuantKind {
    /// 8-bit affine group-64 (1.0625 bytes/weight; 1.88x compression vs BF16).
    Q8,
    /// 4-bit affine group-64 (0.5625 bytes/weight; 3.56x compression vs BF16).
    Q4,
}

/// Immutable shared-storage affine group-64 quantized weight matrix `[rows, cols]`.
pub struct MetalQuantWeight {
    kind: MetalQuantKind,
    rows: usize,
    cols: usize,
    packed: Buffer,
    scales_biases: Buffer,
    bytes: usize,
}

impl MetalQuantWeight {
    #[must_use]
    pub const fn kind(&self) -> MetalQuantKind {
        self.kind
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    #[must_use]
    pub const fn elements(&self) -> usize {
        self.rows * self.cols
    }

    /// Total shared GPU storage in bytes (`packed + scales + biases`).
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes
    }
}

pub(super) struct QuantPipelines {
    q4_gemv_m1: ComputePipelineState,
    q4_gemv_m2: ComputePipelineState,
    q4_gemv_m4: ComputePipelineState,
    q4_gemm_m8: ComputePipelineState,
    q4_gemm_m16: ComputePipelineState,
    q4_gemm_m32: ComputePipelineState,
    q4_embed_gather: ComputePipelineState,
    q4_dual_gemv_m1: ComputePipelineState,
    q4_gate_up_geglu_m1: ComputePipelineState,
    q4_linear_geglu_m1: ComputePipelineState,
    q8_gemv_m1: ComputePipelineState,
    q8_gemv_m2: ComputePipelineState,
    q8_gemv_m4: ComputePipelineState,
    q8_gemm_m8: ComputePipelineState,
    q8_gemm_m16: ComputePipelineState,
    q8_gemm_m32: ComputePipelineState,
    q8_embed_gather: ComputePipelineState,
    q8_dual_gemv_m1: ComputePipelineState,
    q8_gate_up_geglu_m1: ComputePipelineState,
    q8_linear_geglu_m1: ComputePipelineState,
}

impl QuantPipelines {
    pub(super) fn compile(device: &Device) -> Result<Self, MetalBf16Error> {
        let options = CompileOptions::new();
        options.set_language_version(MTLLanguageVersion::V3_1);
        let library = device
            .new_library_with_source(QUANT_SOURCE, &options)
            .map_err(MetalBf16Error::Library)?;
        let load = |name: &str| -> Result<ComputePipelineState, MetalBf16Error> {
            let func = library
                .get_function(name, None)
                .map_err(MetalBf16Error::Function)?;
            device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(MetalBf16Error::Pipeline)
        };
        Ok(Self {
            q4_gemv_m1: load("fast_q4_gemv_m1")?,
            q4_gemv_m2: load("fast_q4_gemv_m2")?,
            q4_gemv_m4: load("fast_q4_gemv_m4")?,
            q4_gemm_m8: load("fast_q4_gemm_m8")?,
            q4_gemm_m16: load("fast_q4_gemm_m16")?,
            q4_gemm_m32: load("fast_q4_gemm_m32")?,
            q4_embed_gather: load("fast_q4_embed_gather")?,
            q4_dual_gemv_m1: load("fast_q4_dual_gemv_m1")?,
            q4_gate_up_geglu_m1: load("fast_q4_gate_up_geglu_m1")?,
            q4_linear_geglu_m1: load("fast_q4_linear_geglu_m1")?,
            q8_gemv_m1: load("fast_q8_gemv_m1")?,
            q8_gemv_m2: load("fast_q8_gemv_m2")?,
            q8_gemv_m4: load("fast_q8_gemv_m4")?,
            q8_gemm_m8: load("fast_q8_gemm_m8")?,
            q8_gemm_m16: load("fast_q8_gemm_m16")?,
            q8_gemm_m32: load("fast_q8_gemm_m32")?,
            q8_embed_gather: load("fast_q8_embed_gather")?,
            q8_dual_gemv_m1: load("fast_q8_dual_gemv_m1")?,
            q8_gate_up_geglu_m1: load("fast_q8_gate_up_geglu_m1")?,
            q8_linear_geglu_m1: load("fast_q8_linear_geglu_m1")?,
        })
    }
}

#[inline]
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "top 16 bits of IEEE-754 f32 always fit in u16"
)]
fn f32_to_bf16_rne(value: f32) -> u16 {
    let mut bits = value.to_bits();
    if (bits & 0x7f80_0000) == 0x7f80_0000 {
        return if (bits & 0x007f_ffff) != 0 {
            ((bits >> 16) | 0x0040) as u16
        } else {
            (bits >> 16) as u16
        };
    }
    bits = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (bits >> 16) as u16
}

fn quantize_rows_q4(
    src_bytes: &[u8],
    cols: usize,
    packed_out: &mut [u32],
    sb_out: &mut [[u16; 2]],
) -> Result<(), MetalBf16Error> {
    let groups_per_row = cols / METAL_QUANT_GROUP_SIZE;
    let words_per_row = cols / 8;
    let row_bytes = cols * 2;
    let rows = src_bytes.len() / row_bytes;
    let mut vals = [0.0_f32; METAL_QUANT_GROUP_SIZE];

    for r in 0..rows {
        let row_src = &src_bytes[r * row_bytes..(r + 1) * row_bytes];
        let row_packed = &mut packed_out[r * words_per_row..(r + 1) * words_per_row];
        let row_sb = &mut sb_out[r * groups_per_row..(r + 1) * groups_per_row];

        for (g, sb_slot) in row_sb.iter_mut().enumerate() {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            let base = g * METAL_QUANT_GROUP_SIZE * 2;
            for (i, slot) in vals.iter_mut().enumerate() {
                let bits = u16::from_le_bytes([row_src[base + i * 2], row_src[base + i * 2 + 1]]);
                let v = bf16_to_f32(bits);
                if !v.is_finite() {
                    return Err(MetalBf16Error::Invalid(
                        "non-finite BF16 weight during Q4 quantization",
                    ));
                }
                *slot = v;
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let span = hi - lo;
            let scale_bits = f32_to_bf16_rne(if span == 0.0 { 0.0 } else { span / 15.0 });
            let bias_bits = f32_to_bf16_rne(lo);
            let s = bf16_to_f32(scale_bits);
            let b = bf16_to_f32(bias_bits);
            let inv_s = if s == 0.0 { 0.0 } else { 1.0 / s };
            *sb_slot = [scale_bits, bias_bits];

            let w_base = g * 8;
            for w in 0..8 {
                let mut word = 0_u32;
                for nib in 0..8 {
                    let v = vals[w * 8 + nib];
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "clamped to [0.0, 15.0] before integer cast"
                    )]
                    let q = ((v - b) * inv_s).round().clamp(0.0, 15.0) as u32;
                    word |= q << (nib * 4);
                }
                row_packed[w_base + w] = word;
            }
        }
    }
    Ok(())
}

fn quantize_rows_q8(
    src_bytes: &[u8],
    cols: usize,
    packed_out: &mut [u8],
    sb_out: &mut [[u16; 2]],
) -> Result<(), MetalBf16Error> {
    let groups_per_row = cols / METAL_QUANT_GROUP_SIZE;
    let row_bytes = cols * 2;
    let rows = src_bytes.len() / row_bytes;
    let mut vals = [0.0_f32; METAL_QUANT_GROUP_SIZE];

    for r in 0..rows {
        let row_src = &src_bytes[r * row_bytes..(r + 1) * row_bytes];
        let row_packed = &mut packed_out[r * cols..(r + 1) * cols];
        let row_sb = &mut sb_out[r * groups_per_row..(r + 1) * groups_per_row];

        for (g, sb_slot) in row_sb.iter_mut().enumerate() {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            let base = g * METAL_QUANT_GROUP_SIZE * 2;
            for (i, slot) in vals.iter_mut().enumerate() {
                let bits = u16::from_le_bytes([row_src[base + i * 2], row_src[base + i * 2 + 1]]);
                let v = bf16_to_f32(bits);
                if !v.is_finite() {
                    return Err(MetalBf16Error::Invalid(
                        "non-finite BF16 weight during Q8 quantization",
                    ));
                }
                *slot = v;
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let span = hi - lo;
            let scale_bits = f32_to_bf16_rne(if span == 0.0 { 0.0 } else { span / 255.0 });
            let bias_bits = f32_to_bf16_rne(lo);
            let s = bf16_to_f32(scale_bits);
            let b = bf16_to_f32(bias_bits);
            let inv_s = if s == 0.0 { 0.0 } else { 1.0 / s };
            *sb_slot = [scale_bits, bias_bits];

            let dst_base = g * METAL_QUANT_GROUP_SIZE;
            for (i, &v) in vals.iter().enumerate() {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "clamped to [0.0, 255.0] before integer cast"
                )]
                let q = ((v - b) * inv_s).round().clamp(0.0, 255.0) as u8;
                row_packed[dst_base + i] = q;
            }
        }
    }
    Ok(())
}

fn checked_u32(value: usize, label: &str) -> Result<u32, MetalBf16Error> {
    u32::try_from(value).map_err(|_| MetalBf16Error::Shape(format!("{label} exceeds u32")))
}

fn quant_layout(
    rows: usize,
    cols: usize,
    kind: MetalQuantKind,
) -> Result<(usize, usize, usize, usize, usize), MetalBf16Error> {
    if rows == 0 || cols == 0 || !cols.is_multiple_of(METAL_QUANT_GROUP_SIZE) {
        return Err(MetalBf16Error::Shape(format!(
            "quantized weight [{rows}, {cols}] requires positive rows and cols divisible by {METAL_QUANT_GROUP_SIZE}"
        )));
    }
    let groups_per_row = cols / METAL_QUANT_GROUP_SIZE;
    let total_groups = rows
        .checked_mul(groups_per_row)
        .ok_or_else(|| MetalBf16Error::Shape("quantized groups overflow".into()))?;
    let packed_bytes = match kind {
        MetalQuantKind::Q4 => rows
            .checked_mul(cols / 2)
            .ok_or_else(|| MetalBf16Error::Shape("Q4 packed bytes overflow".into()))?,
        MetalQuantKind::Q8 => rows
            .checked_mul(cols)
            .ok_or_else(|| MetalBf16Error::Shape("Q8 packed bytes overflow".into()))?,
    };
    let sb_bytes = total_groups
        .checked_mul(4)
        .ok_or_else(|| MetalBf16Error::Shape("quantized scale/bias bytes overflow".into()))?;
    let total_bytes = packed_bytes
        .checked_add(sb_bytes)
        .ok_or_else(|| MetalBf16Error::Shape("quantized total bytes overflow".into()))?;
    Ok((
        groups_per_row,
        total_groups,
        packed_bytes,
        sb_bytes,
        total_bytes,
    ))
}

impl MetalBf16Executor {
    /// Stream a BF16 `[rows, cols]` matrix from `reader` and quantize into an immutable
    /// shared-storage `MetalQuantWeight` (`Q4` or `Q8`, group size 64) using bounded
    /// ~8 MiB host staging and multi-threaded quantization.
    /// # Errors
    /// Rejects non-positive/unaligned dimensions, non-finite weights, short/trailing
    /// reads, or GPU allocation failure.
    #[expect(
        clippy::too_many_lines,
        reason = "streaming bounded-buffer multi-threaded Q4/Q8 quantization with exact EOF checks"
    )]
    pub fn read_quantized_bf16(
        &self,
        rows: usize,
        cols: usize,
        kind: MetalQuantKind,
        reader: &mut dyn Read,
    ) -> Result<MetalQuantWeight, MetalBf16Error> {
        let (groups_per_row, total_groups, packed_bytes, sb_bytes, total_bytes) =
            quant_layout(rows, cols, kind)?;

        let packed = self.allocate_shared(packed_bytes)?;
        let scales_biases = self.allocate_shared(sb_bytes)?;

        let row_bytes = cols
            .checked_mul(2)
            .ok_or_else(|| MetalBf16Error::Shape("BF16 row bytes overflow".into()))?;
        let chunk_rows = (STAGING_TARGET_BYTES / row_bytes).clamp(1, rows);
        let mut staging = vec![0_u8; chunk_rows * row_bytes];
        let workers = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(16);

        // SAFETY: `packed` and `scales_biases` are freshly allocated exclusively owned
        // shared buffers not yet published to any reader or GPU command.
        let sb_all = unsafe {
            std::slice::from_raw_parts_mut(
                scales_biases.contents().cast::<[u16; 2]>(),
                total_groups,
            )
        };

        let mut row_cursor = 0_usize;
        while row_cursor < rows {
            let take = (rows - row_cursor).min(chunk_rows);
            let buf = &mut staging[..take * row_bytes];
            reader.read_exact(buf).map_err(MetalBf16Error::Read)?;

            let sb_slice =
                &mut sb_all[row_cursor * groups_per_row..(row_cursor + take) * groups_per_row];

            let num_threads = workers.min(take).max(1);
            let rows_per_thread = take.div_ceil(num_threads);

            match kind {
                MetalQuantKind::Q4 => {
                    let words_per_row = cols / 8;
                    let total_words = rows * words_per_row;
                    // SAFETY: exclusively owned shared buffer sized for `packed_bytes = total_words * 4`.
                    let packed_all = unsafe {
                        std::slice::from_raw_parts_mut(packed.contents().cast::<u32>(), total_words)
                    };
                    let p_slice = &mut packed_all
                        [row_cursor * words_per_row..(row_cursor + take) * words_per_row];
                    if num_threads == 1 {
                        quantize_rows_q4(buf, cols, p_slice, sb_slice)?;
                    } else {
                        std::thread::scope(|scope| {
                            let mut handles = Vec::with_capacity(num_threads);
                            for ((src_chunk, p_chunk), sb_chunk) in buf
                                .chunks(rows_per_thread * row_bytes)
                                .zip(p_slice.chunks_mut(rows_per_thread * words_per_row))
                                .zip(sb_slice.chunks_mut(rows_per_thread * groups_per_row))
                            {
                                handles.push(scope.spawn(move || {
                                    quantize_rows_q4(src_chunk, cols, p_chunk, sb_chunk)
                                }));
                            }
                            for h in handles {
                                h.join().map_err(|_| {
                                    MetalBf16Error::Invalid("quantizer thread panicked")
                                })??;
                            }
                            Ok::<(), MetalBf16Error>(())
                        })?;
                    }
                }
                MetalQuantKind::Q8 => {
                    // SAFETY: exclusively owned shared buffer sized for `packed_bytes = rows * cols`.
                    let packed_all = unsafe {
                        std::slice::from_raw_parts_mut(packed.contents().cast::<u8>(), rows * cols)
                    };
                    let p_slice = &mut packed_all[row_cursor * cols..(row_cursor + take) * cols];
                    if num_threads == 1 {
                        quantize_rows_q8(buf, cols, p_slice, sb_slice)?;
                    } else {
                        std::thread::scope(|scope| {
                            let mut handles = Vec::with_capacity(num_threads);
                            for ((src_chunk, p_chunk), sb_chunk) in buf
                                .chunks(rows_per_thread * row_bytes)
                                .zip(p_slice.chunks_mut(rows_per_thread * cols))
                                .zip(sb_slice.chunks_mut(rows_per_thread * groups_per_row))
                            {
                                handles.push(scope.spawn(move || {
                                    quantize_rows_q8(src_chunk, cols, p_chunk, sb_chunk)
                                }));
                            }
                            for h in handles {
                                h.join().map_err(|_| {
                                    MetalBf16Error::Invalid("quantizer thread panicked")
                                })??;
                            }
                            Ok::<(), MetalBf16Error>(())
                        })?;
                    }
                }
            }
            row_cursor += take;
        }

        let mut trailing = [0_u8; 1];
        if reader.read(&mut trailing).map_err(MetalBf16Error::Read)? != 0 {
            return Err(MetalBf16Error::Invalid(
                "trailing bytes after quantized BF16 weight",
            ));
        }

        Ok(MetalQuantWeight {
            kind,
            rows,
            cols,
            packed,
            scales_biases,
            bytes: total_bytes,
        })
    }

    /// Stream a pre-quantized affine group-64 `[rows, cols]` weight matrix (`Q4` or `Q8`,
    /// MLX little-endian `U32` packed weights + separate BF16 `scales` and `biases`)
    /// directly into immutable shared-storage Metal buffers with bounded staging.
    /// # Errors
    /// Rejects non-positive/unaligned dimensions, non-finite scales or biases,
    /// short/trailing reads on any stream, or GPU allocation failure.
    pub fn read_prequantized(
        &self,
        rows: usize,
        cols: usize,
        kind: MetalQuantKind,
        packed_reader: &mut dyn Read,
        scales_reader: &mut dyn Read,
        biases_reader: &mut dyn Read,
    ) -> Result<MetalQuantWeight, MetalBf16Error> {
        let (_groups_per_row, total_groups, packed_bytes, sb_bytes, total_bytes) =
            quant_layout(rows, cols, kind)?;

        let packed = self.allocate_shared(packed_bytes)?;
        let scales_biases = self.allocate_shared(sb_bytes)?;

        // SAFETY: `packed` is freshly allocated and exclusively owned with `packed_bytes` bytes.
        let packed_dst =
            unsafe { std::slice::from_raw_parts_mut(packed.contents().cast::<u8>(), packed_bytes) };
        packed_reader
            .read_exact(packed_dst)
            .map_err(MetalBf16Error::Read)?;
        let mut trailing = [0_u8; 1];
        if packed_reader
            .read(&mut trailing)
            .map_err(MetalBf16Error::Read)?
            != 0
        {
            return Err(MetalBf16Error::Invalid(
                "trailing bytes after pre-quantized packed weight",
            ));
        }

        // SAFETY: `scales_biases` is freshly allocated and exclusively owned with `total_groups * 4` bytes.
        let sb_dst = unsafe {
            std::slice::from_raw_parts_mut(
                scales_biases.contents().cast::<[u16; 2]>(),
                total_groups,
            )
        };
        let chunk_groups = (STAGING_TARGET_BYTES / 4).clamp(1, total_groups);
        let mut scale_buf = vec![0_u8; chunk_groups * 2];
        let mut bias_buf = vec![0_u8; chunk_groups * 2];

        let mut cursor = 0_usize;
        while cursor < total_groups {
            let take = (total_groups - cursor).min(chunk_groups);
            let byte_len = take * 2;
            scales_reader
                .read_exact(&mut scale_buf[..byte_len])
                .map_err(MetalBf16Error::Read)?;
            biases_reader
                .read_exact(&mut bias_buf[..byte_len])
                .map_err(MetalBf16Error::Read)?;
            let out_slice = &mut sb_dst[cursor..cursor + take];
            for (i, slot) in out_slice.iter_mut().enumerate() {
                let s_bits = u16::from_le_bytes([scale_buf[i * 2], scale_buf[i * 2 + 1]]);
                let b_bits = u16::from_le_bytes([bias_buf[i * 2], bias_buf[i * 2 + 1]]);
                if (s_bits & 0x7f80) == 0x7f80 || (b_bits & 0x7f80) == 0x7f80 {
                    return Err(MetalBf16Error::Invalid(
                        "non-finite BF16 scale or bias in pre-quantized weight",
                    ));
                }
                *slot = [s_bits, b_bits];
            }
            cursor += take;
        }

        if scales_reader
            .read(&mut trailing)
            .map_err(MetalBf16Error::Read)?
            != 0
        {
            return Err(MetalBf16Error::Invalid(
                "trailing bytes after pre-quantized scales",
            ));
        }
        if biases_reader
            .read(&mut trailing)
            .map_err(MetalBf16Error::Read)?
            != 0
        {
            return Err(MetalBf16Error::Invalid(
                "trailing bytes after pre-quantized biases",
            ));
        }

        Ok(MetalQuantWeight {
            kind,
            rows,
            cols,
            packed,
            scales_biases,
            bytes: total_bytes,
        })
    }

    /// Quantize an in-memory BF16 `[rows, cols]` slice into a `MetalQuantWeight`.
    /// # Errors
    /// Rejects mismatched lengths, unaligned `cols`, non-finite weights, or allocation failure.
    pub fn upload_quantized_bf16(
        &self,
        rows: usize,
        cols: usize,
        kind: MetalQuantKind,
        values: &[u16],
    ) -> Result<MetalQuantWeight, MetalBf16Error> {
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| MetalBf16Error::Shape("quantized upload overflow".into()))?;
        if values.len() != expected {
            return Err(MetalBf16Error::Shape(format!(
                "quantized upload length {} != expected {expected}",
                values.len()
            )));
        }
        let mut bytes = Vec::with_capacity(expected * 2);
        for &bits in values {
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        let mut cursor = bytes.as_slice();
        self.read_quantized_bf16(rows, cols, kind, &mut cursor)
    }
}

impl MetalFastSession<'_> {
    /// Gather `rows` rows of `width` from a quantized embedding `table`, dequantize, and scale.
    /// # Errors
    /// Rejects shape mismatches, non-finite scale, or undersized buffers.
    pub fn embed_gather_quant(
        &mut self,
        token_ids: &MetalDeviceU32,
        table: &MetalQuantWeight,
        output: &MetalDeviceBf16,
        rows: usize,
        width: usize,
        scale: f32,
    ) -> Result<(), MetalBf16Error> {
        if !scale.is_finite() {
            return Err(MetalBf16Error::Invalid(
                "embed_gather_quant scale must be finite",
            ));
        }
        if table.cols != width || !width.is_multiple_of(METAL_QUANT_GROUP_SIZE) {
            return Err(MetalBf16Error::Shape(format!(
                "embed_gather_quant width {width} disagrees with table cols {}",
                table.cols
            )));
        }
        let total = rows
            .checked_mul(width)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("embed_gather_quant overflow".into()))?;
        if token_ids.elements() < rows || output.elements() < total {
            return Err(MetalBf16Error::Shape(
                "embed_gather_quant buffer capacity too small".into(),
            ));
        }
        let total_quads = checked_u32(total / 4, "embed_gather_quant quads")?;
        let width_u32 = checked_u32(width, "embed_gather_quant width")?;

        let pipeline = match table.kind {
            MetalQuantKind::Q4 => &self.executor().quant.q4_embed_gather,
            MetalQuantKind::Q8 => &self.executor().quant.q8_embed_gather,
        };
        let encoder = self.encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(token_ids.raw_buffer()), 0);
        encoder.set_buffer(1, Some(&table.packed), 0);
        encoder.set_buffer(2, Some(&table.scales_biases), 0);
        encoder.set_buffer(3, Some(output.raw_buffer()), 0);
        encoder.set_bytes(4, 4, (&raw const width_u32).cast());
        encoder.set_bytes(5, 4, (&raw const scale).cast());
        encoder.set_bytes(6, 4, (&raw const total_quads).cast());
        let quads_u64 = u64::from(total_quads);
        encoder.dispatch_threads(
            MTLSize::new(quads_u64, 1, 1),
            MTLSize::new(quads_u64.min(256), 1, 1),
        );
        Ok(())
    }

    /// High-throughput quantized `output[rows, columns] = input[rows, inner] @ weight[columns, inner]^T`.
    /// # Errors
    /// Rejects shape mismatches or undersized buffers.
    pub fn linear_quant(
        &mut self,
        input: &MetalDeviceBf16,
        weight: &MetalQuantWeight,
        output: &MetalDeviceBf16,
        rows: usize,
        columns: usize,
        inner: usize,
    ) -> Result<(), MetalBf16Error> {
        let in_req = rows
            .checked_mul(inner)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("linear_quant input overflow".into()))?;
        let out_req = rows
            .checked_mul(columns)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("linear_quant output overflow".into()))?;
        if weight.rows != columns || weight.cols != inner {
            return Err(MetalBf16Error::Shape(format!(
                "linear_quant weight [{}, {}] != expected [{columns}, {inner}]",
                weight.rows, weight.cols
            )));
        }
        if input.elements() < in_req || output.elements() < out_req {
            return Err(MetalBf16Error::Shape(
                "linear_quant input/output buffer capacity too small".into(),
            ));
        }

        let m_u32 = checked_u32(rows, "linear_quant M")?;
        let n_u32 = checked_u32(columns, "linear_quant N")?;
        let k_u32 = checked_u32(inner, "linear_quant K")?;
        let quant = &self.executor().quant;
        let encoder = self.encoder();

        if rows == 1 {
            let pipeline = match weight.kind {
                MetalQuantKind::Q4 => &quant.q4_gemv_m1,
                MetalQuantKind::Q8 => &quant.q8_gemv_m1,
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(input.raw_buffer()), 0);
            encoder.set_buffer(1, Some(&weight.packed), 0);
            encoder.set_buffer(2, Some(&weight.scales_biases), 0);
            encoder.set_buffer(3, Some(output.raw_buffer()), 0);
            encoder.set_bytes(4, 4, (&raw const n_u32).cast());
            encoder.set_bytes(5, 4, (&raw const k_u32).cast());
            let tgs = u64::from(n_u32).div_ceil(4);
            encoder.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(64, 1, 1));
        } else if rows <= 4 {
            let (pipeline, bm) = match (weight.kind, rows) {
                (MetalQuantKind::Q4, 2) => (&quant.q4_gemv_m2, 2_u64),
                (MetalQuantKind::Q4, _) => (&quant.q4_gemv_m4, 4_u64),
                (MetalQuantKind::Q8, 2) => (&quant.q8_gemv_m2, 2_u64),
                (MetalQuantKind::Q8, _) => (&quant.q8_gemv_m4, 4_u64),
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(input.raw_buffer()), 0);
            encoder.set_buffer(1, Some(&weight.packed), 0);
            encoder.set_buffer(2, Some(&weight.scales_biases), 0);
            encoder.set_buffer(3, Some(output.raw_buffer()), 0);
            encoder.set_bytes(4, 4, (&raw const m_u32).cast());
            encoder.set_bytes(5, 4, (&raw const n_u32).cast());
            encoder.set_bytes(6, 4, (&raw const k_u32).cast());
            let gx = u64::from(m_u32).div_ceil(bm);
            let gy = u64::from(n_u32).div_ceil(4);
            encoder.dispatch_thread_groups(MTLSize::new(gx, gy, 1), MTLSize::new(128, 1, 1));
        } else if rows <= 16 {
            let (pipeline, bm) = match (weight.kind, rows) {
                (MetalQuantKind::Q4, 5..=8) => (&quant.q4_gemm_m8, 8_u64),
                (MetalQuantKind::Q4, _) => (&quant.q4_gemm_m16, 16_u64),
                (MetalQuantKind::Q8, 5..=8) => (&quant.q8_gemm_m8, 8_u64),
                (MetalQuantKind::Q8, _) => (&quant.q8_gemm_m16, 16_u64),
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(input.raw_buffer()), 0);
            encoder.set_buffer(1, Some(&weight.packed), 0);
            encoder.set_buffer(2, Some(&weight.scales_biases), 0);
            encoder.set_buffer(3, Some(output.raw_buffer()), 0);
            encoder.set_bytes(4, 4, (&raw const m_u32).cast());
            encoder.set_bytes(5, 4, (&raw const n_u32).cast());
            encoder.set_bytes(6, 4, (&raw const k_u32).cast());
            let gx = u64::from(m_u32).div_ceil(bm);
            let gy = u64::from(n_u32).div_ceil(32);
            encoder.dispatch_thread_groups(MTLSize::new(gx, gy, 1), MTLSize::new(128, 1, 1));
        } else {
            let pipeline = match weight.kind {
                MetalQuantKind::Q4 => &quant.q4_gemm_m32,
                MetalQuantKind::Q8 => &quant.q8_gemm_m32,
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(input.raw_buffer()), 0);
            encoder.set_buffer(1, Some(&weight.packed), 0);
            encoder.set_buffer(2, Some(&weight.scales_biases), 0);
            encoder.set_buffer(3, Some(output.raw_buffer()), 0);
            encoder.set_bytes(4, 4, (&raw const m_u32).cast());
            encoder.set_bytes(5, 4, (&raw const n_u32).cast());
            encoder.set_bytes(6, 4, (&raw const k_u32).cast());
            let gx = u64::from(m_u32).div_ceil(32);
            let gy = u64::from(n_u32).div_ceil(32);
            encoder.dispatch_thread_groups(MTLSize::new(gx, gy, 1), MTLSize::new(128, 1, 1));
        }
        Ok(())
    }

    /// Fused `M = 1` dual quantized projection (`out_a = input @ weight_a^T`, `out_b = input @ weight_b^T`).
    /// # Errors
    /// Rejects shape/kind mismatches or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for fused M=1 dual quantized GEMV"
    )]
    pub fn dual_linear_quant_m1(
        &mut self,
        input: &MetalDeviceBf16,
        weight_a: &MetalQuantWeight,
        weight_b: &MetalQuantWeight,
        out_a: &MetalDeviceBf16,
        out_b: &MetalDeviceBf16,
        columns: usize,
        inner: usize,
    ) -> Result<(), MetalBf16Error> {
        if weight_a.kind != weight_b.kind
            || weight_a.rows != columns
            || weight_a.cols != inner
            || weight_b.rows != columns
            || weight_b.cols != inner
        {
            return Err(MetalBf16Error::Shape(
                "dual_linear_quant_m1 weight shape or kind mismatch".into(),
            ));
        }
        if input.elements() < inner || out_a.elements() < columns || out_b.elements() < columns {
            return Err(MetalBf16Error::Shape(
                "dual_linear_quant_m1 buffer capacity too small".into(),
            ));
        }
        let n_u32 = checked_u32(columns, "dual_linear_quant_m1 N")?;
        let k_u32 = checked_u32(inner, "dual_linear_quant_m1 K")?;
        let pipeline = match weight_a.kind {
            MetalQuantKind::Q4 => &self.executor().quant.q4_dual_gemv_m1,
            MetalQuantKind::Q8 => &self.executor().quant.q8_dual_gemv_m1,
        };
        let encoder = self.encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input.raw_buffer()), 0);
        encoder.set_buffer(1, Some(&weight_a.packed), 0);
        encoder.set_buffer(2, Some(&weight_a.scales_biases), 0);
        encoder.set_buffer(3, Some(&weight_b.packed), 0);
        encoder.set_buffer(4, Some(&weight_b.scales_biases), 0);
        encoder.set_buffer(5, Some(out_a.raw_buffer()), 0);
        encoder.set_buffer(6, Some(out_b.raw_buffer()), 0);
        encoder.set_bytes(7, 4, (&raw const n_u32).cast());
        encoder.set_bytes(8, 4, (&raw const k_u32).cast());
        let tgs = u64::from(n_u32).div_ceil(2);
        encoder.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(64, 1, 1));
        Ok(())
    }

    /// Fused `M = 1` quantized `Gate + Up + exact table GeGLU`.
    /// # Errors
    /// Rejects shape/kind mismatches or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for fused M=1 quantized Gate+Up+GeGLU"
    )]
    pub fn gate_up_geglu_quant_m1(
        &mut self,
        input: &MetalDeviceBf16,
        gate_w: &MetalQuantWeight,
        up_w: &MetalQuantWeight,
        gelu_table: &MetalBf16Buffer,
        output: &MetalDeviceBf16,
        error_flag: &MetalDeviceU32,
        columns: usize,
        inner: usize,
    ) -> Result<(), MetalBf16Error> {
        if gate_w.kind != up_w.kind
            || gate_w.rows != columns
            || gate_w.cols != inner
            || up_w.rows != columns
            || up_w.cols != inner
            || gelu_table.elements != 65_536
        {
            return Err(MetalBf16Error::Shape(
                "gate_up_geglu_quant_m1 shape, kind, or GELU table mismatch".into(),
            ));
        }
        if input.elements() < inner || output.elements() < columns || error_flag.elements() < 1 {
            return Err(MetalBf16Error::Shape(
                "gate_up_geglu_quant_m1 buffer capacity too small".into(),
            ));
        }
        let n_u32 = checked_u32(columns, "gate_up_geglu_quant_m1 N")?;
        let k_u32 = checked_u32(inner, "gate_up_geglu_quant_m1 K")?;
        let pipeline = match gate_w.kind {
            MetalQuantKind::Q4 => &self.executor().quant.q4_gate_up_geglu_m1,
            MetalQuantKind::Q8 => &self.executor().quant.q8_gate_up_geglu_m1,
        };
        let encoder = self.encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input.raw_buffer()), 0);
        encoder.set_buffer(1, Some(&gate_w.packed), 0);
        encoder.set_buffer(2, Some(&gate_w.scales_biases), 0);
        encoder.set_buffer(3, Some(&up_w.packed), 0);
        encoder.set_buffer(4, Some(&up_w.scales_biases), 0);
        encoder.set_buffer(5, Some(&gelu_table.buffer), 0);
        encoder.set_buffer(6, Some(output.raw_buffer()), 0);
        encoder.set_buffer(7, Some(error_flag.raw_buffer()), 0);
        encoder.set_bytes(8, 4, (&raw const n_u32).cast());
        encoder.set_bytes(9, 4, (&raw const k_u32).cast());
        let tgs = u64::from(n_u32).div_ceil(2);
        encoder.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(64, 1, 1));
        Ok(())
    }

    /// Fused `M = 1` quantized linear projection + exact table `GeGLU` with `up_in[up_offset..]`.
    /// # Errors
    /// Rejects shape mismatches or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for fused M=1 quantized linear+GeGLU"
    )]
    pub fn linear_geglu_quant_m1(
        &mut self,
        input: &MetalDeviceBf16,
        weight: &MetalQuantWeight,
        up_in: &MetalDeviceBf16,
        up_offset_elems: usize,
        gelu_table: &MetalBf16Buffer,
        output: &MetalDeviceBf16,
        error_flag: &MetalDeviceU32,
        columns: usize,
        inner: usize,
    ) -> Result<(), MetalBf16Error> {
        if weight.rows != columns || weight.cols != inner || gelu_table.elements != 65_536 {
            return Err(MetalBf16Error::Shape(
                "linear_geglu_quant_m1 shape or GELU table mismatch".into(),
            ));
        }
        let up_end = up_offset_elems.checked_add(columns).ok_or_else(|| {
            MetalBf16Error::Shape("linear_geglu_quant_m1 up offset overflow".into())
        })?;
        if input.elements() < inner
            || up_in.elements() < up_end
            || output.elements() < columns
            || error_flag.elements() < 1
        {
            return Err(MetalBf16Error::Shape(
                "linear_geglu_quant_m1 buffer capacity too small".into(),
            ));
        }
        let up_byte_offset = u64::try_from(
            up_offset_elems
                .checked_mul(2)
                .ok_or_else(|| MetalBf16Error::Shape("up byte offset overflow".into()))?,
        )
        .map_err(|_| MetalBf16Error::Shape("up byte offset exceeds u64".into()))?;
        let n_u32 = checked_u32(columns, "linear_geglu_quant_m1 N")?;
        let k_u32 = checked_u32(inner, "linear_geglu_quant_m1 K")?;
        let pipeline = match weight.kind {
            MetalQuantKind::Q4 => &self.executor().quant.q4_linear_geglu_m1,
            MetalQuantKind::Q8 => &self.executor().quant.q8_linear_geglu_m1,
        };
        let encoder = self.encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input.raw_buffer()), 0);
        encoder.set_buffer(1, Some(&weight.packed), 0);
        encoder.set_buffer(2, Some(&weight.scales_biases), 0);
        encoder.set_buffer(3, Some(up_in.raw_buffer()), up_byte_offset);
        encoder.set_buffer(4, Some(&gelu_table.buffer), 0);
        encoder.set_buffer(5, Some(output.raw_buffer()), 0);
        encoder.set_buffer(6, Some(error_flag.raw_buffer()), 0);
        encoder.set_bytes(7, 4, (&raw const n_u32).cast());
        encoder.set_bytes(8, 4, (&raw const k_u32).cast());
        let tgs = u64::from(n_u32).div_ceil(4);
        encoder.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(64, 1, 1));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dequant_cpu(rows: usize, cols: usize, kind: MetalQuantKind, bf16: &[u16]) -> Vec<f32> {
        let groups = cols / METAL_QUANT_GROUP_SIZE;
        let q_max = match kind {
            MetalQuantKind::Q4 => 15.0_f32,
            MetalQuantKind::Q8 => 255.0_f32,
        };
        let mut out = vec![0.0_f32; rows * cols];
        for r in 0..rows {
            for g in 0..groups {
                let base = r * cols + g * METAL_QUANT_GROUP_SIZE;
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                for i in 0..METAL_QUANT_GROUP_SIZE {
                    let v = bf16_to_f32(bf16[base + i]);
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                let span = hi - lo;
                let s = bf16_to_f32(f32_to_bf16_rne(if span == 0.0 {
                    0.0
                } else {
                    span / q_max
                }));
                let b = bf16_to_f32(f32_to_bf16_rne(lo));
                let inv_s = if s == 0.0 { 0.0 } else { 1.0 / s };
                for i in 0..METAL_QUANT_GROUP_SIZE {
                    let v = bf16_to_f32(bf16[base + i]);
                    let q = ((v - b) * inv_s).round().clamp(0.0, q_max);
                    out[base + i] = q * s + b;
                }
            }
        }
        out
    }

    #[test]
    fn quantized_linear_and_embed_match_dequantized_cpu_reference_across_shapes() {
        let exec = MetalBf16Executor::new().expect("Metal device");
        for kind in [MetalQuantKind::Q4, MetalQuantKind::Q8] {
            for &(m, n, k) in &[
                (1_usize, 13_usize, 64_usize),
                (2, 31, 128),
                (4, 35, 192),
                (8, 65, 256),
                (16, 48, 256),
                (18, 67, 256),
                (37, 99, 320),
            ] {
                let mut x_bf16 = Vec::with_capacity(m * k);
                for i in 0..(m * k) {
                    #[expect(clippy::cast_precision_loss, reason = "small synthetic test index")]
                    let v = ((i * 17) % 97) as f32 * 0.02 - 0.96;
                    x_bf16.push(f32_to_bf16_rne(v));
                }
                let mut w_bf16 = Vec::with_capacity(n * k);
                for i in 0..(n * k) {
                    #[expect(clippy::cast_precision_loss, reason = "small synthetic test index")]
                    let v = ((i * 31 + 7) % 89) as f32 * 0.03 - 1.32;
                    w_bf16.push(f32_to_bf16_rne(v));
                }

                let qw = exec.upload_quantized_bf16(n, k, kind, &w_bf16).unwrap();
                let mut x_dev = exec.alloc_bf16(m * k).unwrap();
                x_dev.write(&x_bf16).unwrap();
                let out_dev = exec.alloc_bf16(m * n).unwrap();

                let mut session = exec.begin_fast_session().unwrap();
                session
                    .linear_quant(&x_dev, &qw, &out_dev, m, n, k)
                    .unwrap();
                session.submit("test linear_quant").unwrap();

                let deq = dequant_cpu(n, k, kind, &w_bf16);
                let got = out_dev.as_slice(m * n);
                for r in 0..m {
                    for c in 0..n {
                        let mut acc = 0.0_f32;
                        for j in 0..k {
                            let wv = if m > 4 {
                                bf16_to_f32(f32_to_bf16_rne(deq[c * k + j]))
                            } else {
                                deq[c * k + j]
                            };
                            acc += bf16_to_f32(x_bf16[r * k + j]) * wv;
                        }
                        let actual = bf16_to_f32(got[r * n + c]);
                        let expected = bf16_to_f32(f32_to_bf16_rne(acc));
                        let tol = 0.01_f32.max(expected.abs() * 0.01);
                        assert!(
                            (actual - expected).abs() <= tol,
                            "{kind:?} ({m}x{n}x{k}) [{r},{c}]: actual={actual} expected={expected}"
                        );
                    }
                }
            }

            // Test quantized embedding gather + scale
            let vocab = 12_usize;
            let width = 128_usize;
            let mut tbl_bf16 = Vec::with_capacity(vocab * width);
            for i in 0..(vocab * width) {
                #[expect(clippy::cast_precision_loss, reason = "small synthetic test index")]
                let v = ((i * 23 + 3) % 73) as f32 * 0.025 - 0.9;
                tbl_bf16.push(f32_to_bf16_rne(v));
            }
            let qtbl = exec
                .upload_quantized_bf16(vocab, width, kind, &tbl_bf16)
                .unwrap();
            let deq = dequant_cpu(vocab, width, kind, &tbl_bf16);
            let toks = [0_u32, 5, 11, 2];
            let mut tok_dev = exec.alloc_u32(toks.len()).unwrap();
            tok_dev.write(&toks).unwrap();
            let out_dev = exec.alloc_bf16(toks.len() * width).unwrap();
            let scale = 1.5_f32;

            let mut session = exec.begin_fast_session().unwrap();
            session
                .embed_gather_quant(&tok_dev, &qtbl, &out_dev, toks.len(), width, scale)
                .unwrap();
            session.submit("test embed_gather_quant").unwrap();

            let got = out_dev.as_slice(toks.len() * width);
            for (r, &t) in toks.iter().enumerate() {
                for c in 0..width {
                    let expected =
                        bf16_to_f32(f32_to_bf16_rne(deq[t as usize * width + c] * scale));
                    let actual = bf16_to_f32(got[r * width + c]);
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{kind:?} embed_gather [{r},{c}]: actual={actual} expected={expected}"
                    );
                }
            }
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "verifies dual_linear, gate_up_geglu, and linear_geglu bit-exactness across Q4 and Q8"
    )]
    fn fused_m1_quant_ops_match_unfused_bit_for_bit() {
        let exec = MetalBf16Executor::new().expect("Metal device");
        let gelu_vals: Vec<u16> = (0..65_536_usize)
            .map(|bits| {
                let val = bf16_to_f32(u16::try_from(bits).expect("u16 range"));
                if val.is_finite() {
                    let coeff = (2.0_f32 / std::f32::consts::PI).sqrt();
                    let gelu =
                        0.5 * val * (1.0 + (coeff * (val + 0.044_715 * val * val * val)).tanh());
                    f32_to_bf16_rne(gelu)
                } else {
                    0
                }
            })
            .collect();
        let gelu_tbl = exec.upload(&gelu_vals).unwrap();
        let cols = 67_usize;
        let inner = 256_usize;
        let x_vals: Vec<u16> = (0..inner)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test index")]
                let v = ((i * 13 + 5) % 83) as f32 * 0.02 - 0.82;
                f32_to_bf16_rne(v)
            })
            .collect();
        let gate_vals: Vec<u16> = (0..cols * inner)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test index")]
                let v = ((i * 29 + 11) % 79) as f32 * 0.025 - 0.95;
                f32_to_bf16_rne(v)
            })
            .collect();
        let up_vals: Vec<u16> = (0..cols * inner)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test index")]
                let v = ((i * 37 + 3) % 71) as f32 * 0.025 - 0.88;
                f32_to_bf16_rne(v)
            })
            .collect();
        let mut x_dev = exec.alloc_bf16(inner).unwrap();
        x_dev.write(&x_vals).unwrap();

        for kind in [MetalQuantKind::Q4, MetalQuantKind::Q8] {
            let gate_w = exec
                .upload_quantized_bf16(cols, inner, kind, &gate_vals)
                .unwrap();
            let up_w = exec
                .upload_quantized_bf16(cols, inner, kind, &up_vals)
                .unwrap();
            let gate_ref = exec.alloc_bf16(cols).unwrap();
            let up_ref = exec.alloc_bf16(cols).unwrap();
            let geglu_ref = exec.alloc_bf16(cols).unwrap();
            let dual_gate = exec.alloc_bf16(cols).unwrap();
            let dual_up = exec.alloc_bf16(cols).unwrap();
            let geglu_fused = exec.alloc_bf16(cols).unwrap();
            let lin_geglu_fused = exec.alloc_bf16(cols).unwrap();
            let err_flag = exec.alloc_u32(1).unwrap();

            let mut session = exec.begin_fast_session().unwrap();
            session
                .linear_quant(&x_dev, &gate_w, &gate_ref, 1, cols, inner)
                .unwrap();
            session
                .linear_quant(&x_dev, &up_w, &up_ref, 1, cols, inner)
                .unwrap();
            session
                .geglu_table(
                    &gate_ref, &up_ref, 0, &gelu_tbl, &geglu_ref, &err_flag, cols,
                )
                .unwrap();
            session
                .dual_linear_quant_m1(&x_dev, &gate_w, &up_w, &dual_gate, &dual_up, cols, inner)
                .unwrap();
            session
                .gate_up_geglu_quant_m1(
                    &x_dev,
                    &gate_w,
                    &up_w,
                    &gelu_tbl,
                    &geglu_fused,
                    &err_flag,
                    cols,
                    inner,
                )
                .unwrap();
            session
                .linear_geglu_quant_m1(
                    &x_dev,
                    &gate_w,
                    &up_ref,
                    0,
                    &gelu_tbl,
                    &lin_geglu_fused,
                    &err_flag,
                    cols,
                    inner,
                )
                .unwrap();
            session.submit("test fused M=1 quant ops").unwrap();

            assert_eq!(err_flag.as_slice(1)[0], 0);
            assert_eq!(dual_gate.as_slice(cols), gate_ref.as_slice(cols));
            assert_eq!(dual_up.as_slice(cols), up_ref.as_slice(cols));
            assert_eq!(geglu_fused.as_slice(cols), geglu_ref.as_slice(cols));
            assert_eq!(lin_geglu_fused.as_slice(cols), geglu_ref.as_slice(cols));
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "verifies Q4 and Q8 pre-quantized parity plus truncation, trailing, and non-finite rejections"
    )]
    fn read_prequantized_matches_online_quantized_bit_for_bit_and_rejects_bad_inputs() {
        let exec = MetalBf16Executor::new().expect("Metal device");
        let rows = 19_usize;
        let cols = 128_usize;
        let m = 4_usize;
        let w_bf16: Vec<u16> = (0..rows * cols)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test index")]
                let v = ((i * 29 + 5) % 91) as f32 * 0.025 - 1.1;
                f32_to_bf16_rne(v)
            })
            .collect();
        let x_bf16: Vec<u16> = (0..m * cols)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test index")]
                let v = ((i * 17 + 3) % 79) as f32 * 0.02 - 0.78;
                f32_to_bf16_rne(v)
            })
            .collect();
        let mut x_dev = exec.alloc_bf16(m * cols).unwrap();
        x_dev.write(&x_bf16).unwrap();

        let src_bytes: Vec<u8> = w_bf16.iter().flat_map(|b| b.to_le_bytes()).collect();
        let total_groups = rows * (cols / METAL_QUANT_GROUP_SIZE);

        for kind in [MetalQuantKind::Q4, MetalQuantKind::Q8] {
            let online = exec
                .upload_quantized_bf16(rows, cols, kind, &w_bf16)
                .unwrap();
            let mut sb = vec![[0_u16; 2]; total_groups];
            let packed_bytes: Vec<u8> = match kind {
                MetalQuantKind::Q4 => {
                    let mut words = vec![0_u32; rows * (cols / 8)];
                    quantize_rows_q4(&src_bytes, cols, &mut words, &mut sb).unwrap();
                    words.iter().flat_map(|w| w.to_le_bytes()).collect()
                }
                MetalQuantKind::Q8 => {
                    let mut bytes = vec![0_u8; rows * cols];
                    quantize_rows_q8(&src_bytes, cols, &mut bytes, &mut sb).unwrap();
                    bytes
                }
            };
            let scale_bytes: Vec<u8> = sb.iter().flat_map(|pair| pair[0].to_le_bytes()).collect();
            let bias_bytes: Vec<u8> = sb.iter().flat_map(|pair| pair[1].to_le_bytes()).collect();

            let preq = exec
                .read_prequantized(
                    rows,
                    cols,
                    kind,
                    &mut packed_bytes.as_slice(),
                    &mut scale_bytes.as_slice(),
                    &mut bias_bytes.as_slice(),
                )
                .unwrap();
            assert_eq!(preq.bytes(), online.bytes());

            let out_online = exec.alloc_bf16(m * rows).unwrap();
            let out_preq = exec.alloc_bf16(m * rows).unwrap();
            let mut session = exec.begin_fast_session().unwrap();
            session
                .linear_quant(&x_dev, &online, &out_online, m, rows, cols)
                .unwrap();
            session
                .linear_quant(&x_dev, &preq, &out_preq, m, rows, cols)
                .unwrap();
            session.submit("test prequantized parity").unwrap();
            assert_eq!(out_preq.as_slice(m * rows), out_online.as_slice(m * rows));

            // Reject truncated packed/scales/biases
            assert!(
                exec.read_prequantized(
                    rows,
                    cols,
                    kind,
                    &mut &packed_bytes[..packed_bytes.len() - 1],
                    &mut scale_bytes.as_slice(),
                    &mut bias_bytes.as_slice(),
                )
                .is_err()
            );
            assert!(
                exec.read_prequantized(
                    rows,
                    cols,
                    kind,
                    &mut packed_bytes.as_slice(),
                    &mut &scale_bytes[..scale_bytes.len() - 2],
                    &mut bias_bytes.as_slice(),
                )
                .is_err()
            );
            // Reject trailing bytes
            let mut extra = packed_bytes.clone();
            extra.push(0);
            assert!(
                exec.read_prequantized(
                    rows,
                    cols,
                    kind,
                    &mut extra.as_slice(),
                    &mut scale_bytes.as_slice(),
                    &mut bias_bytes.as_slice(),
                )
                .is_err()
            );
            // Reject non-finite scale and non-finite bias
            let mut bad_scales = scale_bytes.clone();
            bad_scales[0..2].copy_from_slice(&0x7f80_u16.to_le_bytes());
            assert!(
                exec.read_prequantized(
                    rows,
                    cols,
                    kind,
                    &mut packed_bytes.as_slice(),
                    &mut bad_scales.as_slice(),
                    &mut bias_bytes.as_slice(),
                )
                .is_err()
            );
            let mut bad_biases = bias_bytes.clone();
            bad_biases[0..2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
            assert!(
                exec.read_prequantized(
                    rows,
                    cols,
                    kind,
                    &mut packed_bytes.as_slice(),
                    &mut scale_bytes.as_slice(),
                    &mut bad_biases.as_slice(),
                )
                .is_err()
            );
        }
    }
}
