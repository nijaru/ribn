//! Single-command-buffer fast Metal kernels and persistent device buffers.
//!
//! Designed for model-wide execution: weights remain immutable `MetalBf16Buffer`s,
//! while KV caches and batch scratch live in pre-allocated `MetalDevice*` buffers
//! reused across steps with zero per-step allocations or intermediate CPU syncs.

use super::submission::NativeCommand;
use super::{MetalBf16Buffer, MetalBf16Error, MetalBf16Executor};
use metal::{
    Buffer, CompileOptions, ComputeCommandEncoder, ComputePipelineState, Device,
    MTLLanguageVersion, MTLSize,
};

const FAST_SOURCE: &str = include_str!("../bf16_fast.metal");

const SOFTCAP_MAX_PARALLEL_ROWS: usize = 64;
const SOFTCAP_BLOCKS_PER_ROW: u32 = 64;

pub(super) struct FastPipelines {
    gemv_m1: ComputePipelineState,
    gemv_m2: ComputePipelineState,
    gemv_m4: ComputePipelineState,
    gemv_m8: ComputePipelineState,
    gemm_m32: ComputePipelineState,
    dual_gemv_m1: ComputePipelineState,
    gate_up_geglu_m1: ComputePipelineState,
    linear_geglu_m1: ComputePipelineState,
    embed_gather: ComputePipelineState,
    ple_combine_transpose: ComputePipelineState,
    rms: ComputePipelineState,
    add_rms: ComputePipelineState,
    residual_add_scale: ComputePipelineState,
    post_norm_add_pre_norm: ComputePipelineState,
    rms_norm_residual_add_scale: ComputePipelineState,
    q_norm_rope: ComputePipelineState,
    kv_norm_rope_write: ComputePipelineState,
    flash_attn: ComputePipelineState,
    geglu_table: ComputePipelineState,
    gather_rows: ComputePipelineState,
    gather_rows_rms: ComputePipelineState,
    softcap_argmax: ComputePipelineState,
    softcap_argmax_stage1: ComputePipelineState,
    softcap_argmax_stage2: ComputePipelineState,
    softcap_partial_vals: Buffer,
    softcap_partial_idxs: Buffer,
}

impl FastPipelines {
    pub(super) fn compile(device: &Device) -> Result<Self, MetalBf16Error> {
        let options = CompileOptions::new();
        options.set_language_version(MTLLanguageVersion::V3_1);
        let library = device
            .new_library_with_source(FAST_SOURCE, &options)
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
            gemv_m1: load("fast_bf16_gemv_m1")?,
            gemv_m2: load("fast_bf16_gemv_m2")?,
            gemv_m4: load("fast_bf16_gemv_m4")?,
            gemv_m8: load("fast_bf16_gemv_m8")?,
            gemm_m32: load("fast_bf16_gemm_m32")?,
            dual_gemv_m1: load("fast_bf16_dual_gemv_m1")?,
            gate_up_geglu_m1: load("fast_bf16_gate_up_geglu_m1")?,
            linear_geglu_m1: load("fast_bf16_linear_geglu_m1")?,
            embed_gather: load("fast_bf16_embed_gather")?,
            ple_combine_transpose: load("fast_bf16_ple_combine_transpose")?,
            rms: load("fast_bf16_rms")?,
            add_rms: load("fast_bf16_add_rms")?,
            residual_add_scale: load("fast_bf16_residual_add_scale")?,
            post_norm_add_pre_norm: load("fast_bf16_post_norm_add_pre_norm")?,
            rms_norm_residual_add_scale: load("fast_bf16_rms_norm_residual_add_scale")?,
            q_norm_rope: load("fast_bf16_q_norm_rope")?,
            kv_norm_rope_write: load("fast_bf16_kv_norm_rope_write")?,
            flash_attn: load("fast_bf16_flash_attn")?,
            geglu_table: load("fast_bf16_geglu_table")?,
            gather_rows: load("fast_bf16_gather_rows")?,
            gather_rows_rms: load("fast_bf16_gather_rows_rms")?,
            softcap_argmax: load("fast_bf16_softcap_argmax")?,
            softcap_argmax_stage1: load("fast_bf16_softcap_argmax_stage1")?,
            softcap_argmax_stage2: load("fast_bf16_softcap_argmax_stage2")?,
            softcap_partial_vals: device.new_buffer(
                (SOFTCAP_MAX_PARALLEL_ROWS * SOFTCAP_BLOCKS_PER_ROW as usize * 4) as u64,
                metal::MTLResourceOptions::StorageModePrivate,
            ),
            softcap_partial_idxs: device.new_buffer(
                (SOFTCAP_MAX_PARALLEL_ROWS * SOFTCAP_BLOCKS_PER_ROW as usize * 4) as u64,
                metal::MTLResourceOptions::StorageModePrivate,
            ),
        })
    }
}

/// Reusable shared-storage BF16 device buffer for KV caches and batch scratch.
pub struct MetalDeviceBf16 {
    buffer: Buffer,
    elements: usize,
}

impl MetalDeviceBf16 {
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    pub(super) const fn raw_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Borrow the first `count` BF16 elements after command completion.
    /// # Panics
    /// Panics if `count > self.elements()`.
    #[must_use]
    pub fn as_slice(&self, count: usize) -> &[u16] {
        assert!(count <= self.elements, "MetalDeviceBf16 read out of bounds");
        // SAFETY: shared Metal storage is page-aligned and allocated for `self.elements` u16s.
        unsafe { std::slice::from_raw_parts(self.buffer.contents().cast::<u16>(), count) }
    }

    /// Overwrite the prefix with host BF16 values while idle.
    /// # Errors
    /// Rejects slices longer than the buffer capacity.
    pub fn write(&mut self, values: &[u16]) -> Result<(), MetalBf16Error> {
        if values.len() > self.elements {
            return Err(MetalBf16Error::Shape(
                "MetalDeviceBf16 write exceeds capacity".into(),
            ));
        }
        if !values.is_empty() {
            // SAFETY: destination is exclusively owned (`&mut self`) and fits `values`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    self.buffer.contents().cast::<u16>(),
                    values.len(),
                );
            }
        }
        Ok(())
    }
}

/// Reusable shared-storage `u32` device buffer for token IDs, slots, positions, and argmax.
pub struct MetalDeviceU32 {
    buffer: Buffer,
    elements: usize,
}

impl MetalDeviceU32 {
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    pub(super) const fn raw_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Borrow the first `count` u32 elements after command completion.
    /// # Panics
    /// Panics if `count > self.elements()`.
    #[must_use]
    pub fn as_slice(&self, count: usize) -> &[u32] {
        assert!(count <= self.elements, "MetalDeviceU32 read out of bounds");
        // SAFETY: shared Metal storage is page-aligned and allocated for `self.elements` u32s.
        unsafe { std::slice::from_raw_parts(self.buffer.contents().cast::<u32>(), count) }
    }

    /// Overwrite `values` starting at element `offset` while idle.
    /// # Errors
    /// Rejects writes exceeding capacity.
    pub fn write_at(&mut self, offset: usize, values: &[u32]) -> Result<(), MetalBf16Error> {
        let end = offset
            .checked_add(values.len())
            .ok_or_else(|| MetalBf16Error::Shape("MetalDeviceU32 offset overflow".into()))?;
        if end > self.elements {
            return Err(MetalBf16Error::Shape(
                "MetalDeviceU32 write_at exceeds capacity".into(),
            ));
        }
        if !values.is_empty() {
            // SAFETY: destination is exclusively owned (`&mut self`) and `[offset..end]` fits.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    self.buffer.contents().cast::<u32>().add(offset),
                    values.len(),
                );
            }
        }
        Ok(())
    }

    /// Overwrite the prefix with host `u32` values while idle.
    /// # Errors
    /// Rejects slices longer than the buffer capacity.
    pub fn write(&mut self, values: &[u32]) -> Result<(), MetalBf16Error> {
        if values.len() > self.elements {
            return Err(MetalBf16Error::Shape(
                "MetalDeviceU32 write exceeds capacity".into(),
            ));
        }
        if !values.is_empty() {
            // SAFETY: destination is exclusively owned (`&mut self`) and fits `values`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    self.buffer.contents().cast::<u32>(),
                    values.len(),
                );
            }
        }
        Ok(())
    }
}

/// Reusable shared-storage `f32` device buffer (e.g. per-layer `RoPE` inverse frequencies).
pub struct MetalDeviceF32 {
    buffer: Buffer,
    elements: usize,
}

impl MetalDeviceF32 {
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    /// Overwrite the prefix with host `f32` values while idle.
    /// # Errors
    /// Rejects slices longer than the buffer capacity.
    pub fn write(&mut self, values: &[f32]) -> Result<(), MetalBf16Error> {
        if values.len() > self.elements {
            return Err(MetalBf16Error::Shape(
                "MetalDeviceF32 write exceeds capacity".into(),
            ));
        }
        if !values.is_empty() {
            // SAFETY: destination is exclusively owned (`&mut self`) and fits `values`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    self.buffer.contents().cast::<f32>(),
                    values.len(),
                );
            }
        }
        Ok(())
    }
}

/// One serial compute command buffer encoding an entire forward pass.
pub struct MetalFastSession<'a> {
    executor: &'a MetalBf16Executor,
    command: NativeCommand<'a>,
    encoder: ComputeCommandEncoder,
    ended: bool,
}

impl<'a> MetalFastSession<'a> {
    pub(super) const fn executor(&self) -> &'a MetalBf16Executor {
        self.executor
    }

    pub(super) const fn encoder(&self) -> &ComputeCommandEncoder {
        &self.encoder
    }
}

impl Drop for MetalFastSession<'_> {
    fn drop(&mut self) {
        if !self.ended {
            self.encoder.end_encoding();
            self.ended = true;
        }
    }
}

fn checked_u32(value: usize, label: &str) -> Result<u32, MetalBf16Error> {
    u32::try_from(value).map_err(|_| MetalBf16Error::Shape(format!("{label} exceeds u32")))
}

fn check_len(actual: usize, required: usize, label: &str) -> Result<(), MetalBf16Error> {
    if actual < required {
        Err(MetalBf16Error::Shape(format!(
            "{label}: capacity {actual} < required {required}"
        )))
    } else {
        Ok(())
    }
}

impl MetalBf16Executor {
    /// Allocate a zero-initialized shared BF16 buffer for persistent KV cache or scratch.
    /// # Errors
    /// Rejects zero/overflowing sizes or device OOM.
    pub fn alloc_bf16(&self, elements: usize) -> Result<MetalDeviceBf16, MetalBf16Error> {
        let bytes = elements
            .checked_mul(2)
            .filter(|&b| b > 0)
            .ok_or(MetalBf16Error::Invalid("invalid BF16 buffer size"))?;
        let buffer = self.allocate_shared(bytes)?;
        // SAFETY: newly allocated exclusive buffer of `bytes` writable bytes.
        unsafe {
            std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bytes);
        }
        Ok(MetalDeviceBf16 { buffer, elements })
    }

    /// Allocate a zero-initialized shared `u32` buffer.
    /// # Errors
    /// Rejects zero/overflowing sizes or device OOM.
    pub fn alloc_u32(&self, elements: usize) -> Result<MetalDeviceU32, MetalBf16Error> {
        let bytes = elements
            .checked_mul(4)
            .filter(|&b| b > 0)
            .ok_or(MetalBf16Error::Invalid("invalid u32 buffer size"))?;
        let buffer = self.allocate_shared(bytes)?;
        // SAFETY: newly allocated exclusive buffer of `bytes` writable bytes.
        unsafe {
            std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bytes);
        }
        Ok(MetalDeviceU32 { buffer, elements })
    }

    /// Allocate and initialize a shared `f32` buffer.
    /// # Errors
    /// Rejects empty/overflowing sizes or device OOM.
    pub fn upload_f32(&self, values: &[f32]) -> Result<MetalDeviceF32, MetalBf16Error> {
        let bytes = values
            .len()
            .checked_mul(4)
            .filter(|&b| b > 0)
            .ok_or(MetalBf16Error::Invalid("invalid f32 buffer size"))?;
        let buffer = self.allocate_shared(bytes)?;
        // SAFETY: newly allocated exclusive buffer of `values.len()` f32s.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                buffer.contents().cast::<f32>(),
                values.len(),
            );
        }
        Ok(MetalDeviceF32 {
            buffer,
            elements: values.len(),
        })
    }

    /// Begin a single-command-buffer compute session on this Metal queue.
    /// # Errors
    /// Rejects a faulted execution owner or command/encoder allocation failure.
    pub fn begin_fast_session(&self) -> Result<MetalFastSession<'_>, MetalBf16Error> {
        let command = self.command()?;
        let encoder = Self::compute_encoder(&command)?;
        Ok(MetalFastSession {
            executor: self,
            command,
            encoder,
            ended: false,
        })
    }
}

impl MetalFastSession<'_> {
    /// Gather `rows` rows of length `width` from `table` using `token_ids` and multiply by `scale`.
    /// # Errors
    /// Rejects invalid shapes or undersized buffers.
    pub fn embed_gather(
        &mut self,
        token_ids: &MetalDeviceU32,
        table: &MetalBf16Buffer,
        output: &MetalDeviceBf16,
        rows: usize,
        width: usize,
        scale: f32,
    ) -> Result<(), MetalBf16Error> {
        let total = rows
            .checked_mul(width)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("embed_gather size overflow".into()))?;
        check_len(token_ids.elements, rows, "embed_gather token_ids")?;
        check_len(output.elements, total, "embed_gather output")?;
        let width_u32 = checked_u32(width, "embed_gather width")?;
        let total_u32 = checked_u32(total, "embed_gather total")?;
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.embed_gather);
        self.encoder.set_buffer(0, Some(&token_ids.buffer), 0);
        self.encoder.set_buffer(1, Some(&table.buffer), 0);
        self.encoder.set_buffer(2, Some(&output.buffer), 0);
        self.encoder.set_bytes(3, 4, (&raw const width_u32).cast());
        self.encoder.set_bytes(4, 4, (&raw const scale).cast());
        self.encoder.set_bytes(5, 4, (&raw const total_u32).cast());
        let total_u64 = u64::from(total_u32);
        self.encoder.dispatch_threads(
            MTLSize::new(total_u64, 1, 1),
            MTLSize::new(total_u64.min(256), 1, 1),
        );
        Ok(())
    }

    /// High-throughput `output[rows, columns] = input[rows, inner] @ weight[columns, inner]^T`.
    ///
    /// Automatically dispatches coalesced SIMD-group GEMV (`rows <= 8`) or tiled
    /// `simdgroup_matrix` GEMM (`rows > 8`).
    /// # Errors
    /// Rejects invalid dimensions or undersized buffers.
    pub fn linear(
        &mut self,
        input: &MetalDeviceBf16,
        weight: &MetalBf16Buffer,
        output: &MetalDeviceBf16,
        rows: usize,
        columns: usize,
        inner: usize,
    ) -> Result<(), MetalBf16Error> {
        let in_req = rows
            .checked_mul(inner)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("linear input overflow".into()))?;
        let w_req = columns
            .checked_mul(inner)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("linear weight overflow".into()))?;
        let out_req = rows
            .checked_mul(columns)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("linear output overflow".into()))?;
        check_len(input.elements, in_req, "linear input")?;
        if weight.elements != w_req {
            return Err(MetalBf16Error::Shape(format!(
                "linear weight elements {} != expected {w_req}",
                weight.elements
            )));
        }
        check_len(output.elements, out_req, "linear output")?;

        let m_u32 = checked_u32(rows, "linear M")?;
        let n_u32 = checked_u32(columns, "linear N")?;
        let k_u32 = checked_u32(inner, "linear K")?;

        if rows == 1 {
            self.encoder
                .set_compute_pipeline_state(&self.executor.fast.gemv_m1);
            self.encoder.set_buffer(0, Some(&input.buffer), 0);
            self.encoder.set_buffer(1, Some(&weight.buffer), 0);
            self.encoder.set_buffer(2, Some(&output.buffer), 0);
            self.encoder.set_bytes(3, 4, (&raw const n_u32).cast());
            self.encoder.set_bytes(4, 4, (&raw const k_u32).cast());
            let tgs = u64::from(n_u32).div_ceil(8);
            self.encoder
                .dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(128, 1, 1));
        } else if rows <= 16 {
            let (pipeline, bm) = if rows == 2 {
                (&self.executor.fast.gemv_m2, 2_u64)
            } else if rows <= 4 {
                (&self.executor.fast.gemv_m4, 4_u64)
            } else {
                (&self.executor.fast.gemv_m8, 8_u64)
            };
            self.encoder.set_compute_pipeline_state(pipeline);
            self.encoder.set_buffer(0, Some(&input.buffer), 0);
            self.encoder.set_buffer(1, Some(&weight.buffer), 0);
            self.encoder.set_buffer(2, Some(&output.buffer), 0);
            self.encoder.set_bytes(3, 4, (&raw const m_u32).cast());
            self.encoder.set_bytes(4, 4, (&raw const n_u32).cast());
            self.encoder.set_bytes(5, 4, (&raw const k_u32).cast());
            let gx = u64::from(m_u32).div_ceil(bm);
            let gy = u64::from(n_u32).div_ceil(4);
            self.encoder
                .dispatch_thread_groups(MTLSize::new(gx, gy, 1), MTLSize::new(64, 1, 1));
        } else {
            self.encoder
                .set_compute_pipeline_state(&self.executor.fast.gemm_m32);
            self.encoder.set_buffer(0, Some(&input.buffer), 0);
            self.encoder.set_buffer(1, Some(&weight.buffer), 0);
            self.encoder.set_buffer(2, Some(&output.buffer), 0);
            self.encoder.set_bytes(3, 4, (&raw const m_u32).cast());
            self.encoder.set_bytes(4, 4, (&raw const n_u32).cast());
            self.encoder.set_bytes(5, 4, (&raw const k_u32).cast());
            let gx = u64::from(m_u32).div_ceil(32);
            let gy = u64::from(n_u32).div_ceil(32);
            self.encoder
                .dispatch_thread_groups(MTLSize::new(gx, gy, 1), MTLSize::new(128, 1, 1));
        }
        Ok(())
    }

    /// Fused PLE post-projection: scale by `inv_sqrt_hidden`, per-`(m, layer)` `RMSNorm`
    /// with `ple_norm`, combine with `ple_id` (`* 1/sqrt(2)`), and transpose into
    /// `[layer, rows, ple_dim]`.
    /// # Errors
    /// Rejects invalid dimensions or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for fused PLE post-projection"
    )]
    pub fn ple_combine_transpose(
        &mut self,
        projection: &MetalDeviceBf16,
        ple_id: &MetalDeviceBf16,
        ple_norm: &MetalBf16Buffer,
        per_layer_out: &MetalDeviceBf16,
        rows: usize,
        num_layers: usize,
        ple_dim: usize,
        inv_sqrt_hidden: f32,
        epsilon: f32,
    ) -> Result<(), MetalBf16Error> {
        let total = rows
            .checked_mul(num_layers)
            .and_then(|n| n.checked_mul(ple_dim))
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("ple_combine_transpose overflow".into()))?;
        check_len(projection.elements, total, "ple projection")?;
        check_len(ple_id.elements, total, "ple id")?;
        check_len(per_layer_out.elements, total, "ple per_layer_out")?;
        if ple_norm.elements != ple_dim {
            return Err(MetalBf16Error::Shape("ple_norm width mismatch".into()));
        }
        let m_u32 = checked_u32(rows, "ple M")?;
        let layers_u32 = checked_u32(num_layers, "ple num_layers")?;
        let dim_u32 = checked_u32(ple_dim, "ple dim")?;
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.ple_combine_transpose);
        self.encoder.set_buffer(0, Some(&projection.buffer), 0);
        self.encoder.set_buffer(1, Some(&ple_id.buffer), 0);
        self.encoder.set_buffer(2, Some(&ple_norm.buffer), 0);
        self.encoder.set_buffer(3, Some(&per_layer_out.buffer), 0);
        self.encoder.set_bytes(4, 4, (&raw const m_u32).cast());
        self.encoder.set_bytes(5, 4, (&raw const layers_u32).cast());
        self.encoder.set_bytes(6, 4, (&raw const dim_u32).cast());
        self.encoder
            .set_bytes(7, 4, (&raw const inv_sqrt_hidden).cast());
        self.encoder.set_bytes(8, 4, (&raw const epsilon).cast());
        let groups = u64::from(m_u32) * u64::from(layers_u32);
        self.encoder
            .dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(32, 1, 1));
        Ok(())
    }

    /// Fast SIMD-group `RMSNorm` over `[rows, width]`.
    /// # Errors
    /// Rejects invalid dimensions or undersized buffers.
    pub fn rms_norm(
        &mut self,
        input: &MetalDeviceBf16,
        scale: Option<&MetalBf16Buffer>,
        output: &MetalDeviceBf16,
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<(), MetalBf16Error> {
        let total = rows
            .checked_mul(width)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("rms_norm overflow".into()))?;
        check_len(input.elements, total, "rms input")?;
        check_len(output.elements, total, "rms output")?;
        if let Some(scale) = scale
            && scale.elements != width
        {
            return Err(MetalBf16Error::Shape("rms scale width mismatch".into()));
        }
        let width_u32 = checked_u32(width, "rms width")?;
        let has_scale = u32::from(scale.is_some());
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.rms);
        self.encoder.set_buffer(0, Some(&input.buffer), 0);
        self.encoder
            .set_buffer(1, Some(scale.map_or(&input.buffer, |s| &s.buffer)), 0);
        self.encoder.set_buffer(2, Some(&output.buffer), 0);
        self.encoder.set_bytes(3, 4, (&raw const width_u32).cast());
        self.encoder.set_bytes(4, 4, (&raw const epsilon).cast());
        self.encoder.set_bytes(5, 4, (&raw const has_scale).cast());
        self.encoder
            .dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// Fused in-place residual add (`residual += delta`) + `RMSNorm` into `normed_out`.
    /// # Errors
    /// Rejects invalid dimensions or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for fused add+RMS"
    )]
    pub fn add_rms_norm(
        &mut self,
        residual: &MetalDeviceBf16,
        delta: &MetalDeviceBf16,
        scale: &MetalBf16Buffer,
        normed_out: &MetalDeviceBf16,
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<(), MetalBf16Error> {
        let total = rows
            .checked_mul(width)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("add_rms_norm overflow".into()))?;
        check_len(residual.elements, total, "add_rms residual")?;
        check_len(delta.elements, total, "add_rms delta")?;
        check_len(normed_out.elements, total, "add_rms normed_out")?;
        if scale.elements != width {
            return Err(MetalBf16Error::Shape("add_rms scale width mismatch".into()));
        }
        let width_u32 = checked_u32(width, "add_rms width")?;
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.add_rms);
        self.encoder.set_buffer(0, Some(&residual.buffer), 0);
        self.encoder.set_buffer(1, Some(&delta.buffer), 0);
        self.encoder.set_buffer(2, Some(&scale.buffer), 0);
        self.encoder.set_buffer(3, Some(&normed_out.buffer), 0);
        self.encoder.set_bytes(4, 4, (&raw const width_u32).cast());
        self.encoder.set_bytes(5, 4, (&raw const epsilon).cast());
        self.encoder
            .dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// Fused in-place residual add (`residual += delta`) + optional multiply by `layer_scalar[0]`.
    /// # Errors
    /// Rejects empty/overflowing sizes or undersized buffers.
    pub fn residual_add_scale(
        &mut self,
        residual: &MetalDeviceBf16,
        delta: &MetalDeviceBf16,
        layer_scalar: Option<&MetalBf16Buffer>,
        total: usize,
    ) -> Result<(), MetalBf16Error> {
        if total == 0 {
            return Err(MetalBf16Error::Shape("residual_add_scale empty".into()));
        }
        check_len(residual.elements, total, "residual_add_scale residual")?;
        check_len(delta.elements, total, "residual_add_scale delta")?;
        if let Some(scalar) = layer_scalar
            && scalar.elements == 0
        {
            return Err(MetalBf16Error::Shape("empty layer_scalar".into()));
        }
        let total_u32 = checked_u32(total, "residual_add_scale total")?;
        let apply_scalar = u32::from(layer_scalar.is_some());
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.residual_add_scale);
        self.encoder.set_buffer(0, Some(&residual.buffer), 0);
        self.encoder.set_buffer(1, Some(&delta.buffer), 0);
        self.encoder.set_buffer(
            2,
            Some(layer_scalar.map_or(&residual.buffer, |s| &s.buffer)),
            0,
        );
        self.encoder
            .set_bytes(3, 4, (&raw const apply_scalar).cast());
        self.encoder.set_bytes(4, 4, (&raw const total_u32).cast());
        let total_u64 = u64::from(total_u32);
        self.encoder.dispatch_threads(
            MTLSize::new(total_u64, 1, 1),
            MTLSize::new(total_u64.min(256), 1, 1),
        );
        Ok(())
    }

    /// Fused bit-exact `rms_norm(delta, post_scale)` + `residual += .` + `rms_norm(residual, pre_scale)`.
    /// `delta` and `normed_out` may alias the same device buffer.
    /// # Errors
    /// Rejects invalid dimensions or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for fused post-norm + residual add + pre-norm"
    )]
    pub fn post_norm_add_pre_norm(
        &mut self,
        residual: &MetalDeviceBf16,
        delta: &MetalDeviceBf16,
        post_scale: &MetalBf16Buffer,
        pre_scale: &MetalBf16Buffer,
        normed_out: &MetalDeviceBf16,
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<(), MetalBf16Error> {
        let total = rows
            .checked_mul(width)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("post_norm_add_pre_norm overflow".into()))?;
        check_len(residual.elements, total, "post_norm_add_pre_norm residual")?;
        check_len(delta.elements, total, "post_norm_add_pre_norm delta")?;
        check_len(
            normed_out.elements,
            total,
            "post_norm_add_pre_norm normed_out",
        )?;
        if post_scale.elements != width || pre_scale.elements != width {
            return Err(MetalBf16Error::Shape(
                "post_norm_add_pre_norm scale width mismatch".into(),
            ));
        }
        let width_u32 = checked_u32(width, "post_norm_add_pre_norm width")?;
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.post_norm_add_pre_norm);
        self.encoder.set_buffer(0, Some(&residual.buffer), 0);
        self.encoder.set_buffer(1, Some(&delta.buffer), 0);
        self.encoder.set_buffer(2, Some(&post_scale.buffer), 0);
        self.encoder.set_buffer(3, Some(&pre_scale.buffer), 0);
        self.encoder.set_buffer(4, Some(&normed_out.buffer), 0);
        self.encoder.set_bytes(5, 4, (&raw const width_u32).cast());
        self.encoder.set_bytes(6, 4, (&raw const epsilon).cast());
        self.encoder
            .dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// Fused bit-exact `rms_norm(delta, norm_scale)` + `residual = (residual + .) * optional_scalar`.
    /// # Errors
    /// Rejects invalid dimensions or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for fused post-norm + residual add + optional layer scalar"
    )]
    pub fn rms_norm_residual_add_scale(
        &mut self,
        residual: &MetalDeviceBf16,
        delta: &MetalDeviceBf16,
        norm_scale: &MetalBf16Buffer,
        layer_scalar: Option<&MetalBf16Buffer>,
        rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<(), MetalBf16Error> {
        let total = rows
            .checked_mul(width)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("rms_norm_residual_add_scale overflow".into()))?;
        check_len(
            residual.elements,
            total,
            "rms_norm_residual_add_scale residual",
        )?;
        check_len(delta.elements, total, "rms_norm_residual_add_scale delta")?;
        if norm_scale.elements != width {
            return Err(MetalBf16Error::Shape(
                "rms_norm_residual_add_scale scale width mismatch".into(),
            ));
        }
        if let Some(scalar) = layer_scalar
            && scalar.elements == 0
        {
            return Err(MetalBf16Error::Shape("empty layer_scalar".into()));
        }
        let width_u32 = checked_u32(width, "rms_norm_residual_add_scale width")?;
        let apply_scalar = u32::from(layer_scalar.is_some());
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.rms_norm_residual_add_scale);
        self.encoder.set_buffer(0, Some(&residual.buffer), 0);
        self.encoder.set_buffer(1, Some(&delta.buffer), 0);
        self.encoder.set_buffer(2, Some(&norm_scale.buffer), 0);
        self.encoder.set_buffer(
            3,
            Some(layer_scalar.map_or(&residual.buffer, |s| &s.buffer)),
            0,
        );
        self.encoder
            .set_bytes(4, 4, (&raw const apply_scalar).cast());
        self.encoder.set_bytes(5, 4, (&raw const width_u32).cast());
        self.encoder.set_bytes(6, 4, (&raw const epsilon).cast());
        self.encoder
            .dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// In-place Q `RMSNorm` + split-half `RoPE` for `[rows, heads, head_dim]`.
    /// # Errors
    /// Rejects invalid dimensions or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for fused Q norm+RoPE"
    )]
    pub fn q_norm_rope(
        &mut self,
        q: &MetalDeviceBf16,
        q_norm: &MetalBf16Buffer,
        inv_freq: &MetalDeviceF32,
        row_pos: &MetalDeviceU32,
        rows: usize,
        heads: usize,
        head_dim: usize,
        epsilon: f32,
    ) -> Result<(), MetalBf16Error> {
        let total = rows
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(head_dim))
            .filter(|&n| n > 0 && head_dim.is_multiple_of(2))
            .ok_or_else(|| MetalBf16Error::Shape("q_norm_rope invalid shape".into()))?;
        check_len(q.elements, total, "q_norm_rope q")?;
        check_len(row_pos.elements, rows, "q_norm_rope row_pos")?;
        if q_norm.elements != head_dim || inv_freq.elements != head_dim / 2 {
            return Err(MetalBf16Error::Shape(
                "q_norm_rope norm or inv_freq length mismatch".into(),
            ));
        }
        let heads_u32 = checked_u32(heads, "q_norm_rope heads")?;
        let dim_u32 = checked_u32(head_dim, "q_norm_rope head_dim")?;
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.q_norm_rope);
        self.encoder.set_buffer(0, Some(&q.buffer), 0);
        self.encoder.set_buffer(1, Some(&q_norm.buffer), 0);
        self.encoder.set_buffer(2, Some(&inv_freq.buffer), 0);
        self.encoder.set_buffer(3, Some(&row_pos.buffer), 0);
        self.encoder.set_bytes(4, 4, (&raw const heads_u32).cast());
        self.encoder.set_bytes(5, 4, (&raw const dim_u32).cast());
        self.encoder.set_bytes(6, 4, (&raw const epsilon).cast());
        let groups = (rows as u64) * u64::from(heads_u32);
        self.encoder
            .dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(32, 1, 1));
        Ok(())
    }

    /// Fused K/V Norm + `RoPE` + paged scatter into KV pool `[phys_block, kv_head, 16, head_dim]`.
    /// # Errors
    /// Rejects invalid dimensions or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for fused paged KV norm+RoPE+write"
    )]
    pub fn kv_norm_rope_write(
        &mut self,
        k_in: &MetalDeviceBf16,
        v_in: &MetalDeviceBf16,
        k_norm: &MetalBf16Buffer,
        inv_freq: &MetalDeviceF32,
        row_slot: &MetalDeviceU32,
        row_pos: &MetalDeviceU32,
        block_table: &MetalDeviceU32,
        k_cache: &MetalDeviceBf16,
        v_cache: &MetalDeviceBf16,
        rows: usize,
        kv_heads: usize,
        head_dim: usize,
        max_pages: usize,
        epsilon: f32,
    ) -> Result<(), MetalBf16Error> {
        let total = rows
            .checked_mul(kv_heads)
            .and_then(|n| n.checked_mul(head_dim))
            .filter(|&n| n > 0 && head_dim.is_multiple_of(2))
            .ok_or_else(|| MetalBf16Error::Shape("kv_norm_rope_write invalid shape".into()))?;
        check_len(k_in.elements, total, "kv_norm_rope_write k_in")?;
        check_len(v_in.elements, total, "kv_norm_rope_write v_in")?;
        check_len(row_slot.elements, rows, "kv_norm_rope_write row_slot")?;
        check_len(row_pos.elements, rows, "kv_norm_rope_write row_pos")?;
        check_len(
            block_table.elements,
            max_pages,
            "kv_norm_rope_write block_table",
        )?;
        if k_norm.elements != head_dim || inv_freq.elements != head_dim / 2 {
            return Err(MetalBf16Error::Shape(
                "kv_norm_rope_write norm or inv_freq length mismatch".into(),
            ));
        }
        let kv_heads_u32 = checked_u32(kv_heads, "kv_norm_rope_write kv_heads")?;
        let dim_u32 = checked_u32(head_dim, "kv_norm_rope_write head_dim")?;
        let max_pages_u32 = checked_u32(max_pages, "kv_norm_rope_write max_pages")?;
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.kv_norm_rope_write);
        self.encoder.set_buffer(0, Some(&k_in.buffer), 0);
        self.encoder.set_buffer(1, Some(&v_in.buffer), 0);
        self.encoder.set_buffer(2, Some(&k_norm.buffer), 0);
        self.encoder.set_buffer(3, Some(&inv_freq.buffer), 0);
        self.encoder.set_buffer(4, Some(&row_slot.buffer), 0);
        self.encoder.set_buffer(5, Some(&row_pos.buffer), 0);
        self.encoder.set_buffer(6, Some(&block_table.buffer), 0);
        self.encoder.set_buffer(7, Some(&k_cache.buffer), 0);
        self.encoder.set_buffer(8, Some(&v_cache.buffer), 0);
        self.encoder
            .set_bytes(9, 4, (&raw const kv_heads_u32).cast());
        self.encoder.set_bytes(10, 4, (&raw const dim_u32).cast());
        self.encoder
            .set_bytes(11, 4, (&raw const max_pages_u32).cast());
        self.encoder.set_bytes(12, 4, (&raw const epsilon).cast());
        let groups = (rows as u64) * u64::from(kv_heads_u32);
        self.encoder
            .dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(32, 1, 1));
        Ok(())
    }

    /// Online-softmax paged `FlashAttention` (`PAGE_TOKENS = 16`) for arbitrary packed batches.
    /// # Errors
    /// Rejects invalid geometry (`head_dim > 512`, non-divisible GQA) or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for batched paged FlashAttention"
    )]
    pub fn flash_attention(
        &mut self,
        q: &MetalDeviceBf16,
        k_cache: &MetalDeviceBf16,
        v_cache: &MetalDeviceBf16,
        output: &MetalDeviceBf16,
        row_slot: &MetalDeviceU32,
        row_pos: &MetalDeviceU32,
        block_table: &MetalDeviceU32,
        rows: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        max_pages: usize,
        window: Option<usize>,
    ) -> Result<(), MetalBf16Error> {
        let total = rows
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(head_dim))
            .filter(|&n| n > 0 && kv_heads > 0 && heads.is_multiple_of(kv_heads))
            .ok_or_else(|| MetalBf16Error::Shape("flash_attention invalid shape".into()))?;
        if head_dim > 512 {
            return Err(MetalBf16Error::Shape(
                "flash_attention requires head_dim <= 512".into(),
            ));
        }
        check_len(q.elements, total, "flash_attention q")?;
        check_len(output.elements, total, "flash_attention output")?;
        check_len(row_slot.elements, rows, "flash_attention row_slot")?;
        check_len(row_pos.elements, rows, "flash_attention row_pos")?;
        check_len(
            block_table.elements,
            max_pages,
            "flash_attention block_table",
        )?;
        let heads_u32 = checked_u32(heads, "flash_attention heads")?;
        let kv_heads_u32 = checked_u32(kv_heads, "flash_attention kv_heads")?;
        let dim_u32 = checked_u32(head_dim, "flash_attention head_dim")?;
        let max_pages_u32 = checked_u32(max_pages, "flash_attention max_pages")?;
        let window_u32 = checked_u32(window.unwrap_or(0), "flash_attention window")?;
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.flash_attn);
        self.encoder.set_buffer(0, Some(&q.buffer), 0);
        self.encoder.set_buffer(1, Some(&k_cache.buffer), 0);
        self.encoder.set_buffer(2, Some(&v_cache.buffer), 0);
        self.encoder.set_buffer(3, Some(&output.buffer), 0);
        self.encoder.set_buffer(4, Some(&row_slot.buffer), 0);
        self.encoder.set_buffer(5, Some(&row_pos.buffer), 0);
        self.encoder.set_buffer(6, Some(&block_table.buffer), 0);
        self.encoder.set_bytes(7, 4, (&raw const heads_u32).cast());
        self.encoder
            .set_bytes(8, 4, (&raw const kv_heads_u32).cast());
        self.encoder.set_bytes(9, 4, (&raw const dim_u32).cast());
        self.encoder
            .set_bytes(10, 4, (&raw const max_pages_u32).cast());
        self.encoder
            .set_bytes(11, 4, (&raw const window_u32).cast());
        let groups = (rows as u64) * u64::from(heads_u32);
        self.encoder
            .dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// Exact table-backed `GeGLU` (`table[gate] * up[up_offset..]`) with non-finite gate flagging.
    /// # Errors
    /// Rejects invalid sizes or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for table-backed GeGLU"
    )]
    pub fn geglu_table(
        &mut self,
        gate: &MetalDeviceBf16,
        up: &MetalDeviceBf16,
        up_offset_elems: usize,
        table: &MetalBf16Buffer,
        output: &MetalDeviceBf16,
        error_flag: &MetalDeviceU32,
        total: usize,
    ) -> Result<(), MetalBf16Error> {
        if total == 0 || table.elements != 65_536 {
            return Err(MetalBf16Error::Shape(
                "geglu_table invalid total or table size".into(),
            ));
        }
        check_len(gate.elements, total, "geglu gate")?;
        let up_end = up_offset_elems
            .checked_add(total)
            .ok_or_else(|| MetalBf16Error::Shape("geglu up offset overflow".into()))?;
        check_len(up.elements, up_end, "geglu up")?;
        check_len(output.elements, total, "geglu output")?;
        check_len(error_flag.elements, 1, "geglu error_flag")?;
        let total_u32 = checked_u32(total, "geglu total")?;
        let up_byte_offset = u64::try_from(
            up_offset_elems
                .checked_mul(2)
                .ok_or_else(|| MetalBf16Error::Shape("geglu byte offset overflow".into()))?,
        )
        .map_err(|_| MetalBf16Error::Shape("geglu byte offset exceeds u64".into()))?;
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.geglu_table);
        self.encoder.set_buffer(0, Some(&gate.buffer), 0);
        self.encoder.set_buffer(1, Some(&up.buffer), up_byte_offset);
        self.encoder.set_buffer(2, Some(&table.buffer), 0);
        self.encoder.set_buffer(3, Some(&output.buffer), 0);
        self.encoder.set_buffer(4, Some(&error_flag.buffer), 0);
        self.encoder.set_bytes(5, 4, (&raw const total_u32).cast());
        let total_u64 = u64::from(total_u32);
        self.encoder.dispatch_threads(
            MTLSize::new(total_u64, 1, 1),
            MTLSize::new(total_u64.min(256), 1, 1),
        );
        Ok(())
    }

    /// Gather `out_rows` rows selected by `row_indices` from `src` (at `src_offset_elems`) into `dst`.
    /// # Errors
    /// Rejects invalid shapes or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for output row gather"
    )]
    pub fn gather_rows(
        &mut self,
        src: &MetalDeviceBf16,
        src_offset_elems: usize,
        row_indices: &MetalDeviceU32,
        dst: &MetalDeviceBf16,
        total_rows: usize,
        out_rows: usize,
        width: usize,
    ) -> Result<(), MetalBf16Error> {
        let in_req = total_rows
            .checked_mul(width)
            .filter(|&n| n > 0)
            .and_then(|n| src_offset_elems.checked_add(n))
            .ok_or_else(|| MetalBf16Error::Shape("gather_rows input overflow".into()))?;
        let out_req = out_rows
            .checked_mul(width)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("gather_rows output overflow".into()))?;
        check_len(src.elements, in_req, "gather_rows src")?;
        check_len(row_indices.elements, out_rows, "gather_rows indices")?;
        check_len(dst.elements, out_req, "gather_rows dst")?;
        let width_u32 = checked_u32(width, "gather_rows width")?;
        let src_byte_offset = u64::try_from(
            src_offset_elems
                .checked_mul(2)
                .ok_or_else(|| MetalBf16Error::Shape("gather_rows byte offset overflow".into()))?,
        )
        .map_err(|_| MetalBf16Error::Shape("gather_rows byte offset exceeds u64".into()))?;
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.gather_rows);
        self.encoder
            .set_buffer(0, Some(&src.buffer), src_byte_offset);
        self.encoder.set_buffer(1, Some(&row_indices.buffer), 0);
        self.encoder.set_buffer(2, Some(&dst.buffer), 0);
        self.encoder.set_bytes(3, 4, (&raw const width_u32).cast());
        self.encoder
            .dispatch_thread_groups(MTLSize::new(out_rows as u64, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// Gather `out_rows` rows selected by `output_rows` from `hidden` and apply `scale` `RMSNorm`.
    /// # Errors
    /// Rejects invalid shapes or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for output row gather + final RMSNorm"
    )]
    pub fn gather_rows_rms(
        &mut self,
        hidden: &MetalDeviceBf16,
        output_rows: &MetalDeviceU32,
        scale: &MetalBf16Buffer,
        output: &MetalDeviceBf16,
        total_rows: usize,
        out_rows: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<(), MetalBf16Error> {
        let in_req = total_rows
            .checked_mul(width)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("gather_rows_rms input overflow".into()))?;
        let out_req = out_rows
            .checked_mul(width)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("gather_rows_rms output overflow".into()))?;
        check_len(hidden.elements, in_req, "gather_rows_rms hidden")?;
        check_len(output_rows.elements, out_rows, "gather_rows_rms indices")?;
        check_len(output.elements, out_req, "gather_rows_rms output")?;
        if scale.elements != width {
            return Err(MetalBf16Error::Shape(
                "gather_rows_rms scale width mismatch".into(),
            ));
        }
        let width_u32 = checked_u32(width, "gather_rows_rms width")?;
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.gather_rows_rms);
        self.encoder.set_buffer(0, Some(&hidden.buffer), 0);
        self.encoder.set_buffer(1, Some(&output_rows.buffer), 0);
        self.encoder.set_buffer(2, Some(&scale.buffer), 0);
        self.encoder.set_buffer(3, Some(&output.buffer), 0);
        self.encoder.set_bytes(4, 4, (&raw const width_u32).cast());
        self.encoder.set_bytes(5, 4, (&raw const epsilon).cast());
        self.encoder
            .dispatch_thread_groups(MTLSize::new(out_rows as u64, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// Apply Gemma 4 softcap in-place to `logits[rows, vocab]` and compute deterministic `argmax_out[rows]`.
    /// # Errors
    /// Rejects invalid shapes, non-positive cap, or undersized buffers.
    pub fn softcap_argmax(
        &mut self,
        logits: &MetalDeviceBf16,
        argmax_out: &MetalDeviceU32,
        error_flag: &MetalDeviceU32,
        rows: usize,
        vocab: usize,
        cap: f32,
    ) -> Result<(), MetalBf16Error> {
        if !cap.is_finite() || cap <= 0.0 {
            return Err(MetalBf16Error::Invalid("softcap must be finite positive"));
        }
        let total = rows
            .checked_mul(vocab)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("softcap_argmax overflow".into()))?;
        check_len(logits.elements, total, "softcap_argmax logits")?;
        check_len(argmax_out.elements, rows, "softcap_argmax argmax_out")?;
        check_len(error_flag.elements, 1, "softcap_argmax error_flag")?;
        let vocab_u32 = checked_u32(vocab, "softcap_argmax vocab")?;
        if rows <= SOFTCAP_MAX_PARALLEL_ROWS && vocab >= 4096 {
            let blocks = SOFTCAP_BLOCKS_PER_ROW;
            self.encoder
                .set_compute_pipeline_state(&self.executor.fast.softcap_argmax_stage1);
            self.encoder.set_buffer(0, Some(&logits.buffer), 0);
            self.encoder
                .set_buffer(1, Some(&self.executor.fast.softcap_partial_vals), 0);
            self.encoder
                .set_buffer(2, Some(&self.executor.fast.softcap_partial_idxs), 0);
            self.encoder.set_buffer(3, Some(&error_flag.buffer), 0);
            self.encoder.set_bytes(4, 4, (&raw const vocab_u32).cast());
            self.encoder.set_bytes(5, 4, (&raw const cap).cast());
            self.encoder.set_bytes(6, 4, (&raw const blocks).cast());
            self.encoder.dispatch_thread_groups(
                MTLSize::new(u64::from(blocks), rows as u64, 1),
                MTLSize::new(256, 1, 1),
            );

            self.encoder
                .set_compute_pipeline_state(&self.executor.fast.softcap_argmax_stage2);
            self.encoder
                .set_buffer(0, Some(&self.executor.fast.softcap_partial_vals), 0);
            self.encoder
                .set_buffer(1, Some(&self.executor.fast.softcap_partial_idxs), 0);
            self.encoder.set_buffer(2, Some(&argmax_out.buffer), 0);
            self.encoder.set_bytes(3, 4, (&raw const blocks).cast());
            self.encoder.dispatch_thread_groups(
                MTLSize::new(rows as u64, 1, 1),
                MTLSize::new(u64::from(blocks), 1, 1),
            );
        } else {
            self.encoder
                .set_compute_pipeline_state(&self.executor.fast.softcap_argmax);
            self.encoder.set_buffer(0, Some(&logits.buffer), 0);
            self.encoder.set_buffer(1, Some(&argmax_out.buffer), 0);
            self.encoder.set_buffer(2, Some(&error_flag.buffer), 0);
            self.encoder.set_bytes(3, 4, (&raw const vocab_u32).cast());
            self.encoder.set_bytes(4, 4, (&raw const cap).cast());
            self.encoder
                .dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        }
        Ok(())
    }

    /// Fused `M = 1` BF16 dual linear projection (`out_a = input @ weight_a^T`, `out_b = input @ weight_b^T`).
    /// # Errors
    /// Rejects shape mismatches or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for fused M=1 BF16 dual GEMV"
    )]
    pub fn dual_linear_m1(
        &mut self,
        input: &MetalDeviceBf16,
        weight_a: &MetalBf16Buffer,
        weight_b: &MetalBf16Buffer,
        out_a: &MetalDeviceBf16,
        out_b: &MetalDeviceBf16,
        columns: usize,
        inner: usize,
    ) -> Result<(), MetalBf16Error> {
        let weight_elems = columns
            .checked_mul(inner)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("dual_linear_m1 overflow".into()))?;
        if weight_a.elements != weight_elems || weight_b.elements != weight_elems {
            return Err(MetalBf16Error::Shape(
                "dual_linear_m1 weight shape mismatch".into(),
            ));
        }
        check_len(input.elements, inner, "dual_linear_m1 input")?;
        check_len(out_a.elements, columns, "dual_linear_m1 out_a")?;
        check_len(out_b.elements, columns, "dual_linear_m1 out_b")?;
        let n_u32 = checked_u32(columns, "dual_linear_m1 N")?;
        let k_u32 = checked_u32(inner, "dual_linear_m1 K")?;
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.dual_gemv_m1);
        self.encoder.set_buffer(0, Some(&input.buffer), 0);
        self.encoder.set_buffer(1, Some(&weight_a.buffer), 0);
        self.encoder.set_buffer(2, Some(&weight_b.buffer), 0);
        self.encoder.set_buffer(3, Some(&out_a.buffer), 0);
        self.encoder.set_buffer(4, Some(&out_b.buffer), 0);
        self.encoder.set_bytes(5, 4, (&raw const n_u32).cast());
        self.encoder.set_bytes(6, 4, (&raw const k_u32).cast());
        let tgs = u64::from(n_u32).div_ceil(4);
        self.encoder
            .dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// Fused `M = 1` BF16 `Gate + Up + exact table GeGLU`.
    /// # Errors
    /// Rejects shape mismatches or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for fused M=1 BF16 Gate+Up+GeGLU"
    )]
    pub fn gate_up_geglu_m1(
        &mut self,
        input: &MetalDeviceBf16,
        gate_w: &MetalBf16Buffer,
        up_w: &MetalBf16Buffer,
        gelu_table: &MetalBf16Buffer,
        output: &MetalDeviceBf16,
        error_flag: &MetalDeviceU32,
        columns: usize,
        inner: usize,
    ) -> Result<(), MetalBf16Error> {
        let weight_elems = columns
            .checked_mul(inner)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("gate_up_geglu_m1 overflow".into()))?;
        if gate_w.elements != weight_elems
            || up_w.elements != weight_elems
            || gelu_table.elements != 65_536
        {
            return Err(MetalBf16Error::Shape(
                "gate_up_geglu_m1 weight or GELU table mismatch".into(),
            ));
        }
        check_len(input.elements, inner, "gate_up_geglu_m1 input")?;
        check_len(output.elements, columns, "gate_up_geglu_m1 output")?;
        check_len(error_flag.elements, 1, "gate_up_geglu_m1 error_flag")?;
        let n_u32 = checked_u32(columns, "gate_up_geglu_m1 N")?;
        let k_u32 = checked_u32(inner, "gate_up_geglu_m1 K")?;
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.gate_up_geglu_m1);
        self.encoder.set_buffer(0, Some(&input.buffer), 0);
        self.encoder.set_buffer(1, Some(&gate_w.buffer), 0);
        self.encoder.set_buffer(2, Some(&up_w.buffer), 0);
        self.encoder.set_buffer(3, Some(&gelu_table.buffer), 0);
        self.encoder.set_buffer(4, Some(&output.buffer), 0);
        self.encoder.set_buffer(5, Some(&error_flag.buffer), 0);
        self.encoder.set_bytes(6, 4, (&raw const n_u32).cast());
        self.encoder.set_bytes(7, 4, (&raw const k_u32).cast());
        let tgs = u64::from(n_u32).div_ceil(4);
        self.encoder
            .dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// Fused `M = 1` BF16 linear projection + exact table `GeGLU` with `up_in[up_offset_elems..]`.
    /// # Errors
    /// Rejects shape mismatches or undersized buffers.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit GPU kernel bindings for fused M=1 BF16 linear+GeGLU"
    )]
    pub fn linear_geglu_m1(
        &mut self,
        input: &MetalDeviceBf16,
        weight: &MetalBf16Buffer,
        up_in: &MetalDeviceBf16,
        up_offset_elems: usize,
        gelu_table: &MetalBf16Buffer,
        output: &MetalDeviceBf16,
        error_flag: &MetalDeviceU32,
        columns: usize,
        inner: usize,
    ) -> Result<(), MetalBf16Error> {
        let weight_elems = columns
            .checked_mul(inner)
            .filter(|&n| n > 0)
            .ok_or_else(|| MetalBf16Error::Shape("linear_geglu_m1 overflow".into()))?;
        if weight.elements != weight_elems || gelu_table.elements != 65_536 {
            return Err(MetalBf16Error::Shape(
                "linear_geglu_m1 weight or GELU table mismatch".into(),
            ));
        }
        let up_end = up_offset_elems
            .checked_add(columns)
            .ok_or_else(|| MetalBf16Error::Shape("linear_geglu_m1 up offset overflow".into()))?;
        check_len(input.elements, inner, "linear_geglu_m1 input")?;
        check_len(up_in.elements, up_end, "linear_geglu_m1 up_in")?;
        check_len(output.elements, columns, "linear_geglu_m1 output")?;
        check_len(error_flag.elements, 1, "linear_geglu_m1 error_flag")?;
        let up_byte_offset = u64::try_from(
            up_offset_elems
                .checked_mul(2)
                .ok_or_else(|| MetalBf16Error::Shape("up byte offset overflow".into()))?,
        )
        .map_err(|_| MetalBf16Error::Shape("up byte offset exceeds u64".into()))?;
        let n_u32 = checked_u32(columns, "linear_geglu_m1 N")?;
        let k_u32 = checked_u32(inner, "linear_geglu_m1 K")?;
        self.encoder
            .set_compute_pipeline_state(&self.executor.fast.linear_geglu_m1);
        self.encoder.set_buffer(0, Some(&input.buffer), 0);
        self.encoder.set_buffer(1, Some(&weight.buffer), 0);
        self.encoder
            .set_buffer(2, Some(&up_in.buffer), up_byte_offset);
        self.encoder.set_buffer(3, Some(&gelu_table.buffer), 0);
        self.encoder.set_buffer(4, Some(&output.buffer), 0);
        self.encoder.set_buffer(5, Some(&error_flag.buffer), 0);
        self.encoder.set_bytes(6, 4, (&raw const n_u32).cast());
        self.encoder.set_bytes(7, 4, (&raw const k_u32).cast());
        let tgs = u64::from(n_u32).div_ceil(8);
        self.encoder
            .dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    }

    /// End encoding, commit the single command buffer, and wait for terminal completion.
    /// # Errors
    /// Reports GPU execution failures; the execution owner faults on error.
    pub fn submit(mut self, operation: &str) -> Result<(), MetalBf16Error> {
        self.encoder.end_encoding();
        self.ended = true;
        self.command.submit(operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(v: f32) -> u16 {
        let bits = v.to_bits();
        if (bits & 0x7fff_ffff) > 0x7f80_0000 {
            return 0x7fc0;
        }
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
        (rounded >> 16) as u16
    }

    fn f32_from_bf16(b: u16) -> f32 {
        f32::from_bits(u32::from(b) << 16)
    }

    #[expect(clippy::cast_precision_loss, reason = "small bounded test indices")]
    fn small_f32(value: usize) -> f32 {
        value as f32
    }

    fn cpu_matmul(
        input: &[u16],
        weight: &[u16],
        rows: usize,
        cols: usize,
        inner: usize,
    ) -> Vec<u16> {
        let mut out = vec![0_u16; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let mut sum = 0.0_f32;
                for i in 0..inner {
                    sum +=
                        f32_from_bf16(input[r * inner + i]) * f32_from_bf16(weight[c * inner + i]);
                }
                out[r * cols + c] = bf16(sum);
            }
        }
        out
    }

    #[test]
    fn fast_linear_matches_cpu_across_gemv_and_tiled_gemm_shapes() {
        let exec = match MetalBf16Executor::new() {
            Ok(e) => e,
            Err(MetalBf16Error::Device) => return,
            Err(e) => panic!("Metal init: {e}"),
        };
        // Exercise M=1, M=2, M=3, M=4, M=5, M=8 (GEMV) and M=9, M=18, M=65 (tiled GEMM),
        // with aligned and unaligned N and K!
        let cases = [
            (1, 17, 16),
            (1, 131, 73),
            (2, 64, 128),
            (3, 19, 36),
            (4, 131, 64),
            (5, 33, 20),
            (8, 128, 256),
            (9, 131, 73),
            (18, 195, 128),
            (65, 130, 144),
        ];
        for (rows, cols, inner) in cases {
            let x_vals: Vec<u16> = (0..rows * inner)
                .map(|i| bf16((small_f32(i % 13) - 6.0) * 0.125))
                .collect();
            let w_vals: Vec<u16> = (0..cols * inner)
                .map(|i| bf16((small_f32((i * 7 + 3) % 11) - 5.0) * 0.125))
                .collect();
            let expected = cpu_matmul(&x_vals, &w_vals, rows, cols, inner);

            let mut x_dev = exec.alloc_bf16(rows * inner).unwrap();
            x_dev.write(&x_vals).unwrap();
            let w_dev = exec.upload(&w_vals).unwrap();
            let y_dev = exec.alloc_bf16(rows * cols).unwrap();

            let mut session = exec.begin_fast_session().unwrap();
            session
                .linear(&x_dev, &w_dev, &y_dev, rows, cols, inner)
                .unwrap();
            session.submit("test fast linear").unwrap();

            let actual = y_dev.as_slice(rows * cols);
            for (idx, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                let fa = f32_from_bf16(a);
                let fe = f32_from_bf16(e);
                let diff = (fa - fe).abs();
                let tol = 1e-2_f32.max(fe.abs() * 2e-2);
                assert!(
                    diff <= tol,
                    "shape ({rows},{cols},{inner}) idx {idx}: actual={fa} ({a:#06x}) expected={fe} ({e:#06x})"
                );
            }
        }
    }

    #[test]
    fn fast_flash_attention_matches_reference_across_sliding_and_full_batches() {
        let exec = match MetalBf16Executor::new() {
            Ok(e) => e,
            Err(MetalBf16Error::Device) => return,
            Err(e) => panic!("Metal init: {e}"),
        };
        let heads = 4;
        let kv_heads = 2;
        let max_pages = 2; // 32 positions per sequence
        let num_blocks = 4; // non-identity physical page mapping

        for (head_dim, window) in [(64_usize, Some(4_usize)), (256, Some(5)), (512, None)] {
            // Non-identity block table: slot 0 -> [3, 1], slot 1 -> [2, 0]
            let block_table_host = [3_u32, 1, 2, 0];
            let kv_total = num_blocks * kv_heads * 16 * head_dim;
            let mut k_host = vec![0_u16; kv_total];
            let mut v_host = vec![0_u16; kv_total];
            for (i, slot) in k_host.iter_mut().enumerate() {
                *slot = bf16((small_f32((i * 3 + 1) % 17) - 8.0) * 0.05);
            }
            for (i, slot) in v_host.iter_mut().enumerate() {
                *slot = bf16((small_f32((i * 5 + 2) % 19) - 9.0) * 0.05);
            }
            // Batch of 3 tokens spanning across page boundary (pos=18 -> page 1!)
            let slots = [0_u32, 1, 0];
            let positions = [18_u32, 3, 0];
            let m = slots.len();
            let q_host: Vec<u16> = (0..m * heads * head_dim)
                .map(|i| bf16((small_f32((i * 7 + 1) % 15) - 7.0) * 0.05))
                .collect();

            let mut k_dev = exec.alloc_bf16(kv_total).unwrap();
            k_dev.write(&k_host).unwrap();
            let mut v_dev = exec.alloc_bf16(kv_total).unwrap();
            v_dev.write(&v_host).unwrap();
            let mut q_dev = exec.alloc_bf16(q_host.len()).unwrap();
            q_dev.write(&q_host).unwrap();
            let out_dev = exec.alloc_bf16(q_host.len()).unwrap();
            let mut slot_dev = exec.alloc_u32(m).unwrap();
            slot_dev.write(&slots).unwrap();
            let mut pos_dev = exec.alloc_u32(m).unwrap();
            pos_dev.write(&positions).unwrap();
            let mut bt_dev = exec.alloc_u32(block_table_host.len()).unwrap();
            bt_dev.write(&block_table_host).unwrap();

            let mut session = exec.begin_fast_session().unwrap();
            session
                .flash_attention(
                    &q_dev, &k_dev, &v_dev, &out_dev, &slot_dev, &pos_dev, &bt_dev, m, heads,
                    kv_heads, head_dim, max_pages, window,
                )
                .unwrap();
            session.submit("test flash attention").unwrap();

            let actual = out_dev.as_slice(q_host.len());
            // CPU exact reference over paged block table
            for row in 0..m {
                let slot = slots[row] as usize;
                let pos = positions[row] as usize;
                let first = window.map_or(0, |w| pos.saturating_sub(w - 1));
                for head in 0..heads {
                    let kh = head / (heads / kv_heads);
                    let q_off = (row * heads + head) * head_dim;
                    let mut scores = Vec::new();
                    let mut max_s = f32::NEG_INFINITY;
                    for j in first..=pos {
                        let phys = block_table_host[slot * max_pages + (j >> 4)] as usize;
                        let kv_off = ((phys * kv_heads + kh) * 16 + (j & 15)) * head_dim;
                        let mut dot = 0.0_f32;
                        for d in 0..head_dim {
                            dot += f32_from_bf16(q_host[q_off + d])
                                * f32_from_bf16(k_host[kv_off + d]);
                        }
                        let s = f32_from_bf16(bf16(dot));
                        max_s = max_s.max(s);
                        scores.push((j, s));
                    }
                    let denom: f32 = scores.iter().map(|&(_, s)| (s - max_s).exp()).sum();
                    for d in 0..head_dim {
                        let mut acc = 0.0_f32;
                        for &(j, s) in &scores {
                            let phys = block_table_host[slot * max_pages + (j >> 4)] as usize;
                            let kv_off = ((phys * kv_heads + kh) * 16 + (j & 15)) * head_dim;
                            let w = (s - max_s).exp() / denom;
                            acc += w * f32_from_bf16(v_host[kv_off + d]);
                        }
                        let exp_f = f32_from_bf16(bf16(acc));
                        let act_f = f32_from_bf16(actual[q_off + d]);
                        assert!(
                            (act_f - exp_f).abs() <= 2e-2,
                            "dim={head_dim} row={row} head={head} d={d}: act={act_f} exp={exp_f}"
                        );
                    }
                }
            }
        }
    }
}
