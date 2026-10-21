//! Prepared, charged activation and full-history continuation storage.
use super::DEVICE_ONLY_STAGING;
use crate::{AttentionKind, Gemma4Config, Gemma4Error};
use cudarc::driver::CudaStream;
use engine_nvidia::CudaBf16Matrix;
use ribn_foundation::BytePool;
use std::sync::Arc;

/// Device-resident keys and values for one KV owner layer.
pub(super) struct DeviceCache {
    pub(super) keys: CudaBf16Matrix,
    pub(super) values: CudaBf16Matrix,
}

/// Per-geometry working set; local and global layers differ in head width.
pub(super) struct GeometryBuffers {
    pub(super) width: usize,
    pub(super) cos: CudaBf16Matrix,
    pub(super) sin: CudaBf16Matrix,
    pub(super) normed: CudaBf16Matrix,
    pub(super) queries: CudaBf16Matrix,
    pub(super) query_norm: CudaBf16Matrix,
    pub(super) query_rotary: CudaBf16Matrix,
    pub(super) keys: CudaBf16Matrix,
    pub(super) key_norm: CudaBf16Matrix,
    pub(super) key_rotary: CudaBf16Matrix,
    pub(super) values: CudaBf16Matrix,
    pub(super) value_norm: CudaBf16Matrix,
    pub(super) attended: CudaBf16Matrix,
    pub(super) projected: CudaBf16Matrix,
    pub(super) post_attention: CudaBf16Matrix,
    pub(super) ffn_norm: CudaBf16Matrix,
    pub(super) gate: CudaBf16Matrix,
    pub(super) up: CudaBf16Matrix,
    pub(super) activated: CudaBf16Matrix,
    pub(super) product: CudaBf16Matrix,
    pub(super) down: CudaBf16Matrix,
    pub(super) post_feedforward: CudaBf16Matrix,
    pub(super) ple_gate: CudaBf16Matrix,
    pub(super) ple_activated: CudaBf16Matrix,
    pub(super) ple_product: CudaBf16Matrix,
    pub(super) ple_projected: CudaBf16Matrix,
    pub(super) post_ple: CudaBf16Matrix,
}

/// Model-wide buffers shared by every layer.
pub(super) struct Buffers {
    pub(super) hidden: CudaBf16Matrix,
    pub(super) hidden_next: CudaBf16Matrix,
    pub(super) final_input: CudaBf16Matrix,
    pub(super) final_norm: CudaBf16Matrix,
    pub(super) logits: CudaBf16Matrix,
    pub(super) ple_projection: CudaBf16Matrix,
    pub(super) ple_projected_scaled: CudaBf16Matrix,
    pub(super) ple_normed: CudaBf16Matrix,
    pub(super) ple_context: CudaBf16Matrix,
    pub(super) local: GeometryBuffers,
    pub(super) global: GeometryBuffers,
}

impl Buffers {
    pub(super) fn new(
        config: &Gemma4Config,
        rows: usize,
        stream: &Arc<CudaStream>,
        pool: &Arc<BytePool>,
    ) -> Result<Self, Gemma4Error> {
        let hidden = config.hidden_size();
        let packed = config.packed_ple_dim();
        let ple = config.ple_dim();
        let bytes = |columns: usize| {
            rows.checked_mul(columns)
                .and_then(|n| n.checked_mul(2))
                .ok_or_else(|| Gemma4Error::new("CUDA decoder", "host staging extent overflow"))
        };
        Ok(Self {
            hidden: matrix(rows, hidden, stream, pool, bytes(hidden)?)?,
            hidden_next: matrix(rows, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            final_input: matrix(1, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            final_norm: matrix(1, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            logits: matrix(
                1,
                config.vocab_size(),
                stream,
                pool,
                config.vocab_size().checked_mul(2).ok_or_else(|| {
                    Gemma4Error::new("CUDA decoder", "logit staging extent overflow")
                })?,
            )?,
            ple_projection: matrix(rows, packed, stream, pool, DEVICE_ONLY_STAGING)?,
            ple_projected_scaled: matrix(rows, packed, stream, pool, DEVICE_ONLY_STAGING)?,
            ple_normed: matrix(rows, packed, stream, pool, bytes(packed)?)?,
            ple_context: matrix(rows, ple, stream, pool, bytes(ple)?)?,
            local: GeometryBuffers::new(config, AttentionWidth::Local, rows, stream, pool)?,
            global: GeometryBuffers::new(config, AttentionWidth::Global, rows, stream, pool)?,
        })
    }
}

impl GeometryBuffers {
    fn new(
        config: &Gemma4Config,
        kind: AttentionWidth,
        rows: usize,
        stream: &Arc<CudaStream>,
        pool: &Arc<BytePool>,
    ) -> Result<Self, Gemma4Error> {
        let width = kind.width(config);
        let heads = config.num_attention_heads();
        let kv_heads = config.num_key_value_heads();
        let hidden = config.hidden_size();
        let intermediate = config.intermediate_size();
        let ple = config.ple_dim();
        let q = heads * width;
        let kv = kv_heads * width;
        Ok(Self {
            width,
            cos: matrix(rows, width, stream, pool, DEVICE_ONLY_STAGING)?,
            sin: matrix(rows, width, stream, pool, DEVICE_ONLY_STAGING)?,
            normed: matrix(rows, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            queries: matrix(rows, q, stream, pool, DEVICE_ONLY_STAGING)?,
            query_norm: matrix(rows, q, stream, pool, DEVICE_ONLY_STAGING)?,
            query_rotary: matrix(rows, q, stream, pool, DEVICE_ONLY_STAGING)?,
            keys: matrix(rows, kv, stream, pool, DEVICE_ONLY_STAGING)?,
            key_norm: matrix(rows, kv, stream, pool, DEVICE_ONLY_STAGING)?,
            key_rotary: matrix(rows, kv, stream, pool, DEVICE_ONLY_STAGING)?,
            values: matrix(rows, kv, stream, pool, DEVICE_ONLY_STAGING)?,
            value_norm: matrix(rows, kv, stream, pool, DEVICE_ONLY_STAGING)?,
            attended: matrix(rows, q, stream, pool, DEVICE_ONLY_STAGING)?,
            projected: matrix(rows, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            post_attention: matrix(rows, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            ffn_norm: matrix(rows, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            gate: matrix(rows, intermediate, stream, pool, DEVICE_ONLY_STAGING)?,
            up: matrix(rows, intermediate, stream, pool, DEVICE_ONLY_STAGING)?,
            activated: matrix(rows, intermediate, stream, pool, DEVICE_ONLY_STAGING)?,
            product: matrix(rows, intermediate, stream, pool, DEVICE_ONLY_STAGING)?,
            down: matrix(rows, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            post_feedforward: matrix(rows, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            ple_gate: matrix(rows, ple, stream, pool, DEVICE_ONLY_STAGING)?,
            ple_activated: matrix(rows, ple, stream, pool, DEVICE_ONLY_STAGING)?,
            ple_product: matrix(rows, ple, stream, pool, DEVICE_ONLY_STAGING)?,
            ple_projected: matrix(rows, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
            post_ple: matrix(rows, hidden, stream, pool, DEVICE_ONLY_STAGING)?,
        })
    }
}

#[derive(Clone, Copy)]
enum AttentionWidth {
    Local,
    Global,
}

impl AttentionWidth {
    fn width(self, config: &Gemma4Config) -> usize {
        let kind = match self {
            Self::Local => AttentionKind::Sliding,
            Self::Global => AttentionKind::Full,
        };
        config
            .layers()
            .iter()
            .find(|layer| layer.kind == kind)
            .map_or(0, |layer| layer.head_dim)
    }
}

impl DeviceCache {
    pub(super) fn allocate(
        capacity: usize,
        columns: usize,
        stream: &Arc<CudaStream>,
        pool: &Arc<BytePool>,
    ) -> Result<Self, Gemma4Error> {
        Ok(Self {
            keys: matrix(capacity, columns, stream, pool, DEVICE_ONLY_STAGING)?,
            values: matrix(capacity, columns, stream, pool, DEVICE_ONLY_STAGING)?,
        })
    }
}

fn matrix(
    rows: usize,
    columns: usize,
    stream: &Arc<CudaStream>,
    pool: &Arc<BytePool>,
    staging: usize,
) -> Result<CudaBf16Matrix, Gemma4Error> {
    CudaBf16Matrix::allocate(rows, columns, Arc::clone(stream), pool, staging)
        .map_err(|e| Gemma4Error::new("CUDA decoder", e.to_string()))
}
