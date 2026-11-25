//! Model-wide single-command-buffer Metal engine for Gemma 4 with Paged KV & Prefix Caching.
//!
//! Keeps weights, paged per-owner KV pools (`KV_PAGE_TOKENS = 16`), `RoPE` inverse
//! frequencies, and batch scratch resident on the Apple GPU. Encodes an entire
//! packed multi-sequence forward pass (mixed prefill + decode) into one
//! `MTLCommandBuffer` with zero per-step buffer allocations, zero CPU KV copies,
//! and automatic zero-copy prefix reuse across turns and concurrent requests.

use super::activation::GeluTable;
use super::kv_pool::{KV_PAGE_TOKENS, PagedKvPool, PrefixCacheStats};
use super::{bf16_f32, f32_value, metal_error, usize_to_f32};
use crate::{
    AttentionKind, Gemma4Config, Gemma4Error, Gemma4Source, Gemma4SourceEncoding, LayerParameter,
    ParameterId, ParameterPayload, inverse_frequencies,
};
use engine_metal::{
    METAL_QUANT_GROUP_SIZE, MetalBf16Buffer, MetalBf16Executor, MetalDeviceBf16, MetalDeviceF32,
    MetalDeviceU32, MetalFastSession, MetalQuantKind, MetalQuantWeight,
};
use std::collections::BTreeMap;

/// Weight quantization policy for `Gemma4MetalModelWide`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Gemma4MetalQuant {
    /// Full unconverted BF16 weights (exact BF16 reference precision).
    #[default]
    Bf16,
    /// 8-bit affine group-64 quantization for all compatible 2D weights and embeddings.
    Q8,
    /// 4-bit affine group-64 quantization for all compatible 2D weights and embeddings.
    Q4,
}

impl Gemma4MetalQuant {
    /// Resolve an optional explicit quantization override against a validated source's
    /// native encoding.
    ///
    /// - When `requested` is `None`, matches the package's native precision (`Bf16` for
    ///   BF16 packages, `Q8` for pre-quantized 8-bit packages, `Q4` for pre-quantized 4-bit).
    /// - When `requested` is `Some(q)` on a BF16 package, allows `Bf16` or online `Q8`/`Q4`.
    /// - When `requested` is `Some(q)` on a pre-quantized package, accepts only the matching
    ///   native precision and rejects mismatches explicitly rather than re-quantizing.
    ///
    /// # Errors
    /// Returns [`Gemma4Error`] when `requested` conflicts with a pre-quantized source.
    pub fn resolve_for_source(
        requested: Option<Self>,
        source: Gemma4SourceEncoding,
    ) -> Result<Self, Gemma4Error> {
        let native = match source {
            Gemma4SourceEncoding::Bf16 => Self::Bf16,
            Gemma4SourceEncoding::AffineQ8G64 => Self::Q8,
            Gemma4SourceEncoding::AffineQ4G64 => Self::Q4,
        };
        match (source, requested) {
            (_, None) => Ok(native),
            (Gemma4SourceEncoding::Bf16, Some(q)) => Ok(q),
            (_, Some(q)) if q == native => Ok(q),
            (_, Some(q)) => Err(Gemma4Error::new(
                "quantize",
                format!(
                    "pre-quantized {source:?} package cannot be loaded with --quantize {q:?}; omit --quantize or match the package's native precision ({native:?})"
                ),
            )),
        }
    }
}

struct ModelWideWeights {
    bf16: BTreeMap<ParameterId, MetalBf16Buffer>,
    quant: BTreeMap<ParameterId, MetalQuantWeight>,
    bytes: u64,
    config: Gemma4Config,
}

impl ModelWideWeights {
    fn load(
        source: &Gemma4Source,
        executor: &MetalBf16Executor,
        quant_policy: Gemma4MetalQuant,
    ) -> Result<Self, Gemma4Error> {
        Gemma4MetalQuant::resolve_for_source(Some(quant_policy), source.encoding())?;
        let quant_kind = match quant_policy {
            Gemma4MetalQuant::Bf16 => None,
            Gemma4MetalQuant::Q8 => Some(MetalQuantKind::Q8),
            Gemma4MetalQuant::Q4 => Some(MetalQuantKind::Q4),
        };
        let mut bf16 = BTreeMap::new();
        let mut quant = BTreeMap::new();
        let mut bytes = 0_u64;

        for (&id, parameter) in source.parameters() {
            let spec = parameter.spec();
            match parameter.payload() {
                ParameterPayload::AffineGroup64 {
                    bits,
                    packed,
                    scales,
                    biases,
                } => {
                    let kind = match bits {
                        4 => MetalQuantKind::Q4,
                        8 => MetalQuantKind::Q8,
                        other => {
                            return Err(Gemma4Error::new(
                                &spec.name,
                                format!("unsupported pre-quantized bit width {other}"),
                            ));
                        }
                    };
                    let qw = executor
                        .read_prequantized(
                            spec.shape[0],
                            spec.shape[1],
                            kind,
                            &mut packed.clone(),
                            &mut scales.clone(),
                            &mut biases.clone(),
                        )
                        .map_err(|error| Gemma4Error::new(&spec.name, error.to_string()))?;
                    let quant_bytes = u64::try_from(qw.bytes()).map_err(|_| {
                        Gemma4Error::new("Metal weight", "quantized weight bytes exceed u64")
                    })?;
                    bytes = bytes
                        .checked_add(quant_bytes)
                        .ok_or_else(|| Gemma4Error::new("Metal weight", "total bytes overflow"))?;
                    quant.insert(id, qw);
                }
                ParameterPayload::Bf16(reader) => {
                    let online_quantize = source.encoding() == Gemma4SourceEncoding::Bf16
                        && quant_kind.is_some()
                        && spec.shape.len() == 2
                        && spec.shape[1].is_multiple_of(METAL_QUANT_GROUP_SIZE);
                    if let (Some(kind), true) = (quant_kind, online_quantize) {
                        let qw = executor
                            .read_quantized_bf16(
                                spec.shape[0],
                                spec.shape[1],
                                kind,
                                &mut reader.clone(),
                            )
                            .map_err(|error| Gemma4Error::new(&spec.name, error.to_string()))?;
                        let quant_bytes = u64::try_from(qw.bytes()).map_err(|_| {
                            Gemma4Error::new("Metal weight", "quantized weight bytes exceed u64")
                        })?;
                        bytes = bytes.checked_add(quant_bytes).ok_or_else(|| {
                            Gemma4Error::new("Metal weight", "total bytes overflow")
                        })?;
                        quant.insert(id, qw);
                    } else {
                        let length = spec.byte_len()?;
                        let buffer = executor
                            .read_bf16(length / 2, &mut reader.clone())
                            .map_err(|error| Gemma4Error::new(&spec.name, error.to_string()))?;
                        let entry_bytes = u64::try_from(length).map_err(|_| {
                            Gemma4Error::new("Metal weight", "parameter bytes exceed u64")
                        })?;
                        bytes = bytes.checked_add(entry_bytes).ok_or_else(|| {
                            Gemma4Error::new("Metal weight", "total bytes overflow")
                        })?;
                        bf16.insert(id, buffer);
                    }
                }
            }
        }

        if source.encoding() != Gemma4SourceEncoding::Bf16 || quant_policy == Gemma4MetalQuant::Bf16
        {
            let expected = u64::try_from(source.encoded_bytes())
                .map_err(|_| Gemma4Error::new("Metal weight", "source bytes exceed u64"))?;
            if bytes != expected {
                return Err(Gemma4Error::new("Metal weight", "inventory bytes disagree"));
            }
        }

        Ok(Self {
            bf16,
            quant,
            bytes,
            config: source.config().clone(),
        })
    }

    const fn config(&self) -> &Gemma4Config {
        &self.config
    }

    const fn bytes(&self) -> u64 {
        self.bytes
    }

    fn bf16_param(&self, id: ParameterId) -> Result<&MetalBf16Buffer, Gemma4Error> {
        self.bf16.get(&id).ok_or_else(|| {
            Gemma4Error::new("Metal modelwide", format!("missing BF16 parameter {id:?}"))
        })
    }

    fn layer_bf16(
        &self,
        layer: usize,
        parameter: LayerParameter,
    ) -> Result<&MetalBf16Buffer, Gemma4Error> {
        self.bf16_param(ParameterId::Layer { layer, parameter })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "dispatches BF16 or quantized linear projection over explicit session bindings"
    )]
    fn encode_linear(
        &self,
        session: &mut MetalFastSession<'_>,
        input: &MetalDeviceBf16,
        id: ParameterId,
        output: &MetalDeviceBf16,
        rows: usize,
        columns: usize,
        inner: usize,
    ) -> Result<(), Gemma4Error> {
        if let Some(qw) = self.quant.get(&id) {
            session
                .linear_quant(input, qw, output, rows, columns, inner)
                .map_err(|e| metal_error(&e))
        } else {
            let w = self.bf16_param(id)?;
            session
                .linear(input, w, output, rows, columns, inner)
                .map_err(|e| metal_error(&e))
        }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "dispatches BF16 or quantized dual equal-shape linear projection (K+V) with M=1 fusion"
    )]
    fn encode_dual_linear(
        &self,
        session: &mut MetalFastSession<'_>,
        input: &MetalDeviceBf16,
        id_a: ParameterId,
        id_b: ParameterId,
        out_a: &MetalDeviceBf16,
        out_b: &MetalDeviceBf16,
        rows: usize,
        columns: usize,
        inner: usize,
    ) -> Result<(), Gemma4Error> {
        if rows == 1 {
            if let (Some(qa), Some(qb)) = (self.quant.get(&id_a), self.quant.get(&id_b))
                && qa.kind() == qb.kind()
            {
                return session
                    .dual_linear_quant_m1(input, qa, qb, out_a, out_b, columns, inner)
                    .map_err(|e| metal_error(&e));
            }
            if let (Some(wa), Some(wb)) = (self.bf16.get(&id_a), self.bf16.get(&id_b)) {
                return session
                    .dual_linear_m1(input, wa, wb, out_a, out_b, columns, inner)
                    .map_err(|e| metal_error(&e));
            }
        }
        self.encode_linear(session, input, id_a, out_a, rows, columns, inner)?;
        self.encode_linear(session, input, id_b, out_b, rows, columns, inner)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "dispatches fused M=1 Gate+Up+GeGLU or unfused batched path"
    )]
    fn encode_gate_up_geglu(
        &self,
        session: &mut MetalFastSession<'_>,
        input: &MetalDeviceBf16,
        gate_id: ParameterId,
        up_id: ParameterId,
        gelu_table: &MetalBf16Buffer,
        gate_out: &MetalDeviceBf16,
        up_scratch: &MetalDeviceBf16,
        error_flag: &MetalDeviceU32,
        rows: usize,
        columns: usize,
        inner: usize,
    ) -> Result<(), Gemma4Error> {
        if rows == 1 {
            if let (Some(qg), Some(qu)) = (self.quant.get(&gate_id), self.quant.get(&up_id))
                && qg.kind() == qu.kind()
            {
                return session
                    .gate_up_geglu_quant_m1(
                        input, qg, qu, gelu_table, gate_out, error_flag, columns, inner,
                    )
                    .map_err(|e| metal_error(&e));
            }
            if let (Some(wg), Some(wu)) = (self.bf16.get(&gate_id), self.bf16.get(&up_id)) {
                return session
                    .gate_up_geglu_m1(
                        input, wg, wu, gelu_table, gate_out, error_flag, columns, inner,
                    )
                    .map_err(|e| metal_error(&e));
            }
        }
        self.encode_linear(session, input, gate_id, gate_out, rows, columns, inner)?;
        self.encode_linear(session, input, up_id, up_scratch, rows, columns, inner)?;
        session
            .geglu_table(
                gate_out,
                up_scratch,
                0,
                gelu_table,
                gate_out,
                error_flag,
                rows * columns,
            )
            .map_err(|e| metal_error(&e))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "dispatches fused M=1 linear+GeGLU (PleGate) or unfused batched path"
    )]
    fn encode_linear_geglu(
        &self,
        session: &mut MetalFastSession<'_>,
        input: &MetalDeviceBf16,
        gate_id: ParameterId,
        up_in: &MetalDeviceBf16,
        up_offset_elems: usize,
        gelu_table: &MetalBf16Buffer,
        output: &MetalDeviceBf16,
        error_flag: &MetalDeviceU32,
        rows: usize,
        columns: usize,
        inner: usize,
    ) -> Result<(), Gemma4Error> {
        if rows == 1 {
            if let Some(qg) = self.quant.get(&gate_id) {
                return session
                    .linear_geglu_quant_m1(
                        input,
                        qg,
                        up_in,
                        up_offset_elems,
                        gelu_table,
                        output,
                        error_flag,
                        columns,
                        inner,
                    )
                    .map_err(|e| metal_error(&e));
            }
            if let Some(wg) = self.bf16.get(&gate_id) {
                return session
                    .linear_geglu_m1(
                        input,
                        wg,
                        up_in,
                        up_offset_elems,
                        gelu_table,
                        output,
                        error_flag,
                        columns,
                        inner,
                    )
                    .map_err(|e| metal_error(&e));
            }
        }
        self.encode_linear(session, input, gate_id, output, rows, columns, inner)?;
        session
            .geglu_table(
                output,
                up_in,
                up_offset_elems,
                gelu_table,
                output,
                error_flag,
                rows * columns,
            )
            .map_err(|e| metal_error(&e))
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "dispatches BF16 or quantized embedding gather over explicit session bindings"
    )]
    fn encode_embed(
        &self,
        session: &mut MetalFastSession<'_>,
        token_ids: &MetalDeviceU32,
        id: ParameterId,
        output: &MetalDeviceBf16,
        rows: usize,
        width: usize,
        scale: f32,
    ) -> Result<(), Gemma4Error> {
        if let Some(qw) = self.quant.get(&id) {
            session
                .embed_gather_quant(token_ids, qw, output, rows, width, scale)
                .map_err(|e| metal_error(&e))
        } else {
            let tbl = self.bf16_param(id)?;
            session
                .embed_gather(token_ids, tbl, output, rows, width, scale)
                .map_err(|e| metal_error(&e))
        }
    }
}

/// Default serving bounds for model-wide Metal execution.
///
/// With paged KV (`KV_PAGE_TOKENS = 16`), `2048` physical blocks hold `32,768`
/// pooled tokens (~1.84 GiB for Gemma 4 E4B across all 24 KV-owner layers) while
/// supporting up to `16,384`-token single-sequence contexts, `16` concurrent
/// sequences, and zero-copy shared prefix blocks.
pub const DEFAULT_METAL_MAX_SEQUENCES: usize = 16;
pub const DEFAULT_METAL_MAX_POSITIONS: usize = 16_384;
pub const DEFAULT_METAL_MAX_BATCH_TOKENS: usize = 512;
pub const DEFAULT_METAL_KV_BLOCKS: usize = 2048;

/// Physical resource bounds for one `Gemma4MetalModelWide` instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4MetalModelWideConfig {
    pub max_sequences: usize,
    pub max_positions: usize,
    pub max_batch_tokens: usize,
    /// Number of physical `KV_PAGE_TOKENS`-token KV blocks in the shared pool.
    /// `0` auto-sizes from `max_sequences` and `max_positions` (capped at `DEFAULT_METAL_KV_BLOCKS`).
    pub num_kv_blocks: usize,
    /// Weight quantization policy (`Bf16` default, or `Q8`/`Q4` affine group-64).
    pub quant: Gemma4MetalQuant,
}

impl Default for Gemma4MetalModelWideConfig {
    fn default() -> Self {
        Self {
            max_sequences: DEFAULT_METAL_MAX_SEQUENCES,
            max_positions: DEFAULT_METAL_MAX_POSITIONS,
            max_batch_tokens: DEFAULT_METAL_MAX_BATCH_TOKENS,
            num_kv_blocks: DEFAULT_METAL_KV_BLOCKS,
            quant: Gemma4MetalQuant::Bf16,
        }
    }
}

struct SlotState {
    position: usize,
    blocks: Vec<usize>,
    tokens: Vec<u32>,
    cached_pages: usize,
    last_node: u64,
}

struct OwnerKv {
    keys: MetalDeviceBf16,
    values: MetalDeviceBf16,
}

struct BatchScratch {
    hidden: MetalDeviceBf16,
    normed: MetalDeviceBf16,
    proj_out: MetalDeviceBf16,
    ple_id: MetalDeviceBf16,
    ple_proj: MetalDeviceBf16,
    per_layer_ple: MetalDeviceBf16,
    q_buf: MetalDeviceBf16,
    k_buf: MetalDeviceBf16,
    v_buf: MetalDeviceBf16,
    attn_out: MetalDeviceBf16,
    ffn_gate: MetalDeviceBf16,
    ffn_up: MetalDeviceBf16,
    ple_gate: MetalDeviceBf16,
    tail_hidden: MetalDeviceBf16,
    tail_normed: MetalDeviceBf16,
    tail_ple: MetalDeviceBf16,
    final_normed: MetalDeviceBf16,
    logits: MetalDeviceBf16,
    token_ids: MetalDeviceU32,
    row_slot: MetalDeviceU32,
    row_pos: MetalDeviceU32,
    out_row_slot: MetalDeviceU32,
    out_row_pos: MetalDeviceU32,
    block_table: MetalDeviceU32,
    output_rows: MetalDeviceU32,
    argmax_out: MetalDeviceU32,
    error_flag: MetalDeviceU32,
}

/// One packed sequence slice in a batched model-wide forward pass.
#[derive(Clone, Copy, Debug)]
pub struct PackedStepItem<'a> {
    pub slot: usize,
    pub tokens: &'a [u32],
    pub need_logits: bool,
}

/// Zero-copy view over the completed outputs of a packed forward pass.
pub struct PackedStepOutput<'a> {
    pub argmax: &'a [u32],
    pub logits: &'a [u16],
    pub vocab_size: usize,
}

impl PackedStepOutput<'_> {
    /// Borrow the soft-capped BF16 logits row for the `index`-th output-producing item.
    #[must_use]
    pub fn logits_row(&self, index: usize) -> &[u16] {
        let start = index * self.vocab_size;
        &self.logits[start..start + self.vocab_size]
    }
}

/// Model-wide Gemma 4 Metal engine with paged KV pool, radix prefix caching, and continuous batching.
pub struct Gemma4MetalModelWide {
    executor: MetalBf16Executor,
    weights: ModelWideWeights,
    gelu_table: MetalBf16Buffer,
    inv_freqs: Vec<MetalDeviceF32>,
    kv_caches: Vec<Option<OwnerKv>>,
    last_kv_owner: usize,
    kv_pool: PagedKvPool,
    max_pages_per_seq: usize,
    scratch: BatchScratch,
    slots: Vec<Option<SlotState>>,
    limits: Gemma4MetalModelWideConfig,
    // Reusable host staging vectors to avoid per-step heap allocations.
    host_tokens: Vec<u32>,
    host_slots: Vec<u32>,
    host_pos: Vec<u32>,
    host_out_rows: Vec<u32>,
    host_out_slots: Vec<u32>,
    host_out_pos: Vec<u32>,
    host_page_row: Vec<u32>,
    faulted: bool,
}

impl Gemma4MetalModelWide {
    /// Load weights and pre-allocate paged KV pool and batch scratch on Metal.
    /// # Errors
    /// Rejects invalid limits, device/compilation failure, weight load failure, or OOM.
    #[expect(
        clippy::too_many_lines,
        reason = "pre-allocates weights, paged KV pools, RoPE frequencies, and batch scratch"
    )]
    pub fn load(
        source: &Gemma4Source,
        mut limits: Gemma4MetalModelWideConfig,
    ) -> Result<Self, Gemma4Error> {
        let config = source.config();
        if limits.max_sequences == 0 || limits.max_positions == 0 || limits.max_batch_tokens == 0 {
            return Err(Gemma4Error::new(
                "Metal modelwide",
                "max_sequences, max_positions and max_batch_tokens must be positive",
            ));
        }
        limits.max_positions = limits.max_positions.min(config.max_position_embeddings());
        let max_pages_per_seq = limits.max_positions.div_ceil(KV_PAGE_TOKENS);
        if limits.num_kv_blocks == 0 {
            limits.num_kv_blocks = limits
                .max_sequences
                .saturating_mul(max_pages_per_seq)
                .clamp(max_pages_per_seq, DEFAULT_METAL_KV_BLOCKS);
        }
        for (layer, geometry) in config.layers().iter().enumerate() {
            if geometry.kv_owner > layer {
                return Err(Gemma4Error::new(
                    "Metal modelwide",
                    "key/value owner must not follow its consumer",
                ));
            }
            if geometry.head_dim > 512 {
                return Err(Gemma4Error::new(
                    "Metal modelwide",
                    "fast Metal attention requires head_dim <= 512",
                ));
            }
        }

        let executor = MetalBf16Executor::new().map_err(|e| metal_error(&e))?;
        let gelu = GeluTable::new()?;
        let gelu_table = executor
            .upload(gelu.values())
            .map_err(|e| metal_error(&e))?;
        let weights = ModelWideWeights::load(source, &executor, limits.quant)?;

        let mut inv_freqs = Vec::with_capacity(config.layers().len());
        let mut kv_caches = Vec::with_capacity(config.layers().len());
        let mut max_head_dim = 0_usize;
        let mut last_kv_owner = 0_usize;

        for (layer, geometry) in config.layers().iter().enumerate() {
            max_head_dim = max_head_dim.max(geometry.head_dim);
            let freqs = inverse_frequencies(config, layer)?;
            inv_freqs.push(executor.upload_f32(&freqs).map_err(|e| metal_error(&e))?);

            if geometry.kv_owner == layer {
                last_kv_owner = layer;
                let elems = limits
                    .num_kv_blocks
                    .checked_mul(config.num_key_value_heads())
                    .and_then(|n| n.checked_mul(KV_PAGE_TOKENS))
                    .and_then(|n| n.checked_mul(geometry.head_dim))
                    .ok_or_else(|| Gemma4Error::new("Metal modelwide", "KV pool size overflow"))?;
                let keys = executor.alloc_bf16(elems).map_err(|e| metal_error(&e))?;
                let values = executor.alloc_bf16(elems).map_err(|e| metal_error(&e))?;
                kv_caches.push(Some(OwnerKv { keys, values }));
            } else {
                kv_caches.push(None);
            }
        }

        let max_m = limits.max_batch_tokens;
        let max_out = limits.max_sequences;
        let hidden_size = config.hidden_size();
        let intermediate = config.intermediate_size();
        let packed_ple = config.packed_ple_dim();
        let max_q_cols = config.num_attention_heads() * max_head_dim;
        let max_kv_cols = config.num_key_value_heads() * max_head_dim;

        let alloc = |n: usize| executor.alloc_bf16(n).map_err(|e| metal_error(&e));
        let alloc_u32 = |n: usize| executor.alloc_u32(n).map_err(|e| metal_error(&e));

        let scratch = BatchScratch {
            hidden: alloc(max_m * hidden_size)?,
            normed: alloc(max_m * hidden_size)?,
            proj_out: alloc(max_m * hidden_size)?,
            ple_id: alloc(max_m * packed_ple)?,
            ple_proj: alloc(max_m * packed_ple)?,
            per_layer_ple: alloc(max_m * packed_ple)?,
            q_buf: alloc(max_m * max_q_cols)?,
            k_buf: alloc(max_m * max_kv_cols)?,
            v_buf: alloc(max_m * max_kv_cols)?,
            attn_out: alloc(max_m * max_q_cols)?,
            ffn_gate: alloc(max_m * intermediate)?,
            ffn_up: alloc(max_m * intermediate)?,
            ple_gate: alloc(max_m * config.ple_dim())?,
            tail_hidden: alloc(max_out * hidden_size)?,
            tail_normed: alloc(max_out * hidden_size)?,
            tail_ple: alloc(max_out * config.ple_dim())?,
            final_normed: alloc(max_out * hidden_size)?,
            logits: alloc(max_out * config.vocab_size())?,
            token_ids: alloc_u32(max_m)?,
            row_slot: alloc_u32(max_m)?,
            row_pos: alloc_u32(max_m)?,
            out_row_slot: alloc_u32(max_out)?,
            out_row_pos: alloc_u32(max_out)?,
            block_table: alloc_u32(limits.max_sequences * max_pages_per_seq)?,
            output_rows: alloc_u32(max_out)?,
            argmax_out: alloc_u32(max_out)?,
            error_flag: alloc_u32(1)?,
        };

        let kv_pool = PagedKvPool::new(limits.num_kv_blocks)
            .map_err(|error| Gemma4Error::new("KV pool", error.to_string()))?;
        let mut slots = Vec::with_capacity(limits.max_sequences);
        slots.resize_with(limits.max_sequences, || None);

        Ok(Self {
            executor,
            weights,
            gelu_table,
            inv_freqs,
            kv_caches,
            last_kv_owner,
            kv_pool,
            max_pages_per_seq,
            scratch,
            slots,
            limits,
            host_tokens: Vec::with_capacity(max_m),
            host_slots: Vec::with_capacity(max_m),
            host_pos: Vec::with_capacity(max_m),
            host_out_rows: Vec::with_capacity(max_out),
            host_out_slots: Vec::with_capacity(max_out),
            host_out_pos: Vec::with_capacity(max_out),
            host_page_row: Vec::with_capacity(max_pages_per_seq),
            faulted: false,
        })
    }

    #[must_use]
    pub const fn config(&self) -> &Gemma4Config {
        self.weights.config()
    }

    #[must_use]
    pub const fn limits(&self) -> Gemma4MetalModelWideConfig {
        self.limits
    }

    #[must_use]
    pub const fn weights_bytes(&self) -> u64 {
        self.weights.bytes()
    }

    #[must_use]
    pub fn prefix_cache_stats(&self) -> PrefixCacheStats {
        self.kv_pool.stats()
    }

    /// Reserve a physical slot starting at position 0 (no prefix match).
    /// # Errors
    /// Rejects faulted model or exhausted slot capacity.
    pub fn allocate_slot(&mut self) -> Result<usize, Gemma4Error> {
        self.check_healthy()?;
        let slot = self
            .slots
            .iter()
            .position(Option::is_none)
            .ok_or_else(|| Gemma4Error::new("Metal modelwide", "slot capacity exhausted"))?;
        self.slots[slot] = Some(SlotState {
            position: 0,
            blocks: Vec::new(),
            tokens: Vec::new(),
            cached_pages: 0,
            last_node: 0,
        });
        Ok(slot)
    }

    /// Reserve a physical slot and immediately pin any exact 16-token-aligned
    /// cached prefix matching `prompt` (strictly shorter than `prompt.len()`).
    ///
    /// Returns `(slot, matched_prefix_tokens)`.
    /// # Errors
    /// Rejects faulted model or exhausted slot capacity.
    pub fn allocate_slot_with_prefix(
        &mut self,
        prompt: &[u32],
    ) -> Result<(usize, usize), Gemma4Error> {
        self.check_healthy()?;
        let slot = self
            .slots
            .iter()
            .position(Option::is_none)
            .ok_or_else(|| Gemma4Error::new("Metal modelwide", "slot capacity exhausted"))?;
        let (blocks, last_node) = self.kv_pool.match_and_pin_prefix(prompt);
        let cached_pages = blocks.len();
        let matched = cached_pages * KV_PAGE_TOKENS;
        self.slots[slot] = Some(SlotState {
            position: matched,
            blocks,
            tokens: prompt[..matched].to_vec(),
            cached_pages,
            last_node,
        });
        self.sync_slot_block_table(slot)?;
        Ok((slot, matched))
    }

    /// Release a physical slot, returning private tail blocks to the free list
    /// while leaving published prefix blocks cached for reuse.
    /// # Errors
    /// Rejects release if the execution owner is faulted or `slot` is out of range.
    pub fn free_slot(&mut self, slot: usize) -> Result<(), Gemma4Error> {
        self.check_healthy()?;
        let entry = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| Gemma4Error::new("Metal modelwide", "slot out of range"))?;
        if let Some(state) = entry.take() {
            self.kv_pool.release_slot_blocks(state.blocks);
        }
        Ok(())
    }

    /// Number of currently occupied sequence slots.
    #[must_use]
    pub fn active_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Look up the committed token position of a physical slot.
    #[must_use]
    pub fn slot_position(&self, slot: usize) -> Option<usize> {
        self.slots
            .get(slot)
            .and_then(Option::as_ref)
            .map(|s| s.position)
    }

    /// Number of physical KV blocks currently held by `slot` (0 if unoccupied).
    #[must_use]
    pub fn slot_allocated_blocks(&self, slot: usize) -> usize {
        self.slots
            .get(slot)
            .and_then(Option::as_ref)
            .map_or(0, |s| s.blocks.len())
    }

    /// Total physical KV pages in the shared pool.
    #[must_use]
    pub const fn kv_pool_total_blocks(&self) -> usize {
        self.kv_pool.total_blocks()
    }

    /// Distinct physical KV pages currently pinned by at least one active sequence.
    #[must_use]
    pub const fn kv_pool_pinned_blocks(&self) -> usize {
        self.kv_pool.pinned_blocks()
    }

    /// Non-mutating check of cached prefix pages matching `prompt`, returning
    /// `(matched_pages, newly_pinned_blocks)`.
    #[must_use]
    pub fn peek_prefix_pages(&self, prompt: &[u32]) -> (usize, usize) {
        self.kv_pool.peek_prefix_match(prompt)
    }

    /// Reset an occupied slot to position 0 (preserving cached prefix blocks in the pool).
    /// # Errors
    /// Rejects an unoccupied or out-of-range slot or faulted model.
    pub fn reset_slot(&mut self, slot: usize) -> Result<(), Gemma4Error> {
        self.check_healthy()?;
        let state = self
            .slots
            .get_mut(slot)
            .and_then(Option::as_mut)
            .ok_or_else(|| Gemma4Error::new("Metal modelwide", "slot is not occupied"))?;
        let old_blocks = std::mem::take(&mut state.blocks);
        self.kv_pool.release_slot_blocks(old_blocks);
        state.position = 0;
        state.tokens.clear();
        state.cached_pages = 0;
        state.last_node = 0;
        Ok(())
    }

    /// Nonblocking physical failure settlement; never reopens a faulted owner.
    /// # Errors
    /// Reports unresolved GPU work.
    pub fn retire_failure(&self) -> Result<(), Gemma4Error> {
        self.executor.retire_failure().map_err(|e| metal_error(&e))
    }

    fn check_healthy(&self) -> Result<(), Gemma4Error> {
        if self.faulted {
            return Err(Gemma4Error::new(
                "Metal modelwide",
                "execution owner has faulted",
            ));
        }
        self.executor.check_healthy().map_err(|e| metal_error(&e))
    }

    fn sync_slot_block_table(&mut self, slot: usize) -> Result<(), Gemma4Error> {
        let Some(state) = self.slots[slot].as_ref() else {
            return Ok(());
        };
        self.host_page_row.clear();
        for &phys in &state.blocks {
            let phys_u32 = u32::try_from(phys)
                .map_err(|_| Gemma4Error::new("Metal modelwide", "block index exceeds u32"))?;
            self.host_page_row.push(phys_u32);
        }
        self.scratch
            .block_table
            .write_at(slot * self.max_pages_per_seq, &self.host_page_row)
            .map_err(|e| metal_error(&e))
    }

    /// Execute one packed batch of sequences (mixed prefill + decode) in a SINGLE Metal command buffer.
    ///
    /// Allocates any needed paged KV blocks before GPU launch and publishes newly
    /// completed 16-token blocks into the radix prefix cache on clean completion.
    /// # Errors
    /// Rejects invalid slots/tokens/overflows, KV block exhaustion, non-finite activations/logits, or Metal failure.
    /// # Panics
    /// Panics if an internal validated-slot invariant is violated.
    #[expect(
        clippy::too_many_lines,
        reason = "atomic whole-batch validation, paged block allocation, GPU submission and cache publish"
    )]
    pub fn step_packed(
        &mut self,
        items: &[PackedStepItem<'_>],
    ) -> Result<PackedStepOutput<'_>, Gemma4Error> {
        self.check_healthy()?;
        if items.is_empty() {
            return Err(Gemma4Error::new("Metal modelwide", "empty packed batch"));
        }

        let config = self.weights.config();
        self.host_tokens.clear();
        self.host_slots.clear();
        self.host_pos.clear();
        self.host_out_rows.clear();
        self.host_out_slots.clear();
        self.host_out_pos.clear();

        // 1. Whole-batch validation before allocating blocks or mutating GPU buffers.
        for (idx, item) in items.iter().enumerate() {
            if item.tokens.is_empty() {
                return Err(Gemma4Error::new("Metal modelwide", "empty item tokens"));
            }
            if items[..idx].iter().any(|prev| prev.slot == item.slot) {
                return Err(Gemma4Error::new(
                    "Metal modelwide",
                    "duplicate slot in packed batch",
                ));
            }
            let state = self
                .slots
                .get(item.slot)
                .and_then(Option::as_ref)
                .ok_or_else(|| Gemma4Error::new("Metal modelwide", "unoccupied KV slot"))?;
            let end = state
                .position
                .checked_add(item.tokens.len())
                .ok_or_else(|| Gemma4Error::new("Metal modelwide", "position overflow"))?;
            if end > self.limits.max_positions {
                return Err(Gemma4Error::new(
                    "Metal modelwide",
                    "sequence exceeds prepared KV capacity",
                ));
            }
            config.validate_text_tokens(item.tokens)?;

            let slot_u32 = u32::try_from(item.slot)
                .map_err(|_| Gemma4Error::new("Metal modelwide", "slot exceeds u32"))?;
            for (offset, &tok) in item.tokens.iter().enumerate() {
                let pos = u32::try_from(state.position + offset)
                    .map_err(|_| Gemma4Error::new("Metal modelwide", "position exceeds u32"))?;
                self.host_tokens.push(tok);
                self.host_slots.push(slot_u32);
                self.host_pos.push(pos);
            }
            if item.need_logits {
                let last_idx = self.host_tokens.len() - 1;
                let last_row = u32::try_from(last_idx)
                    .map_err(|_| Gemma4Error::new("Metal modelwide", "row index exceeds u32"))?;
                self.host_out_rows.push(last_row);
                self.host_out_slots.push(self.host_slots[last_idx]);
                self.host_out_pos.push(self.host_pos[last_idx]);
            }
        }

        let total_tokens = self.host_tokens.len();
        let out_rows = self.host_out_rows.len();
        if total_tokens > self.limits.max_batch_tokens {
            return Err(Gemma4Error::new(
                "Metal modelwide",
                format!(
                    "batch tokens {total_tokens} exceeds max_batch_tokens {}",
                    self.limits.max_batch_tokens
                ),
            ));
        }
        if out_rows > self.limits.max_sequences {
            return Err(Gemma4Error::new(
                "Metal modelwide",
                "output rows exceed max_sequences",
            ));
        }

        // 2. Allocate physical KV pages for every item in the batch (atomic rollback on OOM).
        let mut prev_block_lens: Vec<(usize, usize)> = Vec::with_capacity(items.len());
        for item in items {
            let state = self.slots[item.slot].as_mut().expect("validated slot");
            let old_len = state.blocks.len();
            let end_pos = state.position + item.tokens.len();
            if let Err(error) = self
                .kv_pool
                .ensure_blocks_for_range(&mut state.blocks, end_pos)
            {
                for (slot_idx, len) in prev_block_lens {
                    if let Some(st) = self.slots[slot_idx].as_mut() {
                        let extra: Vec<usize> = st.blocks.drain(len..).collect();
                        self.kv_pool.release_slot_blocks(extra);
                    }
                }
                return Err(Gemma4Error::new("KV pool", error.to_string()));
            }
            prev_block_lens.push((item.slot, old_len));
        }
        for &(slot_idx, _) in &prev_block_lens {
            self.sync_slot_block_table(slot_idx)?;
        }

        // 3. Stage batch metadata into shared device buffers while GPU is idle.
        self.scratch
            .token_ids
            .write(&self.host_tokens)
            .map_err(|e| metal_error(&e))?;
        self.scratch
            .row_slot
            .write(&self.host_slots)
            .map_err(|e| metal_error(&e))?;
        self.scratch
            .row_pos
            .write(&self.host_pos)
            .map_err(|e| metal_error(&e))?;
        if out_rows > 0 {
            self.scratch
                .output_rows
                .write(&self.host_out_rows)
                .map_err(|e| metal_error(&e))?;
            if out_rows < total_tokens {
                self.scratch
                    .out_row_slot
                    .write(&self.host_out_slots)
                    .map_err(|e| metal_error(&e))?;
                self.scratch
                    .out_row_pos
                    .write(&self.host_out_pos)
                    .map_err(|e| metal_error(&e))?;
            }
        }
        self.scratch
            .error_flag
            .write(&[0_u32])
            .map_err(|e| metal_error(&e))?;

        // 4. Encode and execute the complete forward pass in ONE command buffer.
        if let Err(error) = self.encode_and_submit(total_tokens, out_rows) {
            self.faulted = true;
            return Err(error);
        }

        let flags = self.scratch.error_flag.as_slice(1)[0];
        if flags != 0 {
            self.faulted = true;
            return Err(Gemma4Error::new(
                "Metal modelwide",
                if flags & 1 != 0 {
                    "non-finite activation encountered in GeGLU"
                } else {
                    "non-finite logit encountered before softcap"
                },
            ));
        }

        // 5. Commit positions and publish newly completed 16-token blocks into the radix cache.
        for item in items {
            let mut sync_needed = false;
            if let Some(state) = self.slots[item.slot].as_mut() {
                state.position += item.tokens.len();
                state.tokens.extend_from_slice(item.tokens);
                let prev_cached = state.cached_pages;
                self.kv_pool.publish_completed_blocks(
                    &mut state.blocks,
                    &state.tokens,
                    &mut state.cached_pages,
                    &mut state.last_node,
                );
                if state.cached_pages != prev_cached {
                    // Deduplication may have remapped a page to an existing cached block.
                    sync_needed = true;
                }
            }
            if sync_needed {
                self.sync_slot_block_table(item.slot)?;
            }
        }

        let vocab_size = self.weights.config().vocab_size();
        Ok(PackedStepOutput {
            argmax: self.scratch.argmax_out.as_slice(out_rows),
            logits: self.scratch.logits.as_slice(out_rows * vocab_size),
            vocab_size,
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "single-command-buffer 42-layer Gemma 4 forward encoding"
    )]
    fn encode_and_submit(&self, total_tokens: usize, out_rows: usize) -> Result<(), Gemma4Error> {
        let config = self.weights.config();
        let hidden_size = config.hidden_size();
        let intermediate = config.intermediate_size();
        let ple_dim = config.ple_dim();
        let packed_ple = config.packed_ple_dim();
        let num_layers = config.layers().len();
        let heads = config.num_attention_heads();
        let kv_heads = config.num_key_value_heads();
        let epsilon = config.rms_norm_eps();
        let max_pages = self.max_pages_per_seq;

        let ple_norm_w = self.weights.bf16_param(ParameterId::PleNorm)?;

        let embed_scale = f32_value(bf16_f32(usize_to_f32(hidden_size).sqrt()));
        let ple_scale = f32_value(bf16_f32(usize_to_f32(ple_dim).sqrt()));
        let inv_sqrt_hidden = 1.0 / usize_to_f32(hidden_size).sqrt();

        let mut session = self
            .executor
            .begin_fast_session()
            .map_err(|e| metal_error(&e))?;

        // 1. Embeddings + PLE projection/norm/combine/transpose
        self.weights.encode_embed(
            &mut session,
            &self.scratch.token_ids,
            ParameterId::TokenEmbedding,
            &self.scratch.hidden,
            total_tokens,
            hidden_size,
            embed_scale,
        )?;
        self.weights.encode_embed(
            &mut session,
            &self.scratch.token_ids,
            ParameterId::PleEmbedding,
            &self.scratch.ple_id,
            total_tokens,
            packed_ple,
            ple_scale,
        )?;
        self.weights.encode_linear(
            &mut session,
            &self.scratch.hidden,
            ParameterId::PleProjection,
            &self.scratch.ple_proj,
            total_tokens,
            packed_ple,
            hidden_size,
        )?;
        session
            .ple_combine_transpose(
                &self.scratch.ple_proj,
                &self.scratch.ple_id,
                ple_norm_w,
                &self.scratch.per_layer_ple,
                total_tokens,
                num_layers,
                ple_dim,
                inv_sqrt_hidden,
                epsilon,
            )
            .map_err(|e| metal_error(&e))?;

        // 2. All decoder layers in the SAME command encoder.
        //    Once the final KV-owner layer (`last_kv_owner`) has populated its
        //    KV cache, no subsequent layer writes to any KV cache and no future
        //    token ever reads non-output rows' hidden states. Intermediate
        //    prefill chunks (`out_rows == 0`) therefore terminate immediately,
        //    and batches with `0 < out_rows < total_tokens` prune directly to
        //    the `out_rows` logit-producing rows (eliminating ~45% of prefill
        //    FLOPs on Gemma 4 E4B with zero approximation).
        let mut pruned = false;
        let mut active_tokens = total_tokens;
        let mut active_hidden = &self.scratch.hidden;
        let mut active_normed = &self.scratch.normed;
        let mut active_slots = &self.scratch.row_slot;
        let mut active_pos = &self.scratch.row_pos;

        for (layer, geometry) in config.layers().iter().enumerate() {
            let head_dim = geometry.head_dim;
            let q_cols = heads * head_dim;
            let kv_cols = kv_heads * head_dim;

            session
                .rms_norm(
                    active_hidden,
                    Some(self.weights.layer_bf16(layer, LayerParameter::InputNorm)?),
                    active_normed,
                    active_tokens,
                    hidden_size,
                    epsilon,
                )
                .map_err(|e| metal_error(&e))?;

            if geometry.kv_owner == layer {
                let owner_kv = self.kv_caches[layer]
                    .as_ref()
                    .ok_or_else(|| Gemma4Error::new("Metal modelwide", "missing KV owner cache"))?;
                self.weights.encode_dual_linear(
                    &mut session,
                    active_normed,
                    ParameterId::Layer {
                        layer,
                        parameter: LayerParameter::Key,
                    },
                    ParameterId::Layer {
                        layer,
                        parameter: LayerParameter::Value,
                    },
                    &self.scratch.k_buf,
                    &self.scratch.v_buf,
                    active_tokens,
                    kv_cols,
                    hidden_size,
                )?;
                session
                    .kv_norm_rope_write(
                        &self.scratch.k_buf,
                        &self.scratch.v_buf,
                        self.weights.layer_bf16(layer, LayerParameter::KeyNorm)?,
                        &self.inv_freqs[layer],
                        active_slots,
                        active_pos,
                        &self.scratch.block_table,
                        &owner_kv.keys,
                        &owner_kv.values,
                        active_tokens,
                        kv_heads,
                        head_dim,
                        max_pages,
                        epsilon,
                    )
                    .map_err(|e| metal_error(&e))?;
            }

            if layer == self.last_kv_owner {
                if out_rows == 0 {
                    break;
                }
                if out_rows < total_tokens {
                    session
                        .gather_rows(
                            &self.scratch.hidden,
                            0,
                            &self.scratch.output_rows,
                            &self.scratch.tail_hidden,
                            total_tokens,
                            out_rows,
                            hidden_size,
                        )
                        .map_err(|e| metal_error(&e))?;
                    session
                        .gather_rows(
                            &self.scratch.normed,
                            0,
                            &self.scratch.output_rows,
                            &self.scratch.tail_normed,
                            total_tokens,
                            out_rows,
                            hidden_size,
                        )
                        .map_err(|e| metal_error(&e))?;
                    pruned = true;
                    active_tokens = out_rows;
                    active_hidden = &self.scratch.tail_hidden;
                    active_normed = &self.scratch.tail_normed;
                    active_slots = &self.scratch.out_row_slot;
                    active_pos = &self.scratch.out_row_pos;
                }
            }

            self.weights.encode_linear(
                &mut session,
                active_normed,
                ParameterId::Layer {
                    layer,
                    parameter: LayerParameter::Query,
                },
                &self.scratch.q_buf,
                active_tokens,
                q_cols,
                hidden_size,
            )?;
            session
                .q_norm_rope(
                    &self.scratch.q_buf,
                    self.weights.layer_bf16(layer, LayerParameter::QueryNorm)?,
                    &self.inv_freqs[layer],
                    active_pos,
                    active_tokens,
                    heads,
                    head_dim,
                    epsilon,
                )
                .map_err(|e| metal_error(&e))?;

            let donor_kv = self.kv_caches[geometry.kv_owner]
                .as_ref()
                .ok_or_else(|| Gemma4Error::new("Metal modelwide", "missing donor KV cache"))?;
            let window = match geometry.kind {
                AttentionKind::Sliding => Some(config.sliding_window()),
                AttentionKind::Full => None,
            };
            session
                .flash_attention(
                    &self.scratch.q_buf,
                    &donor_kv.keys,
                    &donor_kv.values,
                    &self.scratch.attn_out,
                    active_slots,
                    active_pos,
                    &self.scratch.block_table,
                    active_tokens,
                    heads,
                    kv_heads,
                    head_dim,
                    max_pages,
                    window,
                )
                .map_err(|e| metal_error(&e))?;

            self.weights.encode_linear(
                &mut session,
                &self.scratch.attn_out,
                ParameterId::Layer {
                    layer,
                    parameter: LayerParameter::AttentionOutput,
                },
                &self.scratch.proj_out,
                active_tokens,
                hidden_size,
                q_cols,
            )?;
            session
                .post_norm_add_pre_norm(
                    active_hidden,
                    &self.scratch.proj_out,
                    self.weights
                        .layer_bf16(layer, LayerParameter::PostAttentionNorm)?,
                    self.weights
                        .layer_bf16(layer, LayerParameter::PreFeedforwardNorm)?,
                    &self.scratch.proj_out,
                    active_tokens,
                    hidden_size,
                    epsilon,
                )
                .map_err(|e| metal_error(&e))?;

            self.weights.encode_gate_up_geglu(
                &mut session,
                &self.scratch.proj_out,
                ParameterId::Layer {
                    layer,
                    parameter: LayerParameter::Gate,
                },
                ParameterId::Layer {
                    layer,
                    parameter: LayerParameter::Up,
                },
                &self.gelu_table,
                &self.scratch.ffn_gate,
                &self.scratch.ffn_up,
                &self.scratch.error_flag,
                active_tokens,
                intermediate,
                hidden_size,
            )?;
            self.weights.encode_linear(
                &mut session,
                &self.scratch.ffn_gate,
                ParameterId::Layer {
                    layer,
                    parameter: LayerParameter::Down,
                },
                &self.scratch.proj_out,
                active_tokens,
                hidden_size,
                intermediate,
            )?;
            session
                .rms_norm_residual_add_scale(
                    active_hidden,
                    &self.scratch.proj_out,
                    self.weights
                        .layer_bf16(layer, LayerParameter::PostFeedforwardNorm)?,
                    None,
                    active_tokens,
                    hidden_size,
                    epsilon,
                )
                .map_err(|e| metal_error(&e))?;

            let (ple_up, ple_up_offset) = if pruned {
                session
                    .gather_rows(
                        &self.scratch.per_layer_ple,
                        layer * total_tokens * ple_dim,
                        &self.scratch.output_rows,
                        &self.scratch.tail_ple,
                        total_tokens,
                        out_rows,
                        ple_dim,
                    )
                    .map_err(|e| metal_error(&e))?;
                (&self.scratch.tail_ple, 0)
            } else {
                (&self.scratch.per_layer_ple, layer * total_tokens * ple_dim)
            };
            self.weights.encode_linear_geglu(
                &mut session,
                active_hidden,
                ParameterId::Layer {
                    layer,
                    parameter: LayerParameter::PleGate,
                },
                ple_up,
                ple_up_offset,
                &self.gelu_table,
                &self.scratch.ple_gate,
                &self.scratch.error_flag,
                active_tokens,
                ple_dim,
                hidden_size,
            )?;
            self.weights.encode_linear(
                &mut session,
                &self.scratch.ple_gate,
                ParameterId::Layer {
                    layer,
                    parameter: LayerParameter::PleProjection,
                },
                &self.scratch.proj_out,
                active_tokens,
                hidden_size,
                ple_dim,
            )?;
            session
                .rms_norm_residual_add_scale(
                    active_hidden,
                    &self.scratch.proj_out,
                    self.weights
                        .layer_bf16(layer, LayerParameter::PostPleNorm)?,
                    Some(
                        self.weights
                            .layer_bf16(layer, LayerParameter::LayerScalar)?,
                    ),
                    active_tokens,
                    hidden_size,
                    epsilon,
                )
                .map_err(|e| metal_error(&e))?;
        }

        // 3. FinalNorm + Tied LM Head + Softcap + Argmax ONLY for output-producing rows
        if out_rows > 0 {
            let final_norm = self.weights.bf16_param(ParameterId::FinalNorm)?;
            if pruned {
                session
                    .rms_norm(
                        active_hidden,
                        Some(final_norm),
                        &self.scratch.final_normed,
                        out_rows,
                        hidden_size,
                        epsilon,
                    )
                    .map_err(|e| metal_error(&e))?;
            } else {
                session
                    .gather_rows_rms(
                        &self.scratch.hidden,
                        &self.scratch.output_rows,
                        final_norm,
                        &self.scratch.final_normed,
                        total_tokens,
                        out_rows,
                        hidden_size,
                        epsilon,
                    )
                    .map_err(|e| metal_error(&e))?;
            }
            self.weights.encode_linear(
                &mut session,
                &self.scratch.final_normed,
                ParameterId::TokenEmbedding,
                &self.scratch.logits,
                out_rows,
                config.vocab_size(),
                hidden_size,
            )?;
            session
                .softcap_argmax(
                    &self.scratch.logits,
                    &self.scratch.argmax_out,
                    &self.scratch.error_flag,
                    out_rows,
                    config.vocab_size(),
                    config.final_logit_softcapping(),
                )
                .map_err(|e| metal_error(&e))?;
        }

        session
            .submit("gemma4 modelwide forward")
            .map_err(|e| metal_error(&e))
    }

    /// Direct single-sequence prefill convenience helper (automatically chunks by `max_batch_tokens`).
    /// # Errors
    /// Rejects invalid slot/tokens/overflow or Metal failure.
    pub fn prefill_slot(
        &mut self,
        slot: usize,
        tokens: &[u32],
        need_logits: bool,
    ) -> Result<Option<Vec<u16>>, Gemma4Error> {
        if tokens.is_empty() {
            return Err(Gemma4Error::new("Metal modelwide", "empty prefill tokens"));
        }
        let chunk_size = self.limits.max_batch_tokens;
        let total_chunks = tokens.len().div_ceil(chunk_size);
        let mut last_logits = None;
        for (idx, chunk) in tokens.chunks(chunk_size).enumerate() {
            let is_last = idx + 1 == total_chunks;
            let out = self.step_packed(&[PackedStepItem {
                slot,
                tokens: chunk,
                need_logits: need_logits && is_last,
            }])?;
            if need_logits && is_last {
                last_logits = Some(out.logits_row(0).to_vec());
            }
        }
        Ok(last_logits)
    }

    /// Direct single-sequence decode returning full BF16 logits (for reference verification).
    /// # Errors
    /// Rejects invalid slot/token or Metal failure.
    pub fn decode_slot(&mut self, slot: usize, token: u32) -> Result<Vec<u16>, Gemma4Error> {
        let out = self.step_packed(&[PackedStepItem {
            slot,
            tokens: &[token],
            need_logits: true,
        }])?;
        Ok(out.logits_row(0).to_vec())
    }

    /// Direct single-sequence decode returning only the GPU-computed greedy token ID.
    /// # Errors
    /// Rejects invalid slot/token or Metal failure.
    pub fn decode_slot_argmax(&mut self, slot: usize, token: u32) -> Result<u32, Gemma4Error> {
        let out = self.step_packed(&[PackedStepItem {
            slot,
            tokens: &[token],
            need_logits: true,
        }])?;
        Ok(out.argmax[0])
    }
}
