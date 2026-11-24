//! Model-wide single-command-buffer Metal engine for Gemma 4 with Paged KV & Prefix Caching.
//!
//! Keeps weights, paged per-owner KV pools (`KV_PAGE_TOKENS = 16`), `RoPE` inverse
//! frequencies, and batch scratch resident on the Apple GPU. Encodes an entire
//! packed multi-sequence forward pass (mixed prefill + decode) into one
//! `MTLCommandBuffer` with zero per-step buffer allocations, zero CPU KV copies,
//! and automatic zero-copy prefix reuse across turns and concurrent requests.

use super::activation::GeluTable;
use super::kv_pool::{KV_PAGE_TOKENS, PagedKvPool, PrefixCacheStats};
use super::{Gemma4MetalWeights, bf16_f32, f32_value, metal_error, usize_to_f32};
use crate::{
    AttentionKind, Gemma4Config, Gemma4Error, Gemma4Source, LayerParameter, ParameterId,
    inverse_frequencies,
};
use engine_metal::{
    MetalBf16Buffer, MetalBf16Executor, MetalDeviceBf16, MetalDeviceF32, MetalDeviceU32,
};

/// Default serving bounds for model-wide Metal execution.
///
/// With paged KV (`KV_PAGE_TOKENS = 16`), `2048` physical blocks hold `32,768`
/// pooled tokens (~1.84 GiB for Gemma 4 E4B across all 24 KV-owner layers) while
/// supporting up to `16,384`-token single-sequence contexts, `16` concurrent
/// sequences, and zero-copy shared prefix blocks.
pub const DEFAULT_METAL_MAX_SEQUENCES: usize = 16;
pub const DEFAULT_METAL_MAX_POSITIONS: usize = 16_384;
pub const DEFAULT_METAL_MAX_BATCH_TOKENS: usize = 256;
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
}

impl Default for Gemma4MetalModelWideConfig {
    fn default() -> Self {
        Self {
            max_sequences: DEFAULT_METAL_MAX_SEQUENCES,
            max_positions: DEFAULT_METAL_MAX_POSITIONS,
            max_batch_tokens: DEFAULT_METAL_MAX_BATCH_TOKENS,
            num_kv_blocks: DEFAULT_METAL_KV_BLOCKS,
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
    final_normed: MetalDeviceBf16,
    logits: MetalDeviceBf16,
    token_ids: MetalDeviceU32,
    row_slot: MetalDeviceU32,
    row_pos: MetalDeviceU32,
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
    weights: Gemma4MetalWeights,
    gelu_table: MetalBf16Buffer,
    inv_freqs: Vec<MetalDeviceF32>,
    kv_caches: Vec<Option<OwnerKv>>,
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
        let weights = Gemma4MetalWeights::load(source, &executor)?;

        let mut inv_freqs = Vec::with_capacity(config.layers().len());
        let mut kv_caches = Vec::with_capacity(config.layers().len());
        let mut max_head_dim = 0_usize;

        for (layer, geometry) in config.layers().iter().enumerate() {
            max_head_dim = max_head_dim.max(geometry.head_dim);
            let freqs = inverse_frequencies(config, layer)?;
            inv_freqs.push(executor.upload_f32(&freqs).map_err(|e| metal_error(&e))?);

            if geometry.kv_owner == layer {
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
            final_normed: alloc(max_out * hidden_size)?,
            logits: alloc(max_out * config.vocab_size())?,
            token_ids: alloc_u32(max_m)?,
            row_slot: alloc_u32(max_m)?,
            row_pos: alloc_u32(max_m)?,
            block_table: alloc_u32(limits.max_sequences * max_pages_per_seq)?,
            output_rows: alloc_u32(max_out)?,
            argmax_out: alloc_u32(max_out)?,
            error_flag: alloc_u32(1)?,
        };

        let kv_pool = PagedKvPool::new(limits.num_kv_blocks)?;
        let mut slots = Vec::with_capacity(limits.max_sequences);
        slots.resize_with(limits.max_sequences, || None);

        Ok(Self {
            executor,
            weights,
            gelu_table,
            inv_freqs,
            kv_caches,
            kv_pool,
            max_pages_per_seq,
            scratch,
            slots,
            limits,
            host_tokens: Vec::with_capacity(max_m),
            host_slots: Vec::with_capacity(max_m),
            host_pos: Vec::with_capacity(max_m),
            host_out_rows: Vec::with_capacity(max_out),
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
                let last_row = u32::try_from(self.host_tokens.len() - 1)
                    .map_err(|_| Gemma4Error::new("Metal modelwide", "row index exceeds u32"))?;
                self.host_out_rows.push(last_row);
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
                return Err(error);
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

        let token_embd = self
            .weights
            .parameter(ParameterId::TokenEmbedding)
            .ok_or_else(|| Gemma4Error::new("Metal modelwide", "missing token embedding"))?;
        let ple_embd = self
            .weights
            .parameter(ParameterId::PleEmbedding)
            .ok_or_else(|| Gemma4Error::new("Metal modelwide", "missing PLE embedding"))?;
        let ple_proj_w = self
            .weights
            .parameter(ParameterId::PleProjection)
            .ok_or_else(|| Gemma4Error::new("Metal modelwide", "missing PLE projection"))?;
        let ple_norm_w = self
            .weights
            .parameter(ParameterId::PleNorm)
            .ok_or_else(|| Gemma4Error::new("Metal modelwide", "missing PLE norm"))?;

        let embed_scale = f32_value(bf16_f32(usize_to_f32(hidden_size).sqrt()));
        let ple_scale = f32_value(bf16_f32(usize_to_f32(ple_dim).sqrt()));
        let inv_sqrt_hidden = 1.0 / usize_to_f32(hidden_size).sqrt();

        let mut session = self
            .executor
            .begin_fast_session()
            .map_err(|e| metal_error(&e))?;

        // 1. Embeddings + PLE projection/norm/combine/transpose
        session
            .embed_gather(
                &self.scratch.token_ids,
                token_embd,
                &self.scratch.hidden,
                total_tokens,
                hidden_size,
                embed_scale,
            )
            .map_err(|e| metal_error(&e))?;
        session
            .embed_gather(
                &self.scratch.token_ids,
                ple_embd,
                &self.scratch.ple_id,
                total_tokens,
                packed_ple,
                ple_scale,
            )
            .map_err(|e| metal_error(&e))?;
        session
            .linear(
                &self.scratch.hidden,
                ple_proj_w,
                &self.scratch.ple_proj,
                total_tokens,
                packed_ple,
                hidden_size,
            )
            .map_err(|e| metal_error(&e))?;
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

        // 2. All decoder layers in the SAME command encoder
        for (layer, geometry) in config.layers().iter().enumerate() {
            let head_dim = geometry.head_dim;
            let q_cols = heads * head_dim;
            let kv_cols = kv_heads * head_dim;

            session
                .rms_norm(
                    &self.scratch.hidden,
                    Some(self.weights.layer(layer, LayerParameter::InputNorm)?),
                    &self.scratch.normed,
                    total_tokens,
                    hidden_size,
                    epsilon,
                )
                .map_err(|e| metal_error(&e))?;
            session
                .linear(
                    &self.scratch.normed,
                    self.weights.layer(layer, LayerParameter::Query)?,
                    &self.scratch.q_buf,
                    total_tokens,
                    q_cols,
                    hidden_size,
                )
                .map_err(|e| metal_error(&e))?;
            session
                .q_norm_rope(
                    &self.scratch.q_buf,
                    self.weights.layer(layer, LayerParameter::QueryNorm)?,
                    &self.inv_freqs[layer],
                    &self.scratch.row_pos,
                    total_tokens,
                    heads,
                    head_dim,
                    epsilon,
                )
                .map_err(|e| metal_error(&e))?;

            if geometry.kv_owner == layer {
                let owner_kv = self.kv_caches[layer]
                    .as_ref()
                    .ok_or_else(|| Gemma4Error::new("Metal modelwide", "missing KV owner cache"))?;
                session
                    .linear(
                        &self.scratch.normed,
                        self.weights.layer(layer, LayerParameter::Key)?,
                        &self.scratch.k_buf,
                        total_tokens,
                        kv_cols,
                        hidden_size,
                    )
                    .map_err(|e| metal_error(&e))?;
                session
                    .linear(
                        &self.scratch.normed,
                        self.weights.layer(layer, LayerParameter::Value)?,
                        &self.scratch.v_buf,
                        total_tokens,
                        kv_cols,
                        hidden_size,
                    )
                    .map_err(|e| metal_error(&e))?;
                session
                    .kv_norm_rope_write(
                        &self.scratch.k_buf,
                        &self.scratch.v_buf,
                        self.weights.layer(layer, LayerParameter::KeyNorm)?,
                        &self.inv_freqs[layer],
                        &self.scratch.row_slot,
                        &self.scratch.row_pos,
                        &self.scratch.block_table,
                        &owner_kv.keys,
                        &owner_kv.values,
                        total_tokens,
                        kv_heads,
                        head_dim,
                        max_pages,
                        epsilon,
                    )
                    .map_err(|e| metal_error(&e))?;
            }

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
                    &self.scratch.row_slot,
                    &self.scratch.row_pos,
                    &self.scratch.block_table,
                    total_tokens,
                    heads,
                    kv_heads,
                    head_dim,
                    max_pages,
                    window,
                )
                .map_err(|e| metal_error(&e))?;

            session
                .linear(
                    &self.scratch.attn_out,
                    self.weights.layer(layer, LayerParameter::AttentionOutput)?,
                    &self.scratch.proj_out,
                    total_tokens,
                    hidden_size,
                    q_cols,
                )
                .map_err(|e| metal_error(&e))?;
            session
                .rms_norm(
                    &self.scratch.proj_out,
                    Some(
                        self.weights
                            .layer(layer, LayerParameter::PostAttentionNorm)?,
                    ),
                    &self.scratch.normed,
                    total_tokens,
                    hidden_size,
                    epsilon,
                )
                .map_err(|e| metal_error(&e))?;
            session
                .add_rms_norm(
                    &self.scratch.hidden,
                    &self.scratch.normed,
                    self.weights
                        .layer(layer, LayerParameter::PreFeedforwardNorm)?,
                    &self.scratch.proj_out,
                    total_tokens,
                    hidden_size,
                    epsilon,
                )
                .map_err(|e| metal_error(&e))?;

            session
                .linear(
                    &self.scratch.proj_out,
                    self.weights.layer(layer, LayerParameter::Gate)?,
                    &self.scratch.ffn_gate,
                    total_tokens,
                    intermediate,
                    hidden_size,
                )
                .map_err(|e| metal_error(&e))?;
            session
                .linear(
                    &self.scratch.proj_out,
                    self.weights.layer(layer, LayerParameter::Up)?,
                    &self.scratch.ffn_up,
                    total_tokens,
                    intermediate,
                    hidden_size,
                )
                .map_err(|e| metal_error(&e))?;
            session
                .geglu_table(
                    &self.scratch.ffn_gate,
                    &self.scratch.ffn_up,
                    0,
                    &self.gelu_table,
                    &self.scratch.ffn_gate,
                    &self.scratch.error_flag,
                    total_tokens * intermediate,
                )
                .map_err(|e| metal_error(&e))?;
            session
                .linear(
                    &self.scratch.ffn_gate,
                    self.weights.layer(layer, LayerParameter::Down)?,
                    &self.scratch.proj_out,
                    total_tokens,
                    hidden_size,
                    intermediate,
                )
                .map_err(|e| metal_error(&e))?;
            session
                .rms_norm(
                    &self.scratch.proj_out,
                    Some(
                        self.weights
                            .layer(layer, LayerParameter::PostFeedforwardNorm)?,
                    ),
                    &self.scratch.normed,
                    total_tokens,
                    hidden_size,
                    epsilon,
                )
                .map_err(|e| metal_error(&e))?;
            session
                .residual_add_scale(
                    &self.scratch.hidden,
                    &self.scratch.normed,
                    None,
                    total_tokens * hidden_size,
                )
                .map_err(|e| metal_error(&e))?;

            session
                .linear(
                    &self.scratch.hidden,
                    self.weights.layer(layer, LayerParameter::PleGate)?,
                    &self.scratch.ple_gate,
                    total_tokens,
                    ple_dim,
                    hidden_size,
                )
                .map_err(|e| metal_error(&e))?;
            session
                .geglu_table(
                    &self.scratch.ple_gate,
                    &self.scratch.per_layer_ple,
                    layer * total_tokens * ple_dim,
                    &self.gelu_table,
                    &self.scratch.ple_gate,
                    &self.scratch.error_flag,
                    total_tokens * ple_dim,
                )
                .map_err(|e| metal_error(&e))?;
            session
                .linear(
                    &self.scratch.ple_gate,
                    self.weights.layer(layer, LayerParameter::PleProjection)?,
                    &self.scratch.proj_out,
                    total_tokens,
                    hidden_size,
                    ple_dim,
                )
                .map_err(|e| metal_error(&e))?;
            session
                .rms_norm(
                    &self.scratch.proj_out,
                    Some(self.weights.layer(layer, LayerParameter::PostPleNorm)?),
                    &self.scratch.normed,
                    total_tokens,
                    hidden_size,
                    epsilon,
                )
                .map_err(|e| metal_error(&e))?;
            session
                .residual_add_scale(
                    &self.scratch.hidden,
                    &self.scratch.normed,
                    Some(self.weights.layer(layer, LayerParameter::LayerScalar)?),
                    total_tokens * hidden_size,
                )
                .map_err(|e| metal_error(&e))?;
        }

        // 3. FinalNorm + Tied LM Head + Softcap + Argmax ONLY for output-producing rows
        if out_rows > 0 {
            let final_norm = self
                .weights
                .parameter(ParameterId::FinalNorm)
                .ok_or_else(|| Gemma4Error::new("Metal modelwide", "missing final norm"))?;
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
            session
                .linear(
                    &self.scratch.final_normed,
                    token_embd,
                    &self.scratch.logits,
                    out_rows,
                    config.vocab_size(),
                    hidden_size,
                )
                .map_err(|e| metal_error(&e))?;
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
