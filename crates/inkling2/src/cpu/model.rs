//! Full Inkling 2 / Inkling 2 Loop CPU reference model and persistent sequence state.
#![allow(clippy::cast_precision_loss)]

use crate::Inkling2Error;
use crate::config::{AttentionKind, Inkling2Config};
use crate::continuation::KvCachePrecision;
use crate::cpu::ops::{
    depth_mix, l2_clip_heads_in_place, log_scale_tau, loop_control, moe_token,
    rms_norm_heads_in_place, rms_norm_into, short_conv_causal, sigmoid, softcap_in_place,
};
use crate::cpu::weights::{
    QuantizedMatrix, QuantizedTensor3d, f16_to_f32, f32_to_f16, load_f32_vec,
};
use crate::parameters::{
    GlobalParameter, LayerParameter, LoopController, LoopParameter, ParameterId,
};
use crate::source::Inkling2Source;

/// Optional diagnostic callback receiving `(step_index, state_slot, physical_layer, stage, data)`.
pub type LayerTap<'a> = &'a mut dyn FnMut(usize, usize, usize, &str, &[f32]);

/// Weights for one physical attention input path (SWA vs Full).
#[derive(Clone, Debug)]
enum AttentionInputWeights {
    Sliding {
        attn_norm: Vec<f32>,
    },
    Full {
        q_inp_norm: Vec<f32>,
        k_inp_norm: Vec<f32>,
        v_inp_norm: Vec<f32>,
        k_conv_proj: QuantizedMatrix,
        v_conv_proj: QuantizedMatrix,
    },
}

/// Weights for one physical transformer layer `blk.{ip}`.
#[derive(Clone, Debug)]
struct PhysicalLayerWeights {
    wq: QuantizedMatrix,
    wk: QuantizedMatrix,
    wv: QuantizedMatrix,
    wo: QuantizedMatrix,
    wqkv_gate: QuantizedMatrix,
    attn_q_clip: Vec<f32>,
    attn_sink_proj: QuantizedMatrix,
    attn_rel: QuantizedMatrix,
    attn_rel_bias: Vec<f32>,
    attn_rel_clip: Vec<f32>,
    attn_k_conv: Vec<f32>,
    attn_v_conv: Vec<f32>,
    attn_post_conv: Vec<f32>,
    ffn_post_conv: Vec<f32>,
    inp: AttentionInputWeights,
    attn_res_query: Option<Vec<f32>>,
    ffn_res_query: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate_inp: QuantizedMatrix,
    exp_probs_bias: Option<Vec<f32>>,
    ffn_gate_up_exps: QuantizedTensor3d,
    ffn_down_exps: QuantizedTensor3d,
}

/// Cross-pass loop controller weights for one core physical layer.
#[derive(Clone, Debug)]
struct LoopLayerWeights {
    msg_query: Vec<f32>,
    ctrl_down: [QuantizedMatrix; 4],
    ctrl_gain: [QuantizedMatrix; 4],
    ctrl_forget: [QuantizedMatrix; 4],
}

/// Persistent KV cache storage for one logical state slot.
#[derive(Clone, Debug)]
enum KvStorage {
    F16 { k: Vec<u16>, v: Vec<u16> },
    F32 { k: Vec<f32>, v: Vec<f32> },
}

impl KvStorage {
    fn new(precision: KvCachePrecision) -> Self {
        match precision {
            KvCachePrecision::F16 => Self::F16 {
                k: Vec::new(),
                v: Vec::new(),
            },
            KvCachePrecision::F32 => Self::F32 {
                k: Vec::new(),
                v: Vec::new(),
            },
        }
    }

    fn push_kv(&mut self, k_tok: &[f32], v_tok: &[f32]) {
        match self {
            Self::F16 { k, v } => {
                k.extend(k_tok.iter().map(|&x| f32_to_f16(x)));
                v.extend(v_tok.iter().map(|&x| f32_to_f16(x)));
            }
            Self::F32 { k, v } => {
                k.extend_from_slice(k_tok);
                v.extend_from_slice(v_tok);
            }
        }
    }

    #[inline]
    fn k_dot(&self, pos: usize, kv_dim: usize, kv_h: usize, head_dim: usize, q: &[f32]) -> f32 {
        let base = pos * kv_dim + kv_h * head_dim;
        let mut sum = 0.0_f32;
        match self {
            Self::F16 { k, .. } => {
                for (&kb, &qv) in k[base..base + head_dim].iter().zip(q) {
                    sum += f16_to_f32(kb) * qv;
                }
            }
            Self::F32 { k, .. } => {
                for (&kv, &qv) in k[base..base + head_dim].iter().zip(q) {
                    sum += kv * qv;
                }
            }
        }
        sum
    }

    #[inline]
    fn v_axpy(
        &self,
        pos: usize,
        kv_dim: usize,
        kv_h: usize,
        head_dim: usize,
        weight: f32,
        acc: &mut [f32],
    ) {
        let base = pos * kv_dim + kv_h * head_dim;
        match self {
            Self::F16 { v, .. } => {
                for (dst, &vb) in acc.iter_mut().zip(&v[base..base + head_dim]) {
                    *dst += weight * f16_to_f32(vb);
                }
            }
            Self::F32 { v, .. } => {
                for (dst, &vv) in acc.iter_mut().zip(&v[base..base + head_dim]) {
                    *dst += weight * vv;
                }
            }
        }
    }
}

/// Independent persistent continuation state for one logical state slot.
#[derive(Clone, Debug)]
struct SlotState {
    kv: KvStorage,
    k_conv: Vec<f32>,
    v_conv: Vec<f32>,
    attn_post_conv: Vec<f32>,
    ffn_post_conv: Vec<f32>,
}

/// Persistent per-sequence continuation state across all logical state slots.
#[derive(Clone, Debug)]
pub struct Inkling2CpuState {
    position: usize,
    precision: KvCachePrecision,
    slots: Vec<SlotState>,
}

impl Inkling2CpuState {
    /// Allocate fresh zero-initialized continuation state for `config`.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn new(config: &Inkling2Config, precision: KvCachePrecision) -> Self {
        let hist = config.conv_kernel() - 1;
        let hidden = config.hidden_size();
        let slots = (0..config.state_slot_count())
            .map(|slot| {
                let geom = config.slot_geometry(slot).expect("validated slot geometry");
                let kv_dim = geom.kv_dim();
                SlotState {
                    kv: KvStorage::new(precision),
                    k_conv: vec![0.0_f32; kv_dim * hist],
                    v_conv: vec![0.0_f32; kv_dim * hist],
                    attn_post_conv: vec![0.0_f32; hidden * hist],
                    ffn_post_conv: vec![0.0_f32; hidden * hist],
                }
            })
            .collect();
        Self {
            position: 0,
            precision,
            slots,
        }
    }

    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    #[must_use]
    pub const fn kv_precision(&self) -> KvCachePrecision {
        self.precision
    }

    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
}

/// Inkling 2 / Inkling 2 Loop CPU reference model.
#[derive(Clone, Debug)]
pub struct Inkling2CpuModel {
    config: Inkling2Config,
    token_embd: QuantizedMatrix,
    token_embd_norm: Vec<f32>,
    output: QuantizedMatrix,
    output_norm: Vec<f32>,
    output_res_query: Vec<f32>,
    layers: Vec<PhysicalLayerWeights>,
    loop_layers: Vec<LoopLayerWeights>,
}

impl Inkling2CpuModel {
    /// Load all physical parameters from a preflighted [`Inkling2Source`] in their
    /// compact native GGUF encodings (`F32`, `F16`, `Q8_0`, `IQ4_NL`).
    ///
    /// # Errors
    /// Returns [`Inkling2Error`] if any tensor fails to read or validate.
    #[allow(clippy::too_many_lines, clippy::missing_panics_doc)]
    pub fn load_from_source(source: &Inkling2Source) -> Result<Self, Inkling2Error> {
        let config = source.config().clone();
        let n_embd = config.hidden_size();
        let n_vocab = config.vocab_padded();
        let n_rel = config.relative_dim();
        let n_conv = config.conv_kernel();
        let n_exp = config.expert_count();

        let token_embd = QuantizedMatrix::load_from_source(
            source,
            ParameterId::Global(GlobalParameter::TokenEmbedding),
            n_vocab,
            n_embd,
        )?;
        let token_embd_norm = load_f32_vec(
            source,
            ParameterId::Global(GlobalParameter::TokenEmbeddingNorm),
            n_embd,
        )?;
        let output = QuantizedMatrix::load_from_source(
            source,
            ParameterId::Global(GlobalParameter::Output),
            n_vocab,
            n_embd,
        )?;
        let output_norm = load_f32_vec(
            source,
            ParameterId::Global(GlobalParameter::OutputNorm),
            n_embd,
        )?;
        let output_res_query = load_f32_vec(
            source,
            ParameterId::Global(GlobalParameter::OutputResQuery),
            n_embd,
        )?;

        let mut layers = Vec::with_capacity(config.physical_layer_count());
        for geom in config.physical_layers() {
            let i = geom.index;
            let q_dim = geom.q_dim();
            let kv_dim = geom.kv_dim();
            let lid = |parameter| ParameterId::Layer {
                layer: i,
                parameter,
            };

            let inp = match geom.kind {
                AttentionKind::Sliding => AttentionInputWeights::Sliding {
                    attn_norm: load_f32_vec(source, lid(LayerParameter::AttnNorm), n_embd)?,
                },
                AttentionKind::Full => AttentionInputWeights::Full {
                    q_inp_norm: load_f32_vec(source, lid(LayerParameter::AttnQInpNorm), n_embd)?,
                    k_inp_norm: load_f32_vec(source, lid(LayerParameter::AttnKInpNorm), n_embd)?,
                    v_inp_norm: load_f32_vec(source, lid(LayerParameter::AttnVInpNorm), n_embd)?,
                    k_conv_proj: QuantizedMatrix::load_from_source(
                        source,
                        lid(LayerParameter::AttnKConvProj),
                        n_conv * geom.kv_heads,
                        n_embd,
                    )?,
                    v_conv_proj: QuantizedMatrix::load_from_source(
                        source,
                        lid(LayerParameter::AttnVConvProj),
                        n_conv * geom.kv_heads,
                        n_embd,
                    )?,
                },
            };

            let attn_res_query = if geom.has_attn_res_query {
                let n_q = match geom.kind {
                    AttentionKind::Sliding => 1,
                    AttentionKind::Full => 3,
                };
                Some(load_f32_vec(
                    source,
                    lid(LayerParameter::AttnResQuery),
                    n_q * n_embd,
                )?)
            } else {
                None
            };

            let exp_probs_bias = if source
                .parameter_spec(lid(LayerParameter::ExpProbsBias))
                .is_some()
            {
                Some(load_f32_vec(
                    source,
                    lid(LayerParameter::ExpProbsBias),
                    n_exp,
                )?)
            } else {
                None
            };

            layers.push(PhysicalLayerWeights {
                wq: QuantizedMatrix::load_from_source(
                    source,
                    lid(LayerParameter::AttnQ),
                    q_dim,
                    n_embd,
                )?,
                wk: QuantizedMatrix::load_from_source(
                    source,
                    lid(LayerParameter::AttnK),
                    kv_dim,
                    n_embd,
                )?,
                wv: QuantizedMatrix::load_from_source(
                    source,
                    lid(LayerParameter::AttnV),
                    kv_dim,
                    n_embd,
                )?,
                wo: QuantizedMatrix::load_from_source(
                    source,
                    lid(LayerParameter::AttnOutput),
                    n_embd,
                    q_dim,
                )?,
                wqkv_gate: QuantizedMatrix::load_from_source(
                    source,
                    lid(LayerParameter::AttnGate),
                    q_dim,
                    n_embd,
                )?,
                attn_q_clip: load_f32_vec(source, lid(LayerParameter::AttnQClip), q_dim)?,
                attn_sink_proj: QuantizedMatrix::load_from_source(
                    source,
                    lid(LayerParameter::AttnSinkProj),
                    geom.heads,
                    n_embd,
                )?,
                attn_rel: QuantizedMatrix::load_from_source(
                    source,
                    lid(LayerParameter::AttnRel),
                    n_rel * geom.heads,
                    n_embd,
                )?,
                attn_rel_bias: load_f32_vec(
                    source,
                    lid(LayerParameter::AttnRelBias),
                    n_rel * geom.rel_extent,
                )?,
                attn_rel_clip: load_f32_vec(
                    source,
                    lid(LayerParameter::AttnRelClip),
                    n_rel * geom.heads,
                )?,
                attn_k_conv: load_f32_vec(source, lid(LayerParameter::AttnKConv), n_conv * kv_dim)?,
                attn_v_conv: load_f32_vec(source, lid(LayerParameter::AttnVConv), n_conv * kv_dim)?,
                attn_post_conv: load_f32_vec(
                    source,
                    lid(LayerParameter::AttnPostConv),
                    n_conv * n_embd,
                )?,
                ffn_post_conv: load_f32_vec(
                    source,
                    lid(LayerParameter::FfnPostConv),
                    n_conv * n_embd,
                )?,
                inp,
                attn_res_query,
                ffn_res_query: load_f32_vec(source, lid(LayerParameter::FfnResQuery), n_embd)?,
                ffn_norm: load_f32_vec(source, lid(LayerParameter::FfnNorm), n_embd)?,
                ffn_gate_inp: QuantizedMatrix::load_from_source(
                    source,
                    lid(LayerParameter::FfnGateInp),
                    n_exp,
                    n_embd,
                )?,
                exp_probs_bias,
                ffn_gate_up_exps: QuantizedTensor3d::load_from_source(
                    source,
                    lid(LayerParameter::FfnGateUpExps),
                    n_exp,
                    2 * geom.ff_expert,
                    n_embd,
                )?,
                ffn_down_exps: QuantizedTensor3d::load_from_source(
                    source,
                    lid(LayerParameter::FfnDownExps),
                    n_exp,
                    n_embd,
                    geom.ff_expert,
                )?,
            });
        }

        let mut loop_layers = Vec::new();
        if let Some(lc) = config.loop_config() {
            let rank = lc.gate_rank;
            loop_layers.reserve(lc.core_layer_count());
            for ip in lc.core_start..lc.core_end {
                let lp = |parameter| ParameterId::Loop {
                    layer: ip,
                    parameter,
                };
                let msg_query = load_f32_vec(source, lp(LoopParameter::MessageQuery), 4 * n_embd)?;
                let mut down = Vec::with_capacity(4);
                let mut gain = Vec::with_capacity(4);
                let mut forget = Vec::with_capacity(4);
                for ctrl in LoopController::ALL {
                    down.push(QuantizedMatrix::load_from_source(
                        source,
                        lp(LoopParameter::ControlDown(ctrl)),
                        rank,
                        2 * n_embd,
                    )?);
                    gain.push(QuantizedMatrix::load_from_source(
                        source,
                        lp(LoopParameter::ControlGain(ctrl)),
                        n_embd,
                        rank,
                    )?);
                    forget.push(QuantizedMatrix::load_from_source(
                        source,
                        lp(LoopParameter::ControlForget(ctrl)),
                        n_embd,
                        rank,
                    )?);
                }
                loop_layers.push(LoopLayerWeights {
                    msg_query,
                    ctrl_down: down.try_into().expect("4 controllers"),
                    ctrl_gain: gain.try_into().expect("4 controllers"),
                    ctrl_forget: forget.try_into().expect("4 controllers"),
                });
            }
        }

        Ok(Self {
            config,
            token_embd,
            token_embd_norm,
            output,
            output_norm,
            output_res_query,
            layers,
            loop_layers,
        })
    }

    #[must_use]
    pub const fn config(&self) -> &Inkling2Config {
        &self.config
    }

    /// Create fresh persistent continuation state for one sequence.
    #[must_use]
    pub fn new_state(&self, kv_precision: KvCachePrecision) -> Inkling2CpuState {
        Inkling2CpuState::new(&self.config, kv_precision)
    }

    /// Advance one sequence by `tokens`, updating persistent KV and short-conv state.
    /// When `compute_logits` is true, returns the final token's unpadded F32 logits
    /// (length `config.vocab_unpadded()`).
    ///
    /// # Errors
    /// Rejects empty token slices, out-of-range token IDs, mismatched state slot count,
    /// or context overflow.
    pub fn forward_tokens(
        &self,
        state: &mut Inkling2CpuState,
        tokens: &[u32],
        compute_logits: bool,
    ) -> Result<Option<Vec<f32>>, Inkling2Error> {
        self.forward_tokens_with_tap(state, tokens, compute_logits, None)
    }

    /// Same as [`Self::forward_tokens`], with an optional layer tap callback for localization.
    ///
    /// # Errors
    /// Rejects empty token slices, out-of-range token IDs, mismatched state slot count,
    /// or context overflow.
    #[allow(clippy::too_many_lines, clippy::missing_panics_doc)]
    pub fn forward_tokens_with_tap(
        &self,
        state: &mut Inkling2CpuState,
        tokens: &[u32],
        compute_logits: bool,
        mut tap: Option<LayerTap<'_>>,
    ) -> Result<Option<Vec<f32>>, Inkling2Error> {
        if tokens.is_empty() {
            return Err(Inkling2Error::new(
                "forward",
                "at least one input token is required",
            ));
        }
        if state.slots.len() != self.config.state_slot_count() {
            return Err(Inkling2Error::new(
                "state",
                "state slot count does not match model configuration",
            ));
        }
        let end_pos = state
            .position
            .checked_add(tokens.len())
            .ok_or_else(|| Inkling2Error::new("forward", "position overflow"))?;
        if end_pos > self.config.context_length() {
            return Err(Inkling2Error::new(
                "forward",
                format!(
                    "sequence end position {end_pos} exceeds context_length {}",
                    self.config.context_length()
                ),
            ));
        }
        let vocab_unpadded = self.config.vocab_unpadded();
        for &tok in tokens {
            if usize::try_from(tok).is_ok_and(|id| id >= vocab_unpadded) {
                return Err(Inkling2Error::new(
                    "token",
                    format!("token ID {tok} is outside unpadded vocabulary {vocab_unpadded}"),
                ));
            }
        }

        let start_pos = state.position;
        let n_tokens = tokens.len();
        let n_embd = self.config.hidden_size();
        let eps = self.config.rms_norm_eps();

        // 1. Token embedding + embedding RMSNorm
        let mut cur = vec![0.0_f32; n_tokens * n_embd];
        let mut raw_row = vec![0.0_f32; n_embd];
        for (t, &tok) in tokens.iter().enumerate() {
            self.token_embd.decode_row_into(tok as usize, &mut raw_row);
            rms_norm_into(
                &raw_row,
                Some(&self.token_embd_norm),
                eps,
                &mut cur[t * n_embd..(t + 1) * n_embd],
            );
        }

        // Precompute full-attention log-scaling factor tau per token
        let tau: Vec<f32> = (0..n_tokens)
            .map(|t| {
                log_scale_tau(
                    start_pos + t,
                    self.config.log_scale_floor(),
                    self.config.log_scale_alpha(),
                )
            })
            .collect();

        // 2. Execute physical/logical schedule (plain or prelude -> N loop passes -> epilogue)
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        let mut prelude: Option<Vec<f32>> = None;
        let mut prelude_blocks: Option<Vec<Vec<f32>>> = None;
        let mut prev_bank: Vec<Vec<f32>> = Vec::new();
        let mut prev_bank_out: Vec<f32> = Vec::new();

        for step in self.config.execution_schedule() {
            if step.starts_loop_pass {
                if step.loop_pass == Some(0) {
                    prelude = Some(cur.clone());
                    prelude_blocks = Some(blocks.clone());
                } else {
                    let pb = prelude_blocks.as_ref().expect("prelude captured on pass 0");
                    // Previous pass's depth bank: its blocks after the first core block, plus its output.
                    prev_bank = blocks[pb.len() + 1..].to_vec();
                    std::mem::swap(&mut prev_bank_out, &mut cur);
                    cur.clone_from(prelude.as_ref().expect("prelude captured on pass 0"));
                    blocks.clone_from(pb);
                }
            }

            let (messages, loop_weights) = if step.uses_cross_pass_control {
                let lc = self
                    .config
                    .loop_config()
                    .expect("cross-pass step requires loop config");
                let lw = &self.loop_layers[step.physical_layer - lc.core_start];
                let msgs = depth_mix(
                    &prev_bank,
                    &prev_bank_out,
                    &lw.msg_query,
                    n_embd,
                    n_tokens,
                    4,
                );
                (Some(msgs), Some(lw))
            } else {
                (None, None)
            };

            cur = self.run_layer(
                step.step_index,
                step.state_slot,
                step.physical_layer,
                start_pos,
                n_tokens,
                &tau,
                cur,
                &mut blocks,
                messages.as_deref(),
                loop_weights,
                &mut state.slots[step.state_slot],
                &mut tap,
            );
        }

        state.position = end_pos;

        if !compute_logits {
            return Ok(None);
        }

        // 3. Output depth-residual mix + output RMSNorm + unpadded lm_head + softcap
        let mixed = depth_mix(&blocks, &cur, &self.output_res_query, n_embd, n_tokens, 1);
        let last_off = (n_tokens - 1) * n_embd;
        let mut last_norm = vec![0.0_f32; n_embd];
        rms_norm_into(
            &mixed[0][last_off..last_off + n_embd],
            Some(&self.output_norm),
            eps,
            &mut last_norm,
        );
        if let Some(cb) = tap {
            cb(
                self.config.execution_schedule().len(),
                usize::MAX,
                usize::MAX,
                "result_norm",
                &last_norm,
            );
        }

        let mut logits = vec![0.0_f32; vocab_unpadded];
        self.output
            .matvec_prefix(&last_norm, vocab_unpadded, &mut logits);
        softcap_in_place(&mut logits, self.config.final_logit_softcapping());

        Ok(Some(logits))
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn run_layer(
        &self,
        step_index: usize,
        state_slot: usize,
        physical_layer: usize,
        start_pos: usize,
        n_tokens: usize,
        tau: &[f32],
        cur: Vec<f32>,
        blocks: &mut Vec<Vec<f32>>,
        messages: Option<&[Vec<f32>]>,
        loop_weights: Option<&LoopLayerWeights>,
        slot_state: &mut SlotState,
        tap: &mut Option<LayerTap<'_>>,
    ) -> Vec<f32> {
        let geom = &self.config.physical_layers()[physical_layer];
        let layer = &self.layers[physical_layer];
        let is_swa = geom.kind == AttentionKind::Sliding;
        let n_embd = self.config.hidden_size();
        let n_head = geom.heads;
        let n_head_kv = geom.kv_heads;
        let head_dim = geom.head_dim;
        let q_dim = geom.q_dim();
        let kv_dim = geom.kv_dim();
        let n_rel = self.config.relative_dim();
        let n_conv = self.config.conv_kernel();
        let eps = self.config.rms_norm_eps();

        let inp_sa = cur;

        // 1. Attention depth-residual mix (Q/K/V shared for SWA, 3 distinct queries for Full)
        let mut inputs: [Vec<f32>; 3] = if let Some(ref query) = layer.attn_res_query {
            if is_swa {
                let mut m = depth_mix(blocks, &inp_sa, query, n_embd, n_tokens, 1);
                let one = m.pop().expect("1 SWA query");
                [one.clone(), one.clone(), one]
            } else {
                let m = depth_mix(blocks, &inp_sa, query, n_embd, n_tokens, 3);
                m.try_into().expect("3 Full queries")
            }
        } else {
            [inp_sa.clone(), inp_sa.clone(), inp_sa.clone()]
        };

        // 2. Cross-pass loop controls on Q, K, V inputs (pass > 0)
        if let (Some(msgs), Some(lw)) = (messages, loop_weights) {
            for j in 0..3 {
                inputs[j] = loop_control(
                    &inputs[j],
                    &msgs[j],
                    &lw.ctrl_down[j],
                    &lw.ctrl_gain[j],
                    &lw.ctrl_forget[j],
                    n_embd,
                    n_tokens,
                );
            }
        }

        // 3. Attention input RMSNorms
        match &layer.inp {
            AttentionInputWeights::Sliding { attn_norm } => {
                if messages.is_some() {
                    for inp in &mut inputs {
                        norm_tokens_in_place(inp, attn_norm, eps, n_embd, n_tokens);
                    }
                } else {
                    norm_tokens_in_place(&mut inputs[0], attn_norm, eps, n_embd, n_tokens);
                    let first = inputs[0].clone();
                    inputs[1].clone_from(&first);
                    inputs[2].clone_from(&first);
                }
            }
            AttentionInputWeights::Full {
                q_inp_norm,
                k_inp_norm,
                v_inp_norm,
                ..
            } => {
                norm_tokens_in_place(&mut inputs[0], q_inp_norm, eps, n_embd, n_tokens);
                norm_tokens_in_place(&mut inputs[1], k_inp_norm, eps, n_embd, n_tokens);
                norm_tokens_in_place(&mut inputs[2], v_inp_norm, eps, n_embd, n_tokens);
            }
        }

        // 4. Q, K, V, R, sinks, and output gate projections
        let mut q_all = vec![0.0_f32; n_tokens * q_dim];
        let mut k_raw = vec![0.0_f32; n_tokens * kv_dim];
        let mut v_raw = vec![0.0_f32; n_tokens * kv_dim];
        let mut r_all = vec![0.0_f32; n_tokens * n_head * n_rel];
        let mut sinks_all = vec![0.0_f32; n_tokens * n_head];
        let mut gate_all = vec![0.0_f32; n_tokens * q_dim];

        let mut k_dyn_buf = match &layer.inp {
            AttentionInputWeights::Full { .. } => vec![0.0_f32; n_tokens * n_head_kv * n_conv],
            AttentionInputWeights::Sliding { .. } => Vec::new(),
        };
        let mut v_dyn_buf = match &layer.inp {
            AttentionInputWeights::Full { .. } => vec![0.0_f32; n_tokens * n_head_kv * n_conv],
            AttentionInputWeights::Sliding { .. } => Vec::new(),
        };

        for t in 0..n_tokens {
            let iq = &inputs[0][t * n_embd..(t + 1) * n_embd];
            let ik = &inputs[1][t * n_embd..(t + 1) * n_embd];
            let iv = &inputs[2][t * n_embd..(t + 1) * n_embd];

            layer.wq.matvec(iq, &mut q_all[t * q_dim..(t + 1) * q_dim]);
            layer
                .wk
                .matvec(ik, &mut k_raw[t * kv_dim..(t + 1) * kv_dim]);
            layer
                .wv
                .matvec(iv, &mut v_raw[t * kv_dim..(t + 1) * kv_dim]);
            layer
                .attn_rel
                .matvec(iq, &mut r_all[t * n_head * n_rel..(t + 1) * n_head * n_rel]);
            layer
                .attn_sink_proj
                .matvec(iq, &mut sinks_all[t * n_head..(t + 1) * n_head]);
            layer
                .wqkv_gate
                .matvec(iq, &mut gate_all[t * q_dim..(t + 1) * q_dim]);

            if let AttentionInputWeights::Full {
                k_conv_proj,
                v_conv_proj,
                ..
            } = &layer.inp
            {
                let stride = n_head_kv * n_conv;
                k_conv_proj.matvec(ik, &mut k_dyn_buf[t * stride..(t + 1) * stride]);
                v_conv_proj.matvec(iv, &mut v_dyn_buf[t * stride..(t + 1) * stride]);
            }
        }

        // 5. Short causal convolutions on K and V (dynamic on Full layers)
        let mut k_all = short_conv_causal(
            &k_raw,
            &layer.attn_k_conv,
            (!is_swa).then_some((k_dyn_buf.as_slice(), n_head_kv)),
            &mut slot_state.k_conv,
            kv_dim,
            n_conv,
            n_tokens,
        );
        let mut v_all = short_conv_causal(
            &v_raw,
            &layer.attn_v_conv,
            (!is_swa).then_some((v_dyn_buf.as_slice(), n_head_kv)),
            &mut slot_state.v_conv,
            kv_dim,
            n_conv,
            n_tokens,
        );

        // 6. Clip Q/K/R, unweighted RMSNorm on V, log-scale tau on Full Q/R
        for t in 0..n_tokens {
            let qt = &mut q_all[t * q_dim..(t + 1) * q_dim];
            let kt = &mut k_all[t * kv_dim..(t + 1) * kv_dim];
            let vt = &mut v_all[t * kv_dim..(t + 1) * kv_dim];
            let rt = &mut r_all[t * n_head * n_rel..(t + 1) * n_head * n_rel];

            l2_clip_heads_in_place(
                qt,
                Some(&layer.attn_q_clip),
                head_dim,
                self.config.qk_clip(),
            );
            l2_clip_heads_in_place(kt, None, head_dim, self.config.qk_clip());
            rms_norm_heads_in_place(vt, head_dim, eps);
            l2_clip_heads_in_place(
                rt,
                Some(&layer.attn_rel_clip),
                n_rel,
                self.config.relative_clip(),
            );

            if !is_swa {
                let scale = tau[t];
                for v in qt.iter_mut() {
                    *v *= scale;
                }
                for v in rt.iter_mut() {
                    *v *= scale;
                }
            }

            // Append K and V to this logical slot's KV cache
            slot_state.kv.push_kv(kt, vt);
        }

        // 7. Grouped-query attention with relative-position bias and per-query sink
        let attn_scale = 1.0_f32 / (head_dim as f32).sqrt();
        let gqa = geom.gqa_group_size();
        let window = self.config.sliding_window();
        let rel_extent = geom.rel_extent;
        let mut attn_concat = vec![0.0_f32; n_tokens * q_dim];
        let mut scores = Vec::new();

        for t in 0..n_tokens {
            let q_pos = start_pos + t;
            let k_start = if is_swa {
                q_pos.saturating_sub(window - 1)
            } else {
                0
            };
            let span_len = q_pos + 1 - k_start;
            scores.resize(span_len, 0.0);

            for h in 0..n_head {
                let kv_h = h / gqa;
                let q_vec = &q_all[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
                let r_vec = &r_all[(t * n_head + h) * n_rel..(t * n_head + h + 1) * n_rel];
                let sink = sinks_all[t * n_head + h];

                let mut max_s = sink;
                for (j, k_pos) in (k_start..=q_pos).enumerate() {
                    let dist = q_pos - k_pos;
                    let bias = if dist < rel_extent {
                        let b_row = &layer.attn_rel_bias[dist * n_rel..(dist + 1) * n_rel];
                        let mut b = 0.0_f32;
                        for (&bv, &rv) in b_row.iter().zip(r_vec) {
                            b += bv * rv;
                        }
                        b
                    } else {
                        0.0
                    };
                    let sc = slot_state.kv.k_dot(k_pos, kv_dim, kv_h, head_dim, q_vec) * attn_scale
                        + bias;
                    scores[j] = sc;
                    if sc > max_s {
                        max_s = sc;
                    }
                }

                let mut denom = (sink - max_s).exp();
                for sc in &mut scores[..span_len] {
                    *sc = (*sc - max_s).exp();
                    denom += *sc;
                }
                let inv_denom = 1.0_f32 / denom;

                let out_head =
                    &mut attn_concat[t * q_dim + h * head_dim..t * q_dim + (h + 1) * head_dim];
                out_head.fill(0.0);
                for (j, k_pos) in (k_start..=q_pos).enumerate() {
                    let w = scores[j] * inv_denom;
                    slot_state
                        .kv
                        .v_axpy(k_pos, kv_dim, kv_h, head_dim, w, out_head);
                }
            }
        }

        // 8. Output sigmoid gate + o_proj + post-attention short conv
        for (a, &g) in attn_concat.iter_mut().zip(&gate_all) {
            *a *= sigmoid(g);
        }
        let mut attn_proj = vec![0.0_f32; n_tokens * n_embd];
        for t in 0..n_tokens {
            layer.wo.matvec(
                &attn_concat[t * q_dim..(t + 1) * q_dim],
                &mut attn_proj[t * n_embd..(t + 1) * n_embd],
            );
        }
        let mut cur_post_attn = short_conv_causal(
            &attn_proj,
            &layer.attn_post_conv,
            None,
            &mut slot_state.attn_post_conv,
            n_embd,
            n_conv,
            n_tokens,
        );

        if let Some(cb) = tap {
            cb(
                step_index,
                state_slot,
                physical_layer,
                "attn_out",
                &cur_post_attn,
            );
        }

        // 9. Block boundary starts a fresh residual stream; non-boundary adds inp_sa
        if geom.is_block_boundary {
            blocks.push(inp_sa);
        } else {
            for (c, &r) in cur_post_attn.iter_mut().zip(&inp_sa) {
                *c += r;
            }
        }
        let ffn_inp = cur_post_attn;

        // 10. MoE input depth-residual mix + optional cross-pass FFN control + FFN RMSNorm
        let mut ffn_x = depth_mix(blocks, &ffn_inp, &layer.ffn_res_query, n_embd, n_tokens, 1)
            .pop()
            .expect("1 FFN res query");
        if let (Some(msgs), Some(lw)) = (messages, loop_weights) {
            ffn_x = loop_control(
                &ffn_x,
                &msgs[LoopController::Ffn as usize],
                &lw.ctrl_down[LoopController::Ffn as usize],
                &lw.ctrl_gain[LoopController::Ffn as usize],
                &lw.ctrl_forget[LoopController::Ffn as usize],
                n_embd,
                n_tokens,
            );
        }
        norm_tokens_in_place(&mut ffn_x, &layer.ffn_norm, eps, n_embd, n_tokens);

        // 11. MoE routing + SiTU experts + post-FFN short conv + residual
        let mut moe_out = vec![0.0_f32; n_tokens * n_embd];
        for t in 0..n_tokens {
            moe_token(
                &ffn_x[t * n_embd..(t + 1) * n_embd],
                &layer.ffn_gate_inp,
                layer.exp_probs_bias.as_deref(),
                &layer.ffn_gate_up_exps,
                &layer.ffn_down_exps,
                geom.experts_used,
                self.config.expert_weights_scale(),
                self.config.situ_beta(),
                self.config.situ_linear_beta(),
                &mut moe_out[t * n_embd..(t + 1) * n_embd],
            );
        }
        let mut ffn_out = short_conv_causal(
            &moe_out,
            &layer.ffn_post_conv,
            None,
            &mut slot_state.ffn_post_conv,
            n_embd,
            n_conv,
            n_tokens,
        );
        for (dst, &res) in ffn_out.iter_mut().zip(&ffn_inp) {
            *dst += res;
        }

        if let Some(cb) = tap {
            cb(step_index, state_slot, physical_layer, "l_out", &ffn_out);
        }

        ffn_out
    }
}

fn norm_tokens_in_place(buf: &mut [f32], weight: &[f32], eps: f32, n_embd: usize, n_tokens: usize) {
    let mut tmp = vec![0.0_f32; n_embd];
    for t in 0..n_tokens {
        let slice = &mut buf[t * n_embd..(t + 1) * n_embd];
        rms_norm_into(slice, Some(weight), eps, &mut tmp);
        slice.copy_from_slice(&tmp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_gguf::MetadataValue;
    use std::collections::BTreeMap;

    fn det_vec(len: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 37 + seed * 101) % 997) as f32 / 997.0 - 0.5) * 2.0 * scale)
            .collect()
    }

    fn det_mat(rows: usize, cols: usize, seed: usize, scale: f32) -> QuantizedMatrix {
        QuantizedMatrix::from_f32(rows, cols, det_vec(rows * cols, seed, scale)).unwrap()
    }

    fn det_t3(
        slices: usize,
        rows: usize,
        cols: usize,
        seed: usize,
        scale: f32,
    ) -> QuantizedTensor3d {
        QuantizedTensor3d::from_f32(
            slices,
            rows,
            cols,
            det_vec(slices * rows * cols, seed, scale),
        )
        .unwrap()
    }

    #[allow(clippy::too_many_lines)]
    fn synthetic_loop_model(passes: u32, sliding_window: u32) -> Inkling2CpuModel {
        // 6 physical layers, block_size=2, core=[2, 4), SWA on layers 0,2,4 and Full on 1,3,5
        let mut m = BTreeMap::new();
        m.insert(
            "general.architecture".into(),
            MetadataValue::String("inkling2_loop".into()),
        );
        m.insert("inkling2_loop.block_count".into(), MetadataValue::U32(6));
        m.insert(
            "inkling2_loop.context_length".into(),
            MetadataValue::U32(64),
        );
        m.insert(
            "inkling2_loop.embedding_length".into(),
            MetadataValue::U32(8),
        );
        m.insert(
            "inkling2_loop.attention.head_count".into(),
            MetadataValue::U32(4),
        );
        m.insert(
            "inkling2_loop.attention.head_count_kv".into(),
            MetadataValue::U32(2),
        );
        m.insert(
            "inkling2_loop.attention.key_length".into(),
            MetadataValue::U32(4),
        );
        m.insert(
            "inkling2_loop.attention.value_length".into(),
            MetadataValue::U32(4),
        );
        m.insert(
            "inkling2_loop.attention.layer_norm_rms_epsilon".into(),
            MetadataValue::F32(1e-6),
        );
        m.insert(
            "inkling2_loop.attention.qk_clip".into(),
            MetadataValue::F32(10.0),
        );
        m.insert(
            "inkling2_loop.attention.relative_length".into(),
            MetadataValue::U32(3),
        );
        m.insert(
            "inkling2_loop.attention.relative_extent".into(),
            MetadataValue::U32(8),
        );
        m.insert(
            "inkling2_loop.attention.relative_clip".into(),
            MetadataValue::F32(4.0),
        );
        m.insert(
            "inkling2_loop.attention.log_scale_floor".into(),
            MetadataValue::U32(4),
        );
        m.insert(
            "inkling2_loop.attention.log_scale_alpha".into(),
            MetadataValue::F32(0.1),
        );
        m.insert(
            "inkling2_loop.attention.sliding_window".into(),
            MetadataValue::U32(sliding_window),
        );
        m.insert(
            "inkling2_loop.attention.sliding_window_pattern".into(),
            MetadataValue::Array(vec![
                MetadataValue::Bool(true),
                MetadataValue::Bool(false),
                MetadataValue::Bool(true),
                MetadataValue::Bool(false),
                MetadataValue::Bool(true),
                MetadataValue::Bool(false),
            ]),
        );
        m.insert(
            "inkling2_loop.attn_res.block_size".into(),
            MetadataValue::U32(2),
        );
        m.insert(
            "inkling2_loop.shortconv.l_cache".into(),
            MetadataValue::U32(4),
        );
        m.insert("inkling2_loop.vocab_size".into(), MetadataValue::U32(20));
        m.insert(
            "inkling2_loop.vocab_size_unpadded".into(),
            MetadataValue::U32(16),
        );
        m.insert("inkling2_loop.expert_count".into(), MetadataValue::U32(4));
        m.insert(
            "inkling2_loop.expert_used_count".into(),
            MetadataValue::U32(2),
        );
        m.insert(
            "inkling2_loop.expert_feed_forward_length".into(),
            MetadataValue::U32(6),
        );
        m.insert(
            "inkling2_loop.expert_weights_scale".into(),
            MetadataValue::F32(2.0),
        );
        m.insert(
            "inkling2_loop.activation.situ_beta".into(),
            MetadataValue::F32(4.0),
        );
        m.insert(
            "inkling2_loop.activation.situ_linear_beta".into(),
            MetadataValue::F32(25.0),
        );
        m.insert(
            "inkling2_loop.final_logit_softcapping".into(),
            MetadataValue::F32(30.0),
        );
        m.insert("inkling2_loop.loop.start".into(), MetadataValue::U32(2));
        m.insert("inkling2_loop.loop.end".into(), MetadataValue::U32(4));
        m.insert(
            "inkling2_loop.loop.count".into(),
            MetadataValue::U32(passes),
        );
        m.insert("inkling2_loop.loop.gate_rank".into(), MetadataValue::U32(4));
        m.insert(
            "inkling2_loop.loop.varying".into(),
            MetadataValue::Bool(passes == 1),
        );

        let cfg = Inkling2Config::from_metadata(&m).unwrap();
        let n_embd = cfg.hidden_size();
        let n_rel = cfg.relative_dim();
        let n_conv = cfg.conv_kernel();
        let n_exp = cfg.expert_count();

        let layers = cfg
            .physical_layers()
            .iter()
            .map(|g| {
                let s = g.index * 50 + 10;
                let inp = match g.kind {
                    AttentionKind::Sliding => AttentionInputWeights::Sliding {
                        attn_norm: vec![1.0; n_embd],
                    },
                    AttentionKind::Full => AttentionInputWeights::Full {
                        q_inp_norm: vec![1.0; n_embd],
                        k_inp_norm: vec![1.0; n_embd],
                        v_inp_norm: vec![1.0; n_embd],
                        k_conv_proj: det_mat(n_conv * g.kv_heads, n_embd, s + 1, 0.2),
                        v_conv_proj: det_mat(n_conv * g.kv_heads, n_embd, s + 2, 0.2),
                    },
                };
                PhysicalLayerWeights {
                    wq: det_mat(g.q_dim(), n_embd, s + 3, 0.25),
                    wk: det_mat(g.kv_dim(), n_embd, s + 4, 0.25),
                    wv: det_mat(g.kv_dim(), n_embd, s + 5, 0.25),
                    wo: det_mat(n_embd, g.q_dim(), s + 6, 0.25),
                    wqkv_gate: det_mat(g.q_dim(), n_embd, s + 7, 0.25),
                    attn_q_clip: vec![1.0; g.q_dim()],
                    attn_sink_proj: det_mat(g.heads, n_embd, s + 8, 0.2),
                    attn_rel: det_mat(n_rel * g.heads, n_embd, s + 9, 0.25),
                    attn_rel_bias: det_vec(n_rel * g.rel_extent, s + 10, 0.3),
                    attn_rel_clip: vec![1.0; n_rel * g.heads],
                    attn_k_conv: det_vec(n_conv * g.kv_dim(), s + 11, 0.2),
                    attn_v_conv: det_vec(n_conv * g.kv_dim(), s + 12, 0.2),
                    attn_post_conv: det_vec(n_conv * n_embd, s + 13, 0.2),
                    ffn_post_conv: det_vec(n_conv * n_embd, s + 14, 0.2),
                    inp,
                    attn_res_query: g.has_attn_res_query.then(|| {
                        let nq = if g.kind == AttentionKind::Sliding {
                            1
                        } else {
                            3
                        };
                        det_vec(nq * n_embd, s + 15, 0.2)
                    }),
                    ffn_res_query: det_vec(n_embd, s + 16, 0.2),
                    ffn_norm: vec![1.0; n_embd],
                    ffn_gate_inp: det_mat(n_exp, n_embd, s + 17, 0.3),
                    exp_probs_bias: Some(det_vec(n_exp, s + 18, 0.1)),
                    ffn_gate_up_exps: det_t3(n_exp, 2 * g.ff_expert, n_embd, s + 19, 0.2),
                    ffn_down_exps: det_t3(n_exp, n_embd, g.ff_expert, s + 20, 0.2),
                }
            })
            .collect();

        let lc = cfg.loop_config().unwrap();
        let loop_layers = (lc.core_start..lc.core_end)
            .map(|ip| {
                let s = 500 + ip * 30;
                LoopLayerWeights {
                    msg_query: det_vec(4 * n_embd, s, 0.2),
                    ctrl_down: std::array::from_fn(|c| {
                        det_mat(lc.gate_rank, 2 * n_embd, s + 1 + c, 0.2)
                    }),
                    ctrl_gain: std::array::from_fn(|c| {
                        det_mat(n_embd, lc.gate_rank, s + 5 + c, 0.2)
                    }),
                    ctrl_forget: std::array::from_fn(|c| {
                        det_mat(n_embd, lc.gate_rank, s + 9 + c, 0.2)
                    }),
                }
            })
            .collect();

        Inkling2CpuModel {
            config: cfg,
            token_embd: det_mat(20, n_embd, 1, 0.5),
            token_embd_norm: vec![1.0; n_embd],
            output: det_mat(20, n_embd, 2, 0.4),
            output_norm: vec![1.0; n_embd],
            output_res_query: det_vec(n_embd, 3, 0.2),
            layers,
            loop_layers,
        }
    }

    #[test]
    fn looped_chunked_prefill_and_continuation_matches_token_by_token_across_swa_boundary() {
        let model = synthetic_loop_model(3, 5);
        assert_eq!(model.config().state_slot_count(), 10);

        let tokens: Vec<u32> = vec![1, 5, 3, 12, 0, 7, 15, 2, 9, 4, 11];

        // Path A: single-token step-by-step
        let mut st_single = model.new_state(KvCachePrecision::F16);
        let mut logits_single = Vec::new();
        for &tok in &tokens {
            logits_single = model
                .forward_tokens(&mut st_single, &[tok], true)
                .unwrap()
                .unwrap();
        }

        // Path B: multi-chunk prefill (4 + 3 + 2) followed by 2 continuation steps (1 + 1)
        let mut st_chunked = model.new_state(KvCachePrecision::F16);
        assert!(
            model
                .forward_tokens(&mut st_chunked, &tokens[0..4], false)
                .unwrap()
                .is_none()
        );
        assert!(
            model
                .forward_tokens(&mut st_chunked, &tokens[4..7], false)
                .unwrap()
                .is_none()
        );
        let _ = model
            .forward_tokens(&mut st_chunked, &tokens[7..9], true)
            .unwrap()
            .unwrap();
        let _ = model
            .forward_tokens(&mut st_chunked, &tokens[9..10], true)
            .unwrap()
            .unwrap();
        let logits_chunked = model
            .forward_tokens(&mut st_chunked, &tokens[10..11], true)
            .unwrap()
            .unwrap();

        assert_eq!(logits_single.len(), 16);
        let max_diff = logits_single
            .iter()
            .zip(&logits_chunked)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "chunked vs single-token across loop passes and SWA boundary diverged: max_diff={max_diff}"
        );
    }

    #[test]
    fn loop_passes_keep_independent_kv_and_conv_state_slots() {
        let model = synthetic_loop_model(3, 8);
        let mut st = model.new_state(KvCachePrecision::F32);
        let logits = model
            .forward_tokens(&mut st, &[2, 4, 6, 8], true)
            .unwrap()
            .unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));

        // Physical core layer 2 maps to logical slots 2 (pass 0), 6 (pass 1), and 8 (pass 2).
        // Because pass 1 and pass 2 receive cross-pass loop controls, their slot states must differ!
        assert_ne!(st.slots[2].k_conv, st.slots[6].k_conv);
        assert_ne!(st.slots[6].k_conv, st.slots[8].k_conv);
        assert_ne!(st.slots[2].ffn_post_conv, st.slots[6].ffn_post_conv);
    }
}
