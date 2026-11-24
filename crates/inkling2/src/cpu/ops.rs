//! Pure, independently testable Inkling 2 operators matching `inkling2.cpp` and `modeling_inkling2.py`.
#![allow(clippy::cast_precision_loss)]

use crate::cpu::weights::{QuantizedMatrix, QuantizedTensor3d};

/// Weighted or unweighted RMS normalization over a single vector.
pub fn rms_norm_into(x: &[f32], weight: Option<&[f32]>, eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    if let Some(w) = weight {
        debug_assert_eq!(w.len(), x.len());
    }
    let mut sq_sum = 0.0_f32;
    for &v in x {
        sq_sum += v * v;
    }
    let inv_rms = 1.0_f32 / (sq_sum / (x.len() as f32) + eps).sqrt();
    match weight {
        Some(w) => {
            for ((dst, &xv), &wv) in out.iter_mut().zip(x).zip(w) {
                *dst = xv * inv_rms * wv;
            }
        }
        None => {
            for (dst, &xv) in out.iter_mut().zip(x) {
                *dst = xv * inv_rms;
            }
        }
    }
}

/// Allocate and return RMS-normalized vector.
#[must_use]
pub fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0_f32; x.len()];
    rms_norm_into(x, weight, eps, &mut out);
    out
}

/// Per-head L2 norm clipping after an optional per-channel scale:
/// `x = x * scale; x *= min(1.0, threshold / (||x||_2 + 1e-6))`.
pub fn l2_clip_heads_in_place(
    x: &mut [f32],
    scale: Option<&[f32]>,
    head_dim: usize,
    threshold: f32,
) {
    debug_assert!(x.len().is_multiple_of(head_dim));
    if let Some(s) = scale {
        debug_assert_eq!(s.len(), x.len());
        for (xv, &sv) in x.iter_mut().zip(s) {
            *xv *= sv;
        }
    }
    for head in x.chunks_exact_mut(head_dim) {
        let mut sq = 0.0_f32;
        for &v in head.iter() {
            sq += v * v;
        }
        let norm = sq.sqrt() + 1e-6_f32;
        let factor = (threshold / norm).clamp(0.0, 1.0);
        for v in head.iter_mut() {
            *v *= factor;
        }
    }
}

/// Unweighted RMS norm applied independently to each head vector.
pub fn rms_norm_heads_in_place(x: &mut [f32], head_dim: usize, eps: f32) {
    debug_assert!(x.len().is_multiple_of(head_dim));
    for head in x.chunks_exact_mut(head_dim) {
        let mut sq = 0.0_f32;
        for &v in head.iter() {
            sq += v * v;
        }
        let inv_rms = 1.0_f32 / (sq / (head_dim as f32) + eps).sqrt();
        for v in head.iter_mut() {
            *v *= inv_rms;
        }
    }
}

/// Full-attention position log scaling `tau = 1 + alpha * ln(max(1, (pos + 1) / floor))`.
#[must_use]
pub fn log_scale_tau(pos: usize, floor: usize, alpha: f32) -> f32 {
    let ratio = ((pos + 1) as f32 / (floor as f32)).max(1.0);
    1.0 + alpha * ratio.ln()
}

/// Depth-residual attention mix (`inkling2.cpp::mix`):
/// Stacks `blocks` and `cur`, applies unweighted RMS norm (`eps = 1e-6`), and
/// computes scaled dot-product attention with `n_queries` learned query vectors
/// of length `n_embd` (scale = `1 / sqrt(n_embd)`).
///
/// `blocks` contains slices of `[n_tokens * n_embd]`; `cur` has `[n_tokens * n_embd]`.
/// `queries` has length `n_queries * n_embd`.
/// Returns `n_queries` vectors, each of length `n_tokens * n_embd`.
#[must_use]
pub fn depth_mix(
    blocks: &[Vec<f32>],
    cur: &[f32],
    queries: &[f32],
    n_embd: usize,
    n_tokens: usize,
    n_queries: usize,
) -> Vec<Vec<f32>> {
    debug_assert_eq!(cur.len(), n_tokens * n_embd);
    debug_assert_eq!(queries.len(), n_queries * n_embd);

    let n_states = blocks.len() + 1;
    let scale = 1.0_f32 / (n_embd as f32).sqrt();
    let mut outputs = vec![vec![0.0_f32; n_tokens * n_embd]; n_queries];
    let mut norm_states = vec![0.0_f32; n_states * n_embd];
    let mut scores = vec![0.0_f32; n_states];

    for t in 0..n_tokens {
        let tok_off = t * n_embd;
        // Match `inkling2.cpp::mix` state order: blocks then cur (softmax weighted sum is order-invariant up to FP rounding).
        for (s, block) in blocks.iter().enumerate() {
            rms_norm_into(
                &block[tok_off..tok_off + n_embd],
                None,
                1e-6,
                &mut norm_states[s * n_embd..(s + 1) * n_embd],
            );
        }
        rms_norm_into(
            &cur[tok_off..tok_off + n_embd],
            None,
            1e-6,
            &mut norm_states[blocks.len() * n_embd..n_states * n_embd],
        );

        for q in 0..n_queries {
            let query = &queries[q * n_embd..(q + 1) * n_embd];
            let mut max_score = f32::NEG_INFINITY;
            for s in 0..n_states {
                let st = &norm_states[s * n_embd..(s + 1) * n_embd];
                let mut dot = 0.0_f32;
                for (&sv, &qv) in st.iter().zip(query) {
                    dot += sv * qv;
                }
                let sc = dot * scale;
                scores[s] = sc;
                if sc > max_score {
                    max_score = sc;
                }
            }
            let mut sum_exp = 0.0_f32;
            for sc in &mut scores {
                *sc = (*sc - max_score).exp();
                sum_exp += *sc;
            }
            let inv_sum = 1.0_f32 / sum_exp;
            let dst = &mut outputs[q][tok_off..tok_off + n_embd];
            dst.fill(0.0);
            for (s, &w_raw) in scores.iter().enumerate() {
                let w = w_raw * inv_sum;
                let st = &norm_states[s * n_embd..(s + 1) * n_embd];
                for (d, &sv) in dst.iter_mut().zip(st) {
                    *d += w * sv;
                }
            }
        }
    }

    outputs
}

/// Cross-pass loop controller (`inkling2.cpp::control`):
/// `gate = ctrl_down * concat(rmsnorm(cur), rmsnorm(message))`
/// `out = (cur - tanh(ctrl_forget * gate) * cur) + tanh(ctrl_gain * gate) * message`.
#[must_use]
pub fn loop_control(
    cur: &[f32],
    message: &[f32],
    ctrl_down: &QuantizedMatrix,
    ctrl_gain: &QuantizedMatrix,
    ctrl_forget: &QuantizedMatrix,
    n_embd: usize,
    n_tokens: usize,
) -> Vec<f32> {
    debug_assert_eq!(cur.len(), n_tokens * n_embd);
    debug_assert_eq!(message.len(), n_tokens * n_embd);
    let rank = ctrl_down.rows();
    let mut out = vec![0.0_f32; n_tokens * n_embd];
    let mut concat = vec![0.0_f32; 2 * n_embd];
    let mut gate = vec![0.0_f32; rank];
    let mut gain = vec![0.0_f32; n_embd];
    let mut forget = vec![0.0_f32; n_embd];

    for t in 0..n_tokens {
        let off = t * n_embd;
        rms_norm_into(&cur[off..off + n_embd], None, 1e-6, &mut concat[..n_embd]);
        rms_norm_into(
            &message[off..off + n_embd],
            None,
            1e-6,
            &mut concat[n_embd..],
        );
        ctrl_down.matvec(&concat, &mut gate);
        ctrl_gain.matvec(&gate, &mut gain);
        ctrl_forget.matvec(&gate, &mut forget);

        let dst = &mut out[off..off + n_embd];
        let c_tok = &cur[off..off + n_embd];
        let m_tok = &message[off..off + n_embd];
        for d in 0..n_embd {
            let g = gain[d].tanh();
            let f = forget[d].tanh();
            dst[d] = (c_tok[d] - f * c_tok[d]) + g * m_tok[d];
        }
    }

    out
}

/// Causal short convolution over `n_tokens` sequential tokens of one sequence,
/// mutating persistent `history` (`[n_channels, n_conv - 1]`, oldest at 0, newest at `n_conv - 2`).
///
/// - `weight` has layout `[n_channels, n_conv]` (GGUF `[n_conv, n_channels]`): tap `0` multiplies
///   the oldest window element (`t - (n_conv - 1)`), tap `n_conv - 1` multiplies current token `t`.
/// - `dynamic`, when present, has layout `[n_tokens, n_heads, n_conv]`: tap `lag = 0` multiplies
///   current token `t`, tap `lag = n_conv - 1` multiplies `t - (n_conv - 1)`.
#[must_use]
pub fn short_conv_causal(
    input: &[f32],
    weight: &[f32],
    dynamic: Option<(&[f32], usize)>,
    history: &mut [f32],
    n_channels: usize,
    n_conv: usize,
    n_tokens: usize,
) -> Vec<f32> {
    let hist_len = n_conv - 1;
    debug_assert_eq!(input.len(), n_tokens * n_channels);
    debug_assert_eq!(weight.len(), n_channels * n_conv);
    debug_assert_eq!(history.len(), n_channels * hist_len);

    let scale = 1.0_f32 / (((if dynamic.is_some() { 2 } else { 1 }) * n_conv + 1) as f32).sqrt();
    let mut out = vec![0.0_f32; n_tokens * n_channels];

    for t in 0..n_tokens {
        let in_tok = &input[t * n_channels..(t + 1) * n_channels];
        let out_tok = &mut out[t * n_channels..(t + 1) * n_channels];
        let dyn_tok = dynamic.map(|(d, n_heads)| {
            let stride = n_heads * n_conv;
            (&d[t * stride..(t + 1) * stride], n_channels / n_heads)
        });

        for c in 0..n_channels {
            let cur_val = in_tok[c];
            let h_slice = &mut history[c * hist_len..(c + 1) * hist_len];
            let w_slice = &weight[c * n_conv..(c + 1) * n_conv];

            // Static depthwise conv: history[0..hist_len] then cur_val
            let mut acc = 0.0_f32;
            for k in 0..hist_len {
                acc += h_slice[k] * w_slice[k];
            }
            acc += cur_val * w_slice[hist_len];

            // Optional per-head input-dependent taps (lag 0 = current token)
            if let Some((d_tok, head_dim)) = dyn_tok {
                let head = c / head_dim;
                let d_head = &d_tok[head * n_conv..(head + 1) * n_conv];
                acc += cur_val * d_head[0];
                for lag in 1..n_conv {
                    acc += h_slice[hist_len - lag] * d_head[lag];
                }
            }

            // Shift history left and append current input
            h_slice.copy_within(1..hist_len, 0);
            h_slice[hist_len - 1] = cur_val;

            out_tok[c] = (cur_val + acc) * scale;
        }
    }

    out
}

/// `SiTU` gated activation (`inkling2.cpp` `LLM_FFN_SITU`):
/// `beta * tanh(gate / beta) * sigmoid(gate) * linear_beta * tanh(up / linear_beta)`.
#[inline]
#[must_use]
pub fn situ(gate: f32, up: f32, beta: f32, linear_beta: f32) -> f32 {
    let act = beta * (gate / beta).tanh() * sigmoid(gate);
    let up_capped = if linear_beta > 0.0 {
        linear_beta * (up / linear_beta).tanh()
    } else {
        up
    };
    act * up_capped
}

#[inline]
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-x).exp())
}

/// Final logit softcapping `cap * tanh(x / cap)` when `cap > 0.0`.
pub fn softcap_in_place(logits: &mut [f32], cap: f32) {
    if cap > 0.0 {
        let inv = 1.0_f32 / cap;
        for v in logits.iter_mut() {
            *v = cap * (*v * inv).tanh();
        }
    }
}

/// `MoE` routing and expert evaluation for one token (`inkling2.cpp::build_moe_ffn`).
#[allow(clippy::too_many_arguments)]
pub fn moe_token(
    x: &[f32],
    gate_inp: &QuantizedMatrix,
    exp_probs_bias: Option<&[f32]>,
    gate_up_exps: &QuantizedTensor3d,
    down_exps: &QuantizedTensor3d,
    experts_used: usize,
    expert_weights_scale: f32,
    situ_beta: f32,
    situ_linear_beta: f32,
    out: &mut [f32],
) {
    let n_embd = x.len();
    let n_expert = gate_inp.rows();
    let n_ff = down_exps.cols();
    debug_assert_eq!(out.len(), n_embd);
    debug_assert_eq!(gate_up_exps.rows(), 2 * n_ff);

    let router_logits = gate_inp.matvec_vec(x);
    let mut probs = vec![0.0_f32; n_expert];
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(n_expert);
    for e in 0..n_expert {
        let p = sigmoid(router_logits[e]);
        probs[e] = p;
        let sel = p + exp_probs_bias.map_or(0.0, |b| b[e]);
        scored.push((e, sel));
    }

    // Descending by biased selection score; ties broken by smaller expert index.
    scored.sort_by(|(ia, sa), (ib, sb)| sb.total_cmp(sa).then_with(|| ia.cmp(ib)));
    scored.truncate(experts_used);

    let mut weight_sum = 0.0_f32;
    for &(e, _) in &scored {
        weight_sum += probs[e];
    }
    let denom = weight_sum + 1e-20_f32;

    out.fill(0.0);
    let mut gate_up = vec![0.0_f32; 2 * n_ff];
    let mut activated = vec![0.0_f32; n_ff];
    let mut expert_out = vec![0.0_f32; n_embd];

    for &(e, _) in &scored {
        let w = (probs[e] / denom) * expert_weights_scale;
        gate_up_exps.matvec_slice(e, x, &mut gate_up);
        let (gate, up) = gate_up.split_at(n_ff);
        for j in 0..n_ff {
            activated[j] = situ(gate[j], up[j], situ_beta, situ_linear_beta);
        }
        down_exps.matvec_slice(e, &activated, &mut expert_out);
        for (dst, &val) in out.iter_mut().zip(&expert_out) {
            *dst += w * val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_conv_chunked_equals_single_token_with_dynamic_taps() {
        let n_channels = 8;
        let n_heads = 2;
        let n_conv = 4;
        let n_tokens = 6;

        let input: Vec<f32> = (0..n_tokens * n_channels)
            .map(|i| ((i as f32) * 0.13).sin())
            .collect();
        let weight: Vec<f32> = (0..n_channels * n_conv)
            .map(|i| ((i as f32) * 0.27).cos() * 0.5)
            .collect();
        let dynamic: Vec<f32> = (0..n_tokens * n_heads * n_conv)
            .map(|i| ((i as f32) * 0.19).sin() * 0.25)
            .collect();

        let mut hist_chunk = vec![0.0_f32; n_channels * (n_conv - 1)];
        let out_chunk = short_conv_causal(
            &input,
            &weight,
            Some((&dynamic, n_heads)),
            &mut hist_chunk,
            n_channels,
            n_conv,
            n_tokens,
        );

        let mut hist_step = vec![0.0_f32; n_channels * (n_conv - 1)];
        let mut out_step = Vec::new();
        for t in 0..n_tokens {
            let dyn_t = &dynamic[t * n_heads * n_conv..(t + 1) * n_heads * n_conv];
            let y = short_conv_causal(
                &input[t * n_channels..(t + 1) * n_channels],
                &weight,
                Some((dyn_t, n_heads)),
                &mut hist_step,
                n_channels,
                n_conv,
                1,
            );
            out_step.extend(y);
        }

        assert_eq!(out_chunk, out_step);
        assert_eq!(hist_chunk, hist_step);
    }

    #[test]
    fn situ_and_l2_clip_match_analytical_formulas() {
        let y = situ(2.0, 10.0, 4.0, 25.0);
        let expected = (4.0_f32 * (0.5_f32).tanh() * (1.0 / (1.0 + (-2.0_f32).exp())))
            * (25.0_f32 * (10.0_f32 / 25.0).tanh());
        assert!((y - expected).abs() < 1e-6);

        let mut v = vec![3.0_f32, 4.0_f32, 0.3_f32, 0.4_f32];
        l2_clip_heads_in_place(&mut v, None, 2, 2.5);
        // First head had norm 5.0 -> scaled to ~2.5; second head had norm 0.5 < 2.5 -> unchanged
        let n0 = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((n0 - 2.5).abs() < 1e-4);
        assert!((v[2] - 0.3).abs() < 1e-6);
        assert!((v[3] - 0.4).abs() < 1e-6);
    }
}
