//! State-address lowering for qualified per-member and candidate descriptor execution.
use super::{
    ATTN_HEAD_DIM, ATTN_KV_HEADS, ATTN_Q_HEADS, BatchScratch, CudaDecodeError, CudaHybridState,
    CudaQwen35Ops, CudaSlice, CudaStream, GDN_HEAD_DIM, GDN_K_HEADS, GDN_K_OFFSET, GDN_QKV_DIM,
    GDN_V_HEADS, GDN_V_OFFSET, member_row, member_row_mut, missing_kv, missing_recurrent,
};
use crate::model_ops::member::MemberOps;
use crate::state::descriptor::BoundDescriptor;
use std::sync::Arc;

pub(super) enum BatchStates<'a, 's> {
    PerMember(&'a mut [&'s mut CudaHybridState]),
    Descriptor(BoundDescriptor<'a>, &'a MemberOps),
}

impl BatchStates<'_, '_> {
    fn members(&self) -> usize {
        match self {
            Self::PerMember(states) => states.len(),
            Self::Descriptor(bound, _) => bound.layout.members,
        }
    }

    pub(super) fn convolve(
        &mut self,
        ops: &CudaQwen35Ops,
        scratch: &mut BatchScratch,
        slot: u32,
        conv_weight: &CudaSlice<f32>,
    ) -> Result<(), CudaDecodeError> {
        let m = self.members();

        match self {
            BatchStates::Descriptor(bound, ops) => {
                ops.conv(
                    bound,
                    slot,
                    &scratch.qkv_mixed,
                    conv_weight,
                    &mut scratch.conv_out,
                )?;
            }
            BatchStates::PerMember(states) => {
                // Per member: conv history + recurrent matrix live in the member's
                // state; scratch rows are sliced views of the batch buffers.
                for member in 0..m {
                    let state = states.get_mut(member).ok_or_else(|| {
                        CudaDecodeError::InvalidPlan("missing member state".to_owned())
                    })?;
                    let recurrent = state.recurrent_mut().ok_or(missing_recurrent())?;
                    let (_, conv_buffer) = recurrent.layer_mut(slot).ok_or_else(|| {
                        CudaDecodeError::InvalidPlan(format!("recurrent state lacks slot {slot}"))
                    })?;
                    let history = conv_buffer.as_f32_mut().ok_or_else(|| {
                        CudaDecodeError::InvalidPlan(
                            "recurrent convolution history must be F32".to_owned(),
                        )
                    })?;

                    let qkv =
                        member_row(&scratch.qkv_mixed, member, GDN_QKV_DIM).ok_or_else(|| {
                            CudaDecodeError::Driver("qkv row out of range".to_owned())
                        })?;
                    let mut conv_out = member_row_mut(&mut scratch.conv_out, member, GDN_QKV_DIM)
                        .ok_or_else(|| {
                        CudaDecodeError::Driver("conv_out row out of range".to_owned())
                    })?;
                    ops.gdn_conv_silu_views(
                        &qkv,
                        conv_weight,
                        history,
                        &mut conv_out,
                        GDN_QKV_DIM,
                    )?;
                }
            }
        }

        Ok(())
    }

    pub(super) fn gather_qk(
        &mut self,
        stream: &Arc<CudaStream>,
        scratch: &mut BatchScratch,
        slot: u32,
    ) -> Result<(), CudaDecodeError> {
        let m = self.members();
        match self {
            BatchStates::Descriptor(bound, ops) => {
                ops.gather_qk(
                    bound,
                    slot,
                    &scratch.conv_out,
                    &mut scratch.gdn_q,
                    &mut scratch.gdn_k,
                )?;
            }
            BatchStates::PerMember(_) => {
                for member in 0..m {
                    let row = member * GDN_QKV_DIM;
                    let conv_q = scratch
                        .conv_out
                        .try_slice(row..row + GDN_K_OFFSET)
                        .ok_or_else(|| {
                            CudaDecodeError::Driver("conv q slice out of range".to_owned())
                        })?;
                    let mut gdn_q = member_row_mut(&mut scratch.gdn_q, member, GDN_K_OFFSET)
                        .ok_or_else(|| {
                            CudaDecodeError::Driver("gdn_q row out of range".to_owned())
                        })?;
                    stream
                        .memcpy_dtod(&conv_q, &mut gdn_q)
                        .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
                    let conv_k = scratch
                        .conv_out
                        .try_slice(row + GDN_K_OFFSET..row + GDN_V_OFFSET)
                        .ok_or_else(|| {
                            CudaDecodeError::Driver("conv k slice out of range".to_owned())
                        })?;
                    let mut gdn_k = member_row_mut(&mut scratch.gdn_k, member, GDN_K_OFFSET)
                        .ok_or_else(|| {
                            CudaDecodeError::Driver("gdn_k row out of range".to_owned())
                        })?;
                    stream
                        .memcpy_dtod(&conv_k, &mut gdn_k)
                        .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
                }
            }
        }

        Ok(())
    }

    pub(super) fn update_recurrent(
        &mut self,
        ops: &CudaQwen35Ops,
        scratch: &mut BatchScratch,
        slot: u32,
    ) -> Result<(), CudaDecodeError> {
        let m = self.members();
        match self {
            BatchStates::Descriptor(bound, ops) => {
                ops.state(
                    bound,
                    slot,
                    &scratch.gdn_q,
                    &scratch.gdn_k,
                    &scratch.conv_out,
                    &scratch.decay,
                    &scratch.beta,
                    &mut scratch.state_out,
                )?;
            }
            BatchStates::PerMember(states) => {
                let mut matrices: Vec<&mut CudaSlice<f32>> = Vec::with_capacity(m);
                for state in states.iter_mut() {
                    let recurrent = state.recurrent_mut().ok_or(missing_recurrent())?;
                    let (matrix_buffer, _conv_buffer) =
                        recurrent.layer_mut(slot).ok_or_else(|| {
                            CudaDecodeError::InvalidPlan(format!(
                                "recurrent state lacks slot {slot}"
                            ))
                        })?;
                    let matrix = matrix_buffer.as_f32_mut().ok_or_else(|| {
                        CudaDecodeError::InvalidPlan("recurrent matrix must be F32".to_owned())
                    })?;
                    matrices.push(matrix);
                }
                ops.gdn_state_update_batch(
                    &mut matrices,
                    &mut scratch.gdn_matrix_pads,
                    &scratch.gdn_q,
                    &scratch.gdn_k,
                    &scratch.conv_out,
                    &scratch.decay,
                    &scratch.beta,
                    &mut scratch.state_out,
                    GDN_V_HEADS,
                    GDN_K_HEADS,
                    GDN_HEAD_DIM,
                    GDN_V_OFFSET,
                )?;
            }
        }

        Ok(())
    }

    #[expect(
        clippy::too_many_lines,
        reason = "keeps append, score scratch and attention borrows together for each lowering"
    )]
    pub(super) fn attend(
        &mut self,
        ops: &CudaQwen35Ops,
        scratch: &mut BatchScratch,
        slot: u32,
        positions: &[u32],
    ) -> Result<(), CudaDecodeError> {
        let m = self.members();
        match self {
            BatchStates::Descriptor(bound, ops) => {
                ops.append(bound, slot, &scratch.k_normed, &scratch.v_raw)?;
                ops.attention(
                    bound,
                    slot,
                    &scratch.q_packed,
                    &scratch.q_raw,
                    scratch.scores.as_mut().ok_or_else(|| {
                        CudaDecodeError::InvalidPlan("score scratch is unset".to_owned())
                    })?,
                    &mut scratch.attn_out,
                )?;
            }
            BatchStates::PerMember(states) => {
                // Per member: append to this member's KV cache, then score attention
                // against it, using member views of the batch scratch.
                for (member, &position_u32) in positions.iter().enumerate().take(m) {
                    let position = usize::try_from(position_u32)
                        .map_err(|_| CudaDecodeError::Driver("position overflowed".to_owned()))?;
                    let state = states.get_mut(member).ok_or_else(|| {
                        CudaDecodeError::InvalidPlan("missing member state".to_owned())
                    })?;
                    // Read the capacity and this row's physical page before borrowing
                    // the cache mutably.
                    let capacity = state
                        .kv()
                        .map_or(usize::MAX, |kv| kv.spec().capacity_tokens() as usize);
                    let stride = capacity;
                    let tokens = position + 1;
                    let kv = state.kv_mut().ok_or(missing_kv())?;
                    let page_tokens = usize::try_from(kv.spec().page_tokens()).map_err(|_| {
                        CudaDecodeError::InvalidPlan("KV page size overflows usize".to_owned())
                    })?;
                    let physical = kv.physical_token(position_u32).ok_or_else(|| {
                        CudaDecodeError::InvalidPlan(
                            "KV page table lacks the row's page".to_owned(),
                        )
                    })?;
                    let (table, keys_buffer, values_buffer) =
                        kv.layer_paged_mut(slot).ok_or_else(|| {
                            CudaDecodeError::InvalidPlan(format!("KV state lacks slot {slot}"))
                        })?;
                    let cache_keys = keys_buffer.as_f16_mut().ok_or_else(|| {
                        CudaDecodeError::InvalidPlan("KV cache must be F16".to_owned())
                    })?;
                    let cache_values = values_buffer.as_f16_mut().ok_or_else(|| {
                        CudaDecodeError::InvalidPlan("KV cache must be F16".to_owned())
                    })?;
                    let k_member =
                        member_row(&scratch.k_normed, member, ATTN_KV_HEADS * ATTN_HEAD_DIM)
                            .ok_or_else(|| {
                                CudaDecodeError::Driver("k_normed row out of range".to_owned())
                            })?;
                    let v_member =
                        member_row(&scratch.v_raw, member, ATTN_KV_HEADS * ATTN_HEAD_DIM)
                            .ok_or_else(|| {
                                CudaDecodeError::Driver("v_raw row out of range".to_owned())
                            })?;
                    ops.kv_append_f16_views(
                        &k_member,
                        &v_member,
                        cache_keys,
                        cache_values,
                        physical,
                        ATTN_KV_HEADS,
                        ATTN_HEAD_DIM,
                    )?;
                    let cache_keys = keys_buffer.as_f16().ok_or_else(|| {
                        CudaDecodeError::InvalidPlan("KV cache must be F16".to_owned())
                    })?;
                    let cache_values = values_buffer.as_f16().ok_or_else(|| {
                        CudaDecodeError::InvalidPlan("KV cache must be F16".to_owned())
                    })?;

                    let q_member =
                        member_row(&scratch.q_packed, member, ATTN_Q_HEADS * ATTN_HEAD_DIM)
                            .ok_or_else(|| {
                                CudaDecodeError::Driver("q_packed row out of range".to_owned())
                            })?;
                    let gate_member =
                        member_row(&scratch.q_raw, member, ATTN_Q_HEADS * 2 * ATTN_HEAD_DIM)
                            .ok_or_else(|| {
                                CudaDecodeError::Driver("q_raw row out of range".to_owned())
                            })?;
                    let scores = scratch.scores.as_mut().ok_or_else(|| {
                        CudaDecodeError::InvalidPlan("score scratch is unset".to_owned())
                    })?;
                    // No member-major offset: score/softmax/weighted-sum for this
                    // member are queued before the next member overwrites the buffer.
                    // The full allocation remains executor-owned through completion.
                    let mut scores_member = scores
                        .try_slice_mut(..ATTN_Q_HEADS * stride)
                        .ok_or_else(|| {
                            CudaDecodeError::Driver("scores row out of range".to_owned())
                        })?;
                    let mut out_member =
                        member_row_mut(&mut scratch.attn_out, member, ATTN_Q_HEADS * ATTN_HEAD_DIM)
                            .ok_or_else(|| {
                                CudaDecodeError::Driver("attn_out row out of range".to_owned())
                            })?;
                    ops.attn_score_gqa_paged_views(
                        &q_member,
                        cache_keys,
                        cache_values,
                        table,
                        page_tokens,
                        &gate_member,
                        &mut scores_member,
                        &mut out_member,
                        tokens,
                        1,
                        stride,
                        ATTN_Q_HEADS,
                        ATTN_KV_HEADS,
                        ATTN_HEAD_DIM,
                    )?;
                }
            }
        }

        Ok(())
    }
}
