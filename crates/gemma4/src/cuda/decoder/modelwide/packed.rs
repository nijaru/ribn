//! Two independent histories sharing projections and layer work on one stream.
use super::{
    super::{
        AttentionGeometry, AttentionKind, AttentionSpan, Buffers, CudaBf16ComputeError,
        CudaBf16Error, CudaBf16Matrix, CudaBf16RotaryFactors, Executors, Gemma4Config,
        Gemma4CudaWeights, GeometryBuffers, HostLookups, LayerGeometry, LayerParameter,
        ParameterId, decode_bits, encode, layer_id, load, row, scaled_row, scales,
    },
    PhysicalSequence,
};
use engine_nvidia::CudaBf16Commands;

#[cfg(test)]
mod tests;

pub(super) struct LayerInputs {
    pub context: CudaBf16Matrix,
    pub cos: CudaBf16Matrix,
    pub sin: CudaBf16Matrix,
}
pub(super) struct MemberBuffers {
    pub query: CudaBf16Matrix,
    pub attended: CudaBf16Matrix,
}
pub(super) struct Packed {
    pub buffers: Buffers,
    pub layers: Vec<LayerInputs>,
    pub local: MemberBuffers,
    pub global: MemberBuffers,
    pub logits: CudaBf16Matrix,
}

impl Packed {
    pub(super) fn close(&mut self) -> Result<(), CudaBf16Error> {
        self.buffers.close()?;
        for inputs in &mut self.layers {
            inputs.context.close()?;
            inputs.cos.close()?;
            inputs.sin.close()?;
        }
        for buffers in [&mut self.local, &mut self.global] {
            buffers.query.close()?;
            buffers.attended.close()?;
        }
        self.logits.close()
    }

    pub(super) fn prepare(
        &mut self,
        host: &HostLookups,
        cuda: &mut Gemma4CudaWeights,
        executors: &mut Executors,
        factors: &mut [CudaBf16RotaryFactors],
        tokens: [u32; 2],
        positions: [u32; 2],
    ) -> Result<(), CudaBf16ComputeError> {
        let config = &host.config;
        let scale = scales(config);
        let mut hidden = Vec::with_capacity(2 * config.hidden_size());
        let mut identity = Vec::with_capacity(2 * config.packed_ple_dim());
        for token in tokens {
            hidden.extend(scaled_row(row(&host.token_embedding, token)?, scale.embed));
            identity.extend(scaled_row(
                row(&host.ple_embedding, token)?,
                scale.ple_embed,
            ));
        }
        self.buffers.hidden.upload(&encode(&hidden))?;
        cuda.linear_into(
            ParameterId::PleProjection,
            &mut executors.gemm,
            &mut self.buffers.hidden,
            &mut self.buffers.ple_projection,
        )?;
        executors.elementwise.scale_into(
            &self.buffers.ple_projection,
            scale.ple_projection,
            &mut self.buffers.ple_projected_scaled,
        )?;
        cuda.rms_norm_into(
            ParameterId::PleNorm,
            &mut executors.rms,
            &mut self.buffers.ple_projected_scaled,
            &mut self.buffers.ple_normed,
        )?;
        // One explicit PLE host boundary. Every layer context and rotary row is
        // staged before the dependent stack; no per-layer host wait remains there.
        let packed = config.packed_ple_dim();
        let ple = config.ple_dim();
        let projection = decode_bits(&self.buffers.ple_normed.read_chunk(0, 2 * packed * 2)?);
        for (layer, inputs) in self.layers.iter_mut().enumerate() {
            let mut context = Vec::with_capacity(2 * ple);
            for member in 0..2 {
                let base = member * packed + layer * ple;
                context.extend(
                    crate::ple_combine(&projection[base..base + ple], &identity[base..base + ple])
                        .map_err(|_| CudaBf16Error::Invalid("PLE combine failed"))?,
                );
            }
            inputs.context.upload(&encode(&context))?;
            factors[layer].prepare_into(&positions, &mut inputs.cos, &mut inputs.sin)?;
        }
        Ok(())
    }

    pub(super) fn enqueue(
        &mut self,
        host: &HostLookups,
        cuda: &mut Gemma4CudaWeights,
        executors: &mut Executors,
        mut members: [&mut PhysicalSequence; 2],
        commands: &mut CudaBf16Commands,
    ) -> Result<(), CudaBf16ComputeError> {
        let config = &host.config;
        for (layer, geometry) in config.layers().iter().enumerate() {
            let (buffers, member) = if geometry.head_dim == self.buffers.local.width {
                (&mut self.buffers.local, &mut self.local)
            } else {
                (&mut self.buffers.global, &mut self.global)
            };
            layer_forward(
                config,
                host,
                cuda,
                executors,
                &mut members,
                commands,
                &mut self.buffers.hidden,
                &mut self.buffers.hidden_next,
                &mut self.layers[layer],
                buffers,
                member,
                layer,
                geometry,
            )?;
        }
        // Keep the qualified M1 head arithmetic. Only shared layer projections
        // change GEMM geometry; each member gets its own complete vocabulary row.
        for member in 0..2 {
            commands.copy_rows(
                &mut self.buffers.final_input,
                0,
                &self.buffers.hidden,
                member..member + 1,
            )?;
            cuda.rms_norm_queued(
                ParameterId::FinalNorm,
                commands,
                &mut executors.rms,
                &mut self.buffers.final_input,
                &mut self.buffers.final_norm,
            )?;
            cuda.linear_queued(
                ParameterId::TokenEmbedding,
                commands,
                &mut executors.gemm,
                &mut self.buffers.final_norm,
                &mut self.buffers.logits,
            )?;
            commands.copy_rows(&mut self.logits, member, &self.buffers.logits, 0..1)?;
        }
        Ok(())
    }

    pub(super) fn read(
        &mut self,
        config: &Gemma4Config,
    ) -> Result<[Vec<u16>; 2], CudaBf16ComputeError> {
        let width = config.vocab_size();
        let raw = decode_bits(&self.logits.read_chunk(0, 2 * width * 2)?);
        let cap = |row: &[u16]| {
            crate::softcap(row, config.final_logit_softcapping())
                .map_err(|_| CudaBf16ComputeError::from(CudaBf16Error::Invalid("soft cap failed")))
        };
        Ok([cap(&raw[..width])?, cap(&raw[width..])?])
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "explicit layer resources and independent history bindings"
)]
#[expect(
    clippy::too_many_lines,
    reason = "ordered full-layer semantics, not a generic operator graph"
)]
fn layer_forward(
    config: &Gemma4Config,
    host: &HostLookups,
    cuda: &mut Gemma4CudaWeights,
    executors: &mut Executors,
    members: &mut [&mut PhysicalSequence; 2],
    commands: &mut CudaBf16Commands,
    hidden: &mut CudaBf16Matrix,
    hidden_next: &mut CudaBf16Matrix,
    inputs: &mut LayerInputs,
    buffers: &mut GeometryBuffers,
    member_buffers: &mut MemberBuffers,
    layer: usize,
    geometry: &LayerGeometry,
) -> Result<(), CudaBf16ComputeError> {
    let Executors {
        gemm,
        rms,
        rotary,
        elementwise,
        attention,
    } = executors;
    cuda.rms_norm_queued(
        layer_id(layer, LayerParameter::InputNorm),
        commands,
        rms,
        hidden,
        &mut buffers.normed,
    )?;
    cuda.linear_queued(
        layer_id(layer, LayerParameter::Query),
        commands,
        gemm,
        &mut buffers.normed,
        &mut buffers.queries,
    )?;
    cuda.rms_norm_queued(
        layer_id(layer, LayerParameter::QueryNorm),
        commands,
        rms,
        &mut buffers.queries,
        &mut buffers.query_norm,
    )?;
    commands.rotary_into(
        rotary,
        &mut inputs.cos,
        &mut inputs.sin,
        &mut buffers.query_norm,
        &mut buffers.query_rotary,
    )?;
    if geometry.kv_owner == layer {
        cuda.linear_queued(
            layer_id(layer, LayerParameter::Key),
            commands,
            gemm,
            &mut buffers.normed,
            &mut buffers.keys,
        )?;
        cuda.rms_norm_queued(
            layer_id(layer, LayerParameter::KeyNorm),
            commands,
            rms,
            &mut buffers.keys,
            &mut buffers.key_norm,
        )?;
        commands.rotary_into(
            rotary,
            &mut inputs.cos,
            &mut inputs.sin,
            &mut buffers.key_norm,
            &mut buffers.key_rotary,
        )?;
        cuda.linear_queued(
            layer_id(layer, LayerParameter::Value),
            commands,
            gemm,
            &mut buffers.normed,
            &mut buffers.values,
        )?;
        cuda.value_norm_queued(
            layer,
            commands,
            rms,
            &mut buffers.values,
            &mut buffers.value_norm,
        )?;
        for (member, state) in members.iter_mut().enumerate() {
            let cache = state.caches[layer]
                .as_mut()
                .ok_or(CudaBf16Error::Invalid("missing KV owner"))?;
            commands.copy_rows(
                &mut cache.keys,
                state.position,
                &buffers.key_rotary,
                member..member + 1,
            )?;
            commands.copy_rows(
                &mut cache.values,
                state.position,
                &buffers.value_norm,
                member..member + 1,
            )?;
        }
    }
    for (member, state) in members.iter().enumerate() {
        let cache = state.caches[geometry.kv_owner]
            .as_ref()
            .ok_or(CudaBf16Error::Invalid("missing KV owner"))?;
        commands.copy_rows(
            &mut member_buffers.query,
            0,
            &buffers.query_rotary,
            member..member + 1,
        )?;
        commands.attend_into(
            attention,
            &member_buffers.query,
            &cache.keys,
            &cache.values,
            AttentionGeometry {
                heads: config.num_attention_heads(),
                kv_heads: config.num_key_value_heads(),
                width: geometry.head_dim,
                window: match geometry.kind {
                    AttentionKind::Sliding => Some(config.sliding_window()),
                    AttentionKind::Full => None,
                },
            },
            AttentionSpan {
                query_start: state.position,
                key_start: 0,
                key_sequence: state.position + 1,
            },
            &mut member_buffers.attended,
        )?;
        commands.copy_rows(
            &mut buffers.attended,
            member,
            &member_buffers.attended,
            0..1,
        )?;
    }
    cuda.linear_queued(
        layer_id(layer, LayerParameter::AttentionOutput),
        commands,
        gemm,
        &mut buffers.attended,
        &mut buffers.projected,
    )?;
    cuda.rms_norm_queued(
        layer_id(layer, LayerParameter::PostAttentionNorm),
        commands,
        rms,
        &mut buffers.projected,
        &mut buffers.post_attention,
    )?;
    commands.add_into(elementwise, hidden, &buffers.post_attention, hidden_next)?;
    cuda.rms_norm_queued(
        layer_id(layer, LayerParameter::PreFeedforwardNorm),
        commands,
        rms,
        hidden_next,
        &mut buffers.ffn_norm,
    )?;
    cuda.linear_queued(
        layer_id(layer, LayerParameter::Gate),
        commands,
        gemm,
        &mut buffers.ffn_norm,
        &mut buffers.gate,
    )?;
    cuda.linear_queued(
        layer_id(layer, LayerParameter::Up),
        commands,
        gemm,
        &mut buffers.ffn_norm,
        &mut buffers.up,
    )?;
    commands.gelu_tanh_into(elementwise, &buffers.gate, &mut buffers.activated)?;
    commands.multiply_into(
        elementwise,
        &buffers.activated,
        &buffers.up,
        &mut buffers.product,
    )?;
    cuda.linear_queued(
        layer_id(layer, LayerParameter::Down),
        commands,
        gemm,
        &mut buffers.product,
        &mut buffers.down,
    )?;
    cuda.rms_norm_queued(
        layer_id(layer, LayerParameter::PostFeedforwardNorm),
        commands,
        rms,
        &mut buffers.down,
        &mut buffers.post_feedforward,
    )?;
    commands.add_into(elementwise, hidden_next, &buffers.post_feedforward, hidden)?;
    cuda.linear_queued(
        layer_id(layer, LayerParameter::PleGate),
        commands,
        gemm,
        hidden,
        &mut buffers.ple_gate,
    )?;
    commands.gelu_tanh_into(elementwise, &buffers.ple_gate, &mut buffers.ple_activated)?;
    commands.multiply_into(
        elementwise,
        &buffers.ple_activated,
        &inputs.context,
        &mut buffers.ple_product,
    )?;
    cuda.linear_queued(
        layer_id(layer, LayerParameter::PleProjection),
        commands,
        gemm,
        &mut buffers.ple_product,
        &mut buffers.ple_projected,
    )?;
    cuda.rms_norm_queued(
        layer_id(layer, LayerParameter::PostPleNorm),
        commands,
        rms,
        &mut buffers.ple_projected,
        &mut buffers.post_ple,
    )?;
    commands.add_into(elementwise, hidden, &buffers.post_ple, hidden_next)?;
    commands.scale_into(
        elementwise,
        hidden_next,
        load(host.layer_scalars[layer]),
        hidden,
    )?;
    Ok(())
}
