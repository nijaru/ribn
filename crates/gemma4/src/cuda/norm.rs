//! Model normalization semantics over backend-owned BF16 storage/completion.
use super::Gemma4CudaWeights;
use crate::{LayerParameter, ParameterId};
use engine_nvidia::{CudaBf16ComputeError, CudaBf16Error, CudaBf16Matrix, CudaBf16RmsNorm};

impl Gemma4CudaWeights {
    /// Apply one active learned RMS scale directly, with this model's epsilon.
    /// Q/K normalize each head; PLE normalizes each packed layer slice. Other
    /// learned norms normalize the hidden width. No residual/layer ordering is added.
    /// # Errors
    /// Rejects non-normalization/inactive roles or incorrect model widths before
    /// output mutation; backend failures retain the complete accessed storage.
    pub fn rms_norm_into(
        &mut self,
        parameter: ParameterId,
        execution: &mut CudaBf16RmsNorm,
        input: &mut CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        let hidden = self.config.hidden_size();
        let (width, columns) = match parameter {
            ParameterId::PleNorm => (self.config.ple_dim(), self.config.packed_ple_dim()),
            ParameterId::FinalNorm => (hidden, hidden),
            ParameterId::Layer { layer, parameter } => {
                let geometry = self
                    .config
                    .layers()
                    .get(layer)
                    .ok_or(CudaBf16Error::Invalid("RMS layer outside decoder"))?;
                match parameter {
                    LayerParameter::QueryNorm => (
                        geometry.head_dim,
                        self.config.num_attention_heads() * geometry.head_dim,
                    ),
                    LayerParameter::KeyNorm => (
                        geometry.head_dim,
                        self.config.num_key_value_heads() * geometry.head_dim,
                    ),
                    LayerParameter::InputNorm
                    | LayerParameter::PostAttentionNorm
                    | LayerParameter::PreFeedforwardNorm
                    | LayerParameter::PostFeedforwardNorm
                    | LayerParameter::PostPleNorm => (hidden, hidden),
                    _ => return Err(CudaBf16Error::Invalid("parameter is not an RMS scale").into()),
                }
            }
            _ => return Err(CudaBf16Error::Invalid("parameter is not an RMS scale").into()),
        };
        if input.shape().1 != columns {
            return Err(CudaBf16Error::Invalid("RMS input does not match model geometry").into());
        }
        let slot = self
            .parameter_slot(parameter)
            .ok_or(CudaBf16Error::Invalid("RMS scale has no active allocation"))?;
        execution.normalize_into(
            Some((&mut self.storage, slot)),
            width,
            self.config.rms_norm_eps(),
            input,
            output,
        )
    }

    /// Normalize fresh V projections without a learned scale, once at their KV
    /// owner. Shared-tail layers consume the owner's already-normalized values.
    /// # Errors
    /// Rejects invalid/non-owner layers, wrong widths or backend execution errors.
    pub fn value_norm_into(
        &mut self,
        layer: usize,
        execution: &mut CudaBf16RmsNorm,
        input: &mut CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        let geometry = self
            .config
            .layers()
            .get(layer)
            .ok_or(CudaBf16Error::Invalid("V RMS layer outside decoder"))?;
        if geometry.kv_owner != layer
            || input.shape().1 != self.config.num_key_value_heads() * geometry.head_dim
        {
            return Err(CudaBf16Error::Invalid("V RMS requires an owner projection").into());
        }
        execution.normalize_into(
            None,
            geometry.head_dim,
            self.config.rms_norm_eps(),
            input,
            output,
        )
    }
}
