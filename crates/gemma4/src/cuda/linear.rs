use super::Gemma4CudaWeights;
use crate::ParameterId;
use engine_nvidia::{CudaBf16Error, CudaBf16Gemm, CudaBf16GemmError, CudaBf16Matrix};

impl Gemma4CudaWeights {
    /// Bind one active logical projection to native BF16 linear execution.
    /// `TokenEmbedding` is the raw tied output head, not a scaled embedding lookup.
    /// Norms, activation functions, `RoPE` and layer ordering remain model semantics;
    /// this operation alone is not a Gemma decoder.
    ///
    /// # Errors
    /// Rejects inactive/nonmatrix roles, incompatible dimensions/streams or unready
    /// operands before changing output. A backend enqueue/completion error retains
    /// all accessed storage; explicitly retire that error before closing/reusing it.
    pub fn linear_into(
        &mut self,
        parameter: ParameterId,
        execution: &mut CudaBf16Gemm,
        input: &mut CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16GemmError> {
        let slot = self
            .parameter_slot(parameter)
            .ok_or(CudaBf16Error::Invalid(
                "logical projection has no active allocation",
            ))?;
        execution.linear_into(&mut self.storage, slot, input, output)
    }
}
