//! Execution vocabulary shared by the Qwen model and its NVIDIA backend.
//!
//! Request scheduling and serving lifecycle belong to `ribn`, not this crate. The remaining
//! contracts are the prepared batch a backend consumes, continuation state declarations and
//! accounting, and artifact weight descriptions. They are not the general inference API.

pub mod backend;
pub mod device;
pub mod execution;
pub mod request;
pub mod state;
pub mod tensor;
pub mod weights;

pub use backend::{BackendError, BackendSubmissionId, ComputeBackend};
pub use device::DeviceId;
pub use execution::{
    BatchError, ExecutionBatch, ExecutionBatchEvent, ExecutionEvent, ExecutionPhase,
    ExecutionSegment, ExecutionTokenInput,
};
pub use request::{RequestId, SamplingError, SamplingParams};
pub use state::{
    ConvolutionStateShape, InferenceState, InferenceStateSet, KvState, KvStateSpec,
    LogicalStateManager, RecurrentMatrixShape, RecurrentState, RecurrentStateSpec, StateError,
    StateHandle, StateId, StateRequirement, StateSpecError,
};
pub use tensor::{DataType, Quantization, WeightFormat};
pub use weights::{
    F32BlockStream, WeightBinding, WeightBindingError, WeightSpecError, WeightTensorSpec,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn state_requirements() -> Vec<StateRequirement> {
        let kv = KvStateSpec::new(16, 4, 256, 16, DataType::F16).expect("valid KV spec");
        let matrix = RecurrentMatrixShape::new(48, 128, 128).expect("valid matrix shape");
        let convolution =
            ConvolutionStateShape::new(10_240, 3).expect("valid convolution state shape");
        let recurrent =
            RecurrentStateSpec::new(48, matrix, convolution, DataType::F16, DataType::F32)
                .expect("valid recurrent spec");
        vec![
            StateRequirement::Recurrent(recurrent),
            StateRequirement::FullAttentionKv(kv),
        ]
    }

    #[test]
    fn a_batch_rejects_duplicate_request_work() {
        let requirements: std::sync::Arc<[StateRequirement]> = state_requirements().into();
        let request = RequestId::new(1).expect("valid request ID");
        let segment = |phase, tokens| {
            ExecutionSegment::new_shared(
                request,
                phase,
                1,
                tokens,
                0,
                std::sync::Arc::clone(&requirements),
            )
            .expect("valid segment")
        };
        assert_eq!(
            ExecutionBatch::new(vec![
                segment(ExecutionPhase::Decode, 1),
                segment(ExecutionPhase::Decode, 1)
            ]),
            Err(BatchError::DuplicateBatchRequest)
        );
        assert_eq!(ExecutionBatch::new(Vec::new()), Err(BatchError::EmptyBatch));
        let batch = ExecutionBatch::new(vec![segment(ExecutionPhase::Decode, 1)]).expect("batch");
        assert_eq!(batch.total_tokens(), Some(1));
    }

    #[test]
    fn token_input_must_match_the_segment_phase_and_length() {
        let prompt: std::sync::Arc<[u32]> = std::sync::Arc::from([1_u32, 2, 3]);
        let decode = ExecutionTokenInput::decode(7);
        assert_eq!(decode.decode_token(), Some(7));
        assert_eq!(
            ExecutionTokenInput::prompt(prompt.clone(), 1, 2)
                .expect("slice")
                .prompt_slice(),
            Some(&[2, 3][..])
        );
        assert_eq!(
            ExecutionTokenInput::prompt(prompt.clone(), 2, 2),
            Err(BatchError::InvalidTokenInput)
        );
        let segment = ExecutionSegment::new_shared(
            RequestId::new(2).expect("valid request ID"),
            ExecutionPhase::Prefill,
            1,
            2,
            0,
            state_requirements().into(),
        )
        .expect("valid segment");
        assert_eq!(
            segment.with_token_input(decode),
            Err(BatchError::InvalidTokenInput)
        );
    }

    #[test]
    fn typed_state_handles_reject_cross_family_construction() {
        let spec = KvStateSpec::new(16, 4, 256, 16, DataType::F16).expect("valid KV spec");
        let requirement = StateRequirement::FullAttentionKv(spec);
        let handle = StateHandle::new(
            StateId::new(1).expect("valid state ID"),
            requirement,
            DeviceId::new(0),
            32,
        );
        assert!(KvState::new(handle, spec).is_some());

        let matrix = RecurrentMatrixShape::new(48, 128, 128).expect("valid matrix shape");
        let convolution =
            ConvolutionStateShape::new(10_240, 3).expect("valid convolution state shape");
        let recurrent =
            RecurrentStateSpec::new(48, matrix, convolution, DataType::F16, DataType::F32)
                .expect("valid recurrent spec");
        let wrong_family_handle = StateHandle::new(
            StateId::new(2).expect("valid state ID"),
            requirement,
            DeviceId::new(0),
            32,
        );
        assert!(RecurrentState::new(wrong_family_handle, recurrent).is_none());
    }

    #[test]
    fn logical_state_manager_accounts_for_typed_allocations() {
        let device = DeviceId::new(0);
        let spec = KvStateSpec::new(1, 1, 2, 4, DataType::F16).expect("KV spec");
        assert_eq!(spec.byte_size(), Some(32));
        let mut manager =
            LogicalStateManager::new(device, ribn_foundation::BytePool::new(32).shared());
        let state = manager.allocate_kv(spec).expect("first allocation");
        assert_eq!(manager.used_bytes(), 32);
        assert!(matches!(
            manager.allocate_kv(spec),
            Err(StateError::CapacityExceeded { .. })
        ));

        let state_set = InferenceStateSet::try_new(Some(state), None).expect("state set");
        let mut committed = state_set;
        manager.commit(&mut committed, 4).expect("commit");
        assert_eq!(committed.token_position(), Some(4));
        manager
            .release(committed.kv().expect("KV state").handle())
            .expect("release");
        assert_eq!(manager.used_bytes(), 0);
    }
}
