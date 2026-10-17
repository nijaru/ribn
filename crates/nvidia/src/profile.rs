//! Declared execution geometry and lane capacity for the specialized Qwen CUDA path.
//!
//! The current kernels are compiled against fixed dimensions, so this profile is the
//! backend's declaration of the single geometry it can execute. Model loading compares an
//! artifact's declared geometry against [`QwenCudaProfile::SUPPORTED`] before tokenizer or
//! device access; supporting a second geometry is then a new profile plus its kernel
//! specialization, not another copy of the comparison.
//!
//! Every value mirrors a compiled constant. The unit tests fail if they drift, so a kernel
//! change cannot silently leave the accepted geometry stale.

use std::fmt;

use crate::qwen_reference::{
    ATTN_HEAD_DIM, ATTN_KV_HEADS, ATTN_Q_HEADS, ATTN_ROPE_BASE, ATTN_ROT_DIMS, GDN_D_CONV,
    GDN_HEAD_DIM, GDN_INNER, GDN_K_HEADS, GDN_V_HEADS, N_EMBD, N_FF,
};

/// Maximum row count the weights-read-once CUDA kernels execute in one step.
///
/// This is a compiled capability, not a tuning choice: kernel-side validation and the
/// model's default lane both derive from it. Moving it requires re-qualifying the lane.
pub const MAX_BATCH_MEMBERS: usize = 8;

/// Geometry and numerical parameters a Qwen-family artifact declares for CUDA execution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QwenCudaProfile {
    pub embedding_length: u64,
    pub feed_forward_length: u64,
    pub attention_heads: u64,
    pub kv_heads: u64,
    pub key_length: u64,
    pub value_length: u64,
    pub rope_dimensions: u64,
    /// Compared by bit pattern: the executor applies this value verbatim.
    pub rope_frequency_base: f32,
    /// Compared by bit pattern: the executor applies this value verbatim.
    pub rms_norm_epsilon: f32,
    pub ssm_group_count: u64,
    pub ssm_inner_size: u64,
    pub ssm_state_size: u64,
    pub ssm_time_step_rank: u64,
    pub ssm_conv_kernel: u64,
}

impl QwenCudaProfile {
    /// The one geometry the compiled Qwen CUDA kernels execute.
    pub const SUPPORTED: Self = Self {
        embedding_length: N_EMBD as u64,
        feed_forward_length: N_FF as u64,
        attention_heads: ATTN_Q_HEADS as u64,
        kv_heads: ATTN_KV_HEADS as u64,
        key_length: ATTN_HEAD_DIM as u64,
        value_length: ATTN_HEAD_DIM as u64,
        rope_dimensions: ATTN_ROT_DIMS as u64,
        rope_frequency_base: ATTN_ROPE_BASE,
        rms_norm_epsilon: 1.0e-6,
        ssm_group_count: GDN_K_HEADS as u64,
        ssm_inner_size: GDN_INNER as u64,
        ssm_state_size: GDN_HEAD_DIM as u64,
        ssm_time_step_rank: GDN_V_HEADS as u64,
        ssm_conv_kernel: GDN_D_CONV as u64,
    };

    /// The first field where `model` disagrees with this profile, in declaration order.
    ///
    /// Comparing whole profiles keeps the accepted geometry in one place: callers describe
    /// what the artifact declares and receive the precise disagreement, if any.
    #[must_use]
    pub fn mismatch(&self, model: &Self) -> Option<ProfileMismatch> {
        for (field, model, supported) in [
            (
                "embedding_length",
                model.embedding_length,
                self.embedding_length,
            ),
            (
                "feed_forward_length",
                model.feed_forward_length,
                self.feed_forward_length,
            ),
            (
                "attention_heads",
                model.attention_heads,
                self.attention_heads,
            ),
            ("kv_heads", model.kv_heads, self.kv_heads),
            ("key_length", model.key_length, self.key_length),
            ("value_length", model.value_length, self.value_length),
            (
                "rope_dimensions",
                model.rope_dimensions,
                self.rope_dimensions,
            ),
            (
                "ssm_group_count",
                model.ssm_group_count,
                self.ssm_group_count,
            ),
            ("ssm_inner_size", model.ssm_inner_size, self.ssm_inner_size),
            ("ssm_state_size", model.ssm_state_size, self.ssm_state_size),
            (
                "ssm_time_step_rank",
                model.ssm_time_step_rank,
                self.ssm_time_step_rank,
            ),
            (
                "ssm_conv_kernel",
                model.ssm_conv_kernel,
                self.ssm_conv_kernel,
            ),
        ] {
            if model != supported {
                return Some(ProfileMismatch::Length {
                    field,
                    model,
                    supported,
                });
            }
        }
        for (field, model, supported) in [
            (
                "rms_norm_epsilon",
                model.rms_norm_epsilon,
                self.rms_norm_epsilon,
            ),
            (
                "rope_frequency_base",
                model.rope_frequency_base,
                self.rope_frequency_base,
            ),
        ] {
            if model.to_bits() != supported.to_bits() {
                return Some(ProfileMismatch::Float {
                    field,
                    model,
                    supported,
                });
            }
        }
        None
    }
}

/// One disagreement between an artifact's declared geometry and the executable profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProfileMismatch {
    Length {
        field: &'static str,
        model: u64,
        supported: u64,
    },
    Float {
        field: &'static str,
        model: f32,
        supported: f32,
    },
}

impl fmt::Display for ProfileMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Length {
                field,
                model,
                supported,
            } => write!(
                f,
                "CUDA Qwen does not support {field} {model}; requires {supported}"
            ),
            Self::Float {
                field,
                model,
                supported,
            } => write!(
                f,
                "CUDA Qwen does not support {field} {model}; requires {supported}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> QwenCudaProfile {
        QwenCudaProfile::SUPPORTED
    }

    #[test]
    fn supported_profile_mirrors_the_compiled_kernel_geometry() {
        let profile = QwenCudaProfile::SUPPORTED;
        assert_eq!(profile.embedding_length, N_EMBD as u64);
        assert_eq!(profile.feed_forward_length, N_FF as u64);
        assert_eq!(profile.attention_heads, ATTN_Q_HEADS as u64);
        assert_eq!(profile.kv_heads, ATTN_KV_HEADS as u64);
        assert_eq!(profile.key_length, ATTN_HEAD_DIM as u64);
        assert_eq!(profile.value_length, ATTN_HEAD_DIM as u64);
        assert_eq!(profile.rope_dimensions, ATTN_ROT_DIMS as u64);
        assert_eq!(
            profile.rope_frequency_base.to_bits(),
            ATTN_ROPE_BASE.to_bits()
        );
        assert_eq!(profile.rms_norm_epsilon.to_bits(), 1.0e-6_f32.to_bits());
        assert_eq!(profile.ssm_group_count, GDN_K_HEADS as u64);
        assert_eq!(profile.ssm_inner_size, GDN_INNER as u64);
        assert_eq!(profile.ssm_state_size, GDN_HEAD_DIM as u64);
        assert_eq!(profile.ssm_time_step_rank, GDN_V_HEADS as u64);
        assert_eq!(profile.ssm_conv_kernel, GDN_D_CONV as u64);
    }

    #[test]
    fn identical_profiles_agree() {
        assert_eq!(model().mismatch(&model()), None);
    }

    #[test]
    fn length_mismatch_names_the_field_and_both_values() {
        let declared = QwenCudaProfile {
            embedding_length: 4096,
            ..model()
        };
        let mismatch = QwenCudaProfile::SUPPORTED.mismatch(&declared).unwrap();
        assert_eq!(
            mismatch,
            ProfileMismatch::Length {
                field: "embedding_length",
                model: 4096,
                supported: 5120,
            }
        );
        assert_eq!(
            mismatch.to_string(),
            "CUDA Qwen does not support embedding_length 4096; requires 5120"
        );
    }

    #[test]
    fn float_mismatch_uses_exact_bits() {
        let declared = QwenCudaProfile {
            rms_norm_epsilon: 1.0e-5,
            ..model()
        };
        assert_eq!(
            QwenCudaProfile::SUPPORTED.mismatch(&declared).unwrap(),
            ProfileMismatch::Float {
                field: "rms_norm_epsilon",
                model: 1.0e-5,
                supported: 1.0e-6,
            }
        );
        // One bit apart is still a mismatch: the executor applies the value verbatim.
        let nearly = QwenCudaProfile {
            rope_frequency_base: f32::from_bits(ATTN_ROPE_BASE.to_bits() + 1),
            ..model()
        };
        assert!(matches!(
            QwenCudaProfile::SUPPORTED.mismatch(&nearly),
            Some(ProfileMismatch::Float {
                field: "rope_frequency_base",
                ..
            })
        ));
    }

    #[test]
    fn the_first_disagreement_is_reported() {
        let declared = QwenCudaProfile {
            feed_forward_length: 1,
            ssm_conv_kernel: 2,
            ..model()
        };
        assert!(matches!(
            QwenCudaProfile::SUPPORTED.mismatch(&declared),
            Some(ProfileMismatch::Length {
                field: "feed_forward_length",
                ..
            })
        ));
    }
}
