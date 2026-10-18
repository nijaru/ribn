//! Qwen model metadata, state requirements, and weight bindings.
use crate::{QwenConfig, QwenLayerKind};
use engine_core::{
    ConvolutionStateShape, DataType, DeviceId, KvStateSpec, Quantization, RecurrentMatrixShape,
    RecurrentStateSpec, StateRequirement, WeightBinding,
};
use engine_gguf::{GgufError, GgufFile, MetadataValue, TensorDataReader};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Fixed KV page size every Qwen attention layer uses.
///
/// A sequence holds whole pages, so its capacity rounds up to a multiple of this
/// value and growth adds pages. Pages are the addressing unit the backend
/// materializes and the unit a prefix cache would retain.
pub(crate) const KV_PAGE_TOKENS: u32 = 16;

pub(crate) mod config;
use config::read as config_from_gguf;

/// Declare this artifact's continuation components: one full-attention KV family and one
/// recurrent family. These are the model's own maximum shapes; a request materializes and is
/// charged for the capacity it can actually reach.
fn qwen35_state_requirements(
    config: &QwenConfig,
    kv_block_tokens: u32,
) -> Result<Vec<StateRequirement>, GgufError> {
    let language_layers =
        config
            .language_layer_count()
            .ok_or(GgufError::InvalidModelConfiguration(
                "MTP layer count exceeds total block count",
            ))?;
    let full_layers = language_layers / config.full_attention_interval();
    let recurrent_layers = language_layers - full_layers;
    let full_spec = KvStateSpec::with_pages(
        narrow_u16(full_layers, "full-attention layer count")?,
        narrow_u16(config.kv_heads(), "KV head count")?,
        narrow_u16(config.key_length(), "full-attention head dimension")?,
        KV_PAGE_TOKENS,
        kv_block_tokens.div_ceil(KV_PAGE_TOKENS),
        DataType::F16,
    )
    .map_err(|_| {
        GgufError::InvalidModelConfiguration("full-attention state dimensions are invalid")
    })?;
    let matrix = RecurrentMatrixShape::new(
        narrow_u16(config.ssm_time_step_rank(), "recurrent matrix count")?,
        narrow_u16(config.ssm_state_size(), "recurrent matrix row dimension")?,
        narrow_u16(
            config.ssm_inner_size() / config.ssm_time_step_rank(),
            "recurrent matrix column dimension",
        )?,
    )
    .ok_or(GgufError::InvalidModelConfiguration(
        "recurrent matrix dimensions are invalid",
    ))?;
    let convolution_channels = config
        .ssm_inner_size()
        .checked_add(
            config
                .ssm_group_count()
                .checked_mul(config.ssm_state_size())
                .and_then(|channels| channels.checked_mul(2))
                .ok_or(GgufError::InvalidModelConfiguration(
                    "recurrent convolution dimensions overflow",
                ))?,
        )
        .ok_or(GgufError::InvalidModelConfiguration(
            "recurrent convolution dimensions overflow",
        ))?;
    let convolution_history =
        config
            .ssm_conv_kernel()
            .checked_sub(1)
            .ok_or(GgufError::InvalidModelConfiguration(
                "recurrent convolution kernel has no history",
            ))?;
    let recurrent_spec = RecurrentStateSpec::new(
        narrow_u16(recurrent_layers, "recurrent layer count")?,
        matrix,
        ConvolutionStateShape::new(
            convolution_channels,
            narrow_u16(convolution_history, "recurrent convolution history")?,
        )
        .ok_or(GgufError::InvalidModelConfiguration(
            "recurrent convolution dimensions are invalid",
        ))?,
        DataType::F32,
        DataType::F32,
    )
    .map_err(|_| GgufError::InvalidModelConfiguration("recurrent state dimensions are invalid"))?;
    Ok(vec![
        StateRequirement::Recurrent(recurrent_spec),
        StateRequirement::FullAttentionKv(full_spec),
    ])
}

fn narrow_u16(value: u32, name: &'static str) -> Result<u16, GgufError> {
    u16::try_from(value).map_err(|_| GgufError::InvalidModelConfiguration(name))
}

const QWEN35_RECURRENT_LAYER_TENSORS: &[&str] = &[
    "attn_gate.weight",
    "attn_norm.weight",
    "attn_qkv.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "post_attention_norm.weight",
    "ssm_a",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_conv1d.weight",
    "ssm_dt.bias",
    "ssm_norm.weight",
    "ssm_out.weight",
];

const QWEN35_FULL_LAYER_TENSORS: &[&str] = &[
    "attn_k.weight",
    "attn_k_norm.weight",
    "attn_norm.weight",
    "attn_output.weight",
    "attn_q.weight",
    "attn_q_norm.weight",
    "attn_v.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "post_attention_norm.weight",
];

/// A provider for the text-only Qwen3.8 language artifact carried by GGUF.
/// It owns validated format metadata, artifact identity and the model's declared continuation
/// components; tensor execution remains a separate backend/dispatcher responsibility.
/// Opening hashes the complete pinned source in bounded memory to identify its contents.
/// Path replacement cannot change its metadata or readers. The selected file must
/// remain unmodified in place while the provider and tensor readers live.
pub struct QwenGguf {
    file: GgufFile,
    config: QwenConfig,
    /// Complete-artifact hash identifying the exact weights, not a model name.
    artifact_id: String,
    quantization: Quantization,
    state_requirements: Vec<StateRequirement>,
}

impl QwenGguf {
    /// Open and validate a Qwen3.8 GGUF artifact without materializing weights.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] when the artifact, model metadata, tensor ranges or
    /// declared continuation components are invalid.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, GgufError> {
        Self::open_with_kv_block_tokens(path, 16)
    }

    /// Open with an explicit full-attention KV allocation block size.
    /// This is a performance/layout choice, not a model semantic.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] when the block size, artifact, model metadata, tensor
    /// ranges or declared continuation components are invalid.
    pub fn open_with_kv_block_tokens(
        path: impl Into<PathBuf>,
        kv_block_tokens: u32,
    ) -> Result<Self, GgufError> {
        if kv_block_tokens == 0 {
            return Err(GgufError::InvalidModelConfiguration(
                "KV block size must be non-zero",
            ));
        }
        let file = GgufFile::open(path)?;
        let config = config_from_gguf(&file)?;
        for tensor in file.tensors() {
            file.tensor_data_range(tensor.name())?;
        }
        let artifact_id = artifact_identity(file.reader(), file.path())?;
        let quantization = artifact_quantization(&file);
        let state_requirements = qwen35_state_requirements(&config, kv_block_tokens)?;
        Ok(Self {
            file,
            config,
            artifact_id,
            quantization,
            state_requirements,
        })
    }

    /// The complete-artifact identity: `gguf:sha256:<digest>`.
    #[must_use]
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    /// The artifact's mixture recipe, not each tensor's encoding.
    #[must_use]
    pub const fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// Continuation components every request of this artifact must reserve.
    #[must_use]
    pub fn state_requirements(&self) -> &[StateRequirement] {
        &self.state_requirements
    }

    #[must_use]
    pub fn file(&self) -> &GgufFile {
        &self.file
    }

    #[must_use]
    pub const fn config(&self) -> &QwenConfig {
        &self.config
    }

    /// Open one validated tensor as a bounded canonical F32 stream. The
    /// provider still leaves physical allocation and execution to a backend.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] when the tensor is missing, unsupported, or its
    /// encoded range cannot be opened.
    pub fn open_tensor(&self, name: &str) -> Result<TensorDataReader, GgufError> {
        self.file.open_tensor(name)
    }

    /// Build a logical model/device binding for selected tensors without
    /// allocating or uploading their data. The backend materializes those
    /// streams and retains the physical buffers.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] when a requested tensor cannot be opened or the
    /// names contain a duplicate.
    pub fn weight_binding(
        &self,
        device: DeviceId,
        names: &[&str],
    ) -> Result<WeightBinding, GgufError> {
        let specs = names
            .iter()
            .map(|name| self.open_tensor(name).map(|reader| reader.spec().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        WeightBinding::new(device, specs)
            .map_err(|error| GgufError::InvalidWeightBinding(error.to_string()))
    }

    /// Return the execution kind for one language layer. MTP blocks are not
    /// included in this accessor and are intentionally a separate capability.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError::InvalidModelConfiguration`] when `layer` is not a
    /// language-layer index.
    pub fn layer_kind(&self, layer: u32) -> Result<QwenLayerKind, GgufError> {
        self.config
            .layer_kind(layer)
            .map_err(|error| GgufError::InvalidModelConfiguration(error.reason()))
    }

    /// Build a logical binding for all checkpoint tensors used by one language
    /// layer. This validates names and descriptors without allocating weight
    /// buffers or claiming that the layer is executable.
    ///
    /// # Errors
    ///
    /// Returns [`GgufError`] when the layer index or any required tensor is
    /// invalid or missing.
    pub fn layer_weight_binding(
        &self,
        device: DeviceId,
        layer: u32,
    ) -> Result<WeightBinding, GgufError> {
        let kind = self.layer_kind(layer)?;
        let suffixes = match kind {
            QwenLayerKind::Recurrent => QWEN35_RECURRENT_LAYER_TENSORS,
            QwenLayerKind::FullAttention => QWEN35_FULL_LAYER_TENSORS,
        };
        let names = suffixes
            .iter()
            .map(|suffix| format!("blk.{layer}.{suffix}"))
            .collect::<Vec<_>>();
        let specs = names
            .iter()
            .map(|name| self.open_tensor(name).map(|reader| reader.spec().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        WeightBinding::new(device, specs)
            .map_err(|error| GgufError::InvalidWeightBinding(error.to_string()))
    }
}

/// Hash the complete artifact in bounded memory, including actual weight bytes.
/// This deliberately pays a startup read instead of treating a model family or
/// file name as proof that two prepared artifacts contain the same weights.
fn artifact_identity(mut reader: impl Read, path: &Path) -> Result<String, GgufError> {
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer).map_err(|error| GgufError::Io {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let hash = digest
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut text, byte| {
            use std::fmt::Write;
            write!(text, "{byte:02x}").expect("writing to String cannot fail");
            text
        });
    Ok(format!("gguf:sha256:{hash}"))
}

fn artifact_quantization(file: &GgufFile) -> Quantization {
    // GGUF file_type identifies the mixture recipe, not each tensor's encoding.
    // Unknown recipes remain explicit; per-tensor encodings are retained intact.
    quantization_for_recipe(
        file.metadata("general.file_type")
            .and_then(MetadataValue::as_u64),
    )
}

fn quantization_for_recipe(recipe: Option<u64>) -> Quantization {
    match recipe {
        Some(0 | 1 | 32) => Quantization::None,
        Some(15) => Quantization::GgufQ4Km,
        _ => Quantization::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn artifact_identity_tracks_bytes_instead_of_model_names() {
        let path =
            std::env::temp_dir().join(format!("engine-artifact-identity-{}", std::process::id()));
        let first = artifact_identity(b"abc".as_slice(), &path).unwrap();
        assert_eq!(
            first.as_str(),
            "gguf:sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_ne!(first, artifact_identity(b"abd".as_slice(), &path).unwrap());
    }

    #[test]
    fn geometry_tokenizer_template_and_hash_keep_the_selected_source() {
        use config::tests::{encode, metadata};
        let path =
            std::env::temp_dir().join(format!("engine-qwen-{}-source.gguf", std::process::id()));
        let mut metadata = metadata();
        for (key, value) in [
            ("tokenizer.ggml.model", MetadataValue::String("gpt2".into())),
            ("tokenizer.ggml.pre", MetadataValue::String("qwen35".into())),
            (
                "tokenizer.chat_template",
                MetadataValue::String("original template".into()),
            ),
            (
                "tokenizer.ggml.tokens",
                MetadataValue::Array(
                    ["a", "b", "ab"]
                        .map(|value| MetadataValue::String(value.into()))
                        .to_vec(),
                ),
            ),
            (
                "tokenizer.ggml.merges",
                MetadataValue::Array(vec![MetadataValue::String("a b".into())]),
            ),
            (
                "tokenizer.ggml.token_type",
                MetadataValue::Array(vec![MetadataValue::I32(1); 3]),
            ),
            ("tokenizer.ggml.bos_token_id", MetadataValue::U32(1)),
            ("tokenizer.ggml.eos_token_id", MetadataValue::U32(2)),
        ] {
            metadata.insert(key.into(), value);
        }
        let original = encode(&metadata);
        std::fs::write(&path, &original).unwrap();
        let selected = QwenGguf::open(&path).unwrap();
        let expected_id = artifact_identity(original.as_slice(), &path).unwrap();
        metadata.insert("qwen35.context_length".into(), MetadataValue::U64(4096));
        metadata.insert(
            "tokenizer.chat_template".into(),
            MetadataValue::String("replacement template".into()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            MetadataValue::Array(
                ["x", "y", "xy"]
                    .map(|value| MetadataValue::String(value.into()))
                    .to_vec(),
            ),
        );
        metadata.insert(
            "tokenizer.ggml.merges".into(),
            MetadataValue::Array(vec![MetadataValue::String("x y".into())]),
        );
        let incoming = path.with_extension("incoming");
        std::fs::write(&incoming, encode(&metadata)).unwrap();
        std::fs::rename(incoming, &path).unwrap();
        let replacement = QwenGguf::open(&path).unwrap();
        assert_eq!(replacement.config().context_length(), 4096);
        assert_ne!(replacement.artifact_id(), selected.artifact_id());
        assert_eq!(selected.config().context_length(), 262_144);
        assert_eq!(selected.artifact_id(), expected_id);
        let tokenizer = selected.file().tokenizer().unwrap();
        assert_eq!(tokenizer.tokens(), &["a", "b", "ab"]);
        assert_eq!(
            selected
                .file()
                .metadata("tokenizer.chat_template")
                .and_then(MetadataValue::as_str),
            Some("original template")
        );
        assert_eq!(
            artifact_identity(selected.file().reader(), &path).unwrap(),
            expected_id
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn quantization_comes_from_artifact_recipe_metadata() {
        assert_eq!(quantization_for_recipe(None), Quantization::Other);
        for (recipe, expected) in [
            (15, Quantization::GgufQ4Km),
            (0, Quantization::None),
            (7, Quantization::Other),
        ] {
            assert_eq!(quantization_for_recipe(Some(recipe)), expected);
        }
    }

    #[test]
    fn declares_distinct_continuation_state_families() {
        let mut config = QwenConfig {
            context_length: 262_144,
            embedding_length: 5120,
            feed_forward_length: 17_408,
            block_count: 65,
            attention_heads: 24,
            kv_heads: 4,
            key_length: 256,
            value_length: 256,
            rms_norm_epsilon: 1.0e-6,
            rope_dimensions: 64,
            rope_frequency_base: 1.0e7,
            full_attention_interval: 4,
            ssm_group_count: 16,
            ssm_inner_size: 6144,
            ssm_state_size: 128,
            ssm_time_step_rank: 48,
            ssm_conv_kernel: 4,
            nextn_predict_layers: 1,
        };
        let requirements = qwen35_state_requirements(&config, 16).expect("state requirements");
        assert_eq!(requirements.len(), 2);
        let recurrent = requirements
            .iter()
            .find_map(|requirement| match requirement {
                StateRequirement::Recurrent(spec) => Some(*spec),
                StateRequirement::FullAttentionKv(_) => None,
            })
            .expect("recurrent state");
        assert_eq!(recurrent.matrix().matrix_count(), 48);
        assert_eq!(recurrent.matrix().rows(), 128);
        assert_eq!(recurrent.matrix().columns(), 128);
        assert_eq!(recurrent.convolution().channels(), 10_240);
        assert_eq!(recurrent.convolution().history_tokens(), 3);
        assert!(
            requirements
                .iter()
                .any(|requirement| matches!(requirement, StateRequirement::FullAttentionKv(_)))
        );
        // A concrete request re-derives each component at the capacity it can reach.
        assert!(
            requirements
                .iter()
                .all(|requirement| requirement.for_tokens(1).is_ok())
        );
        // Valid head geometry fits the state representation, but doubling the
        // group product for convolution channels must fail without panic.
        config.ssm_group_count = u32::from(u16::MAX);
        config.ssm_time_step_rank = u32::from(u16::MAX);
        config.ssm_state_size = u32::from(u16::MAX);
        config.ssm_inner_size = u32::from(u16::MAX).pow(2);
        config.validate().unwrap();
        assert!(matches!(
            qwen35_state_requirements(&config, 16),
            Err(GgufError::InvalidModelConfiguration(
                "recurrent convolution dimensions overflow"
            ))
        ));
    }
}
