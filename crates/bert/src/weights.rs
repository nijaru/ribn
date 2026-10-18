//! Host-side parameter loading and shape validation.
//!
//! The encoder declares which parameters a BERT checkpoint must carry and what
//! shape each one has. Resolving the package and reading tensor bytes belongs to
//! [`ribn_hf`]; interpreting those bytes as this model's parameters belongs here.

use std::fmt;
use std::io::{self, Read};

use ribn_hf::{LocalModelPackage, LocalWeightSet, PackageError};

use crate::config::{BertConfig, ConfigError};

/// One dense projection: `weight` is `[output, input]` in row-major order, matching
/// the checkpoint's `torch.nn.Linear` layout, and `bias` has one value per output.
#[derive(Clone, Debug, PartialEq)]
pub struct Dense {
    pub input: usize,
    pub output: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// One normalization layer's affine parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// Every parameter of one encoder layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub query: Dense,
    pub key: Dense,
    pub value: Dense,
    pub attention_output: Dense,
    pub attention_norm: LayerNorm,
    pub intermediate: Dense,
    pub output: Dense,
    pub output_norm: LayerNorm,
}

/// A complete encoder's parameters on the host.
#[derive(Clone, Debug, PartialEq)]
pub struct HostWeights {
    pub word_embeddings: Vec<f32>,
    pub position_embeddings: Vec<f32>,
    pub token_type_embeddings: Vec<f32>,
    pub embedding_norm: LayerNorm,
    pub layers: Vec<Layer>,
    pub pooler: Dense,
}

impl HostWeights {
    /// Read every parameter this encoder needs from a resolved local package.
    ///
    /// Decoded parameters move into the returned owner once. The reader retains
    /// at most one source shard; arbitrary shard layouts may require rereads.
    /// This is not a byte bound or streaming device upload.
    ///
    /// # Errors
    /// Returns [`WeightError`] for a missing parameter, a wrong shape or dtype, or
    /// a package resolution failure.
    pub fn load(package: &LocalModelPackage, config: &BertConfig) -> Result<Self, WeightError> {
        config.validate()?;
        let hidden = config.hidden_size;
        let mut reader = Reader::new(package)?;
        let weights = Self {
            word_embeddings: reader.tensor(
                "embeddings.word_embeddings.weight",
                &[config.vocab_size, hidden],
            )?,
            position_embeddings: reader.tensor(
                "embeddings.position_embeddings.weight",
                &[config.max_position_embeddings, hidden],
            )?,
            token_type_embeddings: reader.tensor(
                "embeddings.token_type_embeddings.weight",
                &[config.type_vocab_size, hidden],
            )?,
            embedding_norm: LayerNorm {
                weight: reader.tensor("embeddings.LayerNorm.weight", &[hidden])?,
                bias: reader.tensor("embeddings.LayerNorm.bias", &[hidden])?,
            },
            layers: (0..config.num_hidden_layers)
                .map(|layer| {
                    let prefix = format!("encoder.layer.{layer}");
                    Ok(Layer {
                        query: reader.dense(
                            &format!("{prefix}.attention.self.query"),
                            hidden,
                            hidden,
                        )?,
                        key: reader.dense(
                            &format!("{prefix}.attention.self.key"),
                            hidden,
                            hidden,
                        )?,
                        value: reader.dense(
                            &format!("{prefix}.attention.self.value"),
                            hidden,
                            hidden,
                        )?,
                        attention_output: reader.dense(
                            &format!("{prefix}.attention.output.dense"),
                            hidden,
                            hidden,
                        )?,
                        attention_norm: LayerNorm {
                            weight: reader.tensor(
                                &format!("{prefix}.attention.output.LayerNorm.weight"),
                                &[hidden],
                            )?,
                            bias: reader.tensor(
                                &format!("{prefix}.attention.output.LayerNorm.bias"),
                                &[hidden],
                            )?,
                        },
                        intermediate: reader.dense(
                            &format!("{prefix}.intermediate.dense"),
                            hidden,
                            config.intermediate_size,
                        )?,
                        output: reader.dense(
                            &format!("{prefix}.output.dense"),
                            config.intermediate_size,
                            hidden,
                        )?,
                        output_norm: LayerNorm {
                            weight: reader
                                .tensor(&format!("{prefix}.output.LayerNorm.weight"), &[hidden])?,
                            bias: reader
                                .tensor(&format!("{prefix}.output.LayerNorm.bias"), &[hidden])?,
                        },
                    })
                })
                .collect::<Result<Vec<_>, WeightError>>()?,
            pooler: reader.dense("pooler.dense", hidden, hidden)?,
        };
        weights.validate(config)?;
        Ok(weights)
    }

    /// Check all parameter geometry before backend materialization.
    ///
    /// Host parameters are mutable, so artifact-time validation alone cannot
    /// justify unchecked device reads when a caller supplies them directly.
    ///
    /// # Errors
    /// Rejects invalid configuration, dimensions, layer counts or buffer lengths.
    pub fn validate(&self, config: &BertConfig) -> Result<(), WeightError> {
        config.validate()?;
        let hidden = config.hidden_size;
        validate_length(
            "embeddings.word_embeddings.weight",
            &self.word_embeddings,
            &[config.vocab_size, hidden],
        )?;
        validate_length(
            "embeddings.position_embeddings.weight",
            &self.position_embeddings,
            &[config.max_position_embeddings, hidden],
        )?;
        validate_length(
            "embeddings.token_type_embeddings.weight",
            &self.token_type_embeddings,
            &[config.type_vocab_size, hidden],
        )?;
        self.embedding_norm
            .validate("embeddings.LayerNorm", hidden)?;
        if self.layers.len() != config.num_hidden_layers {
            return Err(WeightError::Length {
                parameter: "encoder.layers".to_owned(),
                expected: config.num_hidden_layers,
                found: self.layers.len(),
            });
        }
        for (index, layer) in self.layers.iter().enumerate() {
            let prefix = format!("encoder.layer.{index}");
            for (name, dense) in [
                ("attention.self.query", &layer.query),
                ("attention.self.key", &layer.key),
                ("attention.self.value", &layer.value),
                ("attention.output.dense", &layer.attention_output),
            ] {
                dense.validate(&format!("{prefix}.{name}"), hidden, hidden)?;
            }
            layer
                .attention_norm
                .validate(&format!("{prefix}.attention.output.LayerNorm"), hidden)?;
            layer.intermediate.validate(
                &format!("{prefix}.intermediate.dense"),
                hidden,
                config.intermediate_size,
            )?;
            layer.output.validate(
                &format!("{prefix}.output.dense"),
                config.intermediate_size,
                hidden,
            )?;
            layer
                .output_norm
                .validate(&format!("{prefix}.output.LayerNorm"), hidden)?;
        }
        self.pooler.validate("pooler.dense", hidden, hidden)
    }
}

fn element_count(parameter: &str, shape: &[usize]) -> Result<usize, WeightError> {
    shape
        .iter()
        .try_fold(1_usize, |count, &dimension| count.checked_mul(dimension))
        .ok_or_else(|| WeightError::SizeOverflow {
            parameter: parameter.to_owned(),
        })
}

fn validate_length(parameter: &str, values: &[f32], shape: &[usize]) -> Result<(), WeightError> {
    let expected = element_count(parameter, shape)?;
    if values.len() != expected {
        return Err(WeightError::Length {
            parameter: parameter.to_owned(),
            expected,
            found: values.len(),
        });
    }
    Ok(())
}

impl LayerNorm {
    fn validate(&self, prefix: &str, hidden: usize) -> Result<(), WeightError> {
        validate_length(&format!("{prefix}.weight"), &self.weight, &[hidden])?;
        validate_length(&format!("{prefix}.bias"), &self.bias, &[hidden])
    }
}

impl Dense {
    fn validate(&self, prefix: &str, input: usize, output: usize) -> Result<(), WeightError> {
        if self.input != input || self.output != output {
            return Err(WeightError::Shape {
                parameter: format!("{prefix}.weight"),
                expected: vec![output, input],
                found: vec![self.output, self.input],
            });
        }
        validate_length(&format!("{prefix}.weight"), &self.weight, &[output, input])?;
        validate_length(&format!("{prefix}.bias"), &self.bias, &[output])
    }

    /// Flattened row-major offset of `weight[row][column]`.
    #[must_use]
    pub const fn offset(&self, row: usize, column: usize) -> usize {
        row * self.input + column
    }
}

/// Why a checkpoint cannot supply this encoder's parameters.
#[derive(Debug)]
pub enum WeightError {
    Config(ConfigError),
    Package(PackageError),
    Read {
        parameter: String,
        source: io::Error,
    },
    SizeOverflow {
        parameter: String,
    },
    Missing(String),
    Shape {
        parameter: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    Dtype {
        parameter: String,
    },
    Length {
        parameter: String,
        expected: usize,
        found: usize,
    },
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(error) => write!(f, "invalid parameter geometry: {error}"),
            Self::Package(error) => write!(f, "package resolution failed: {error}"),
            Self::Read { parameter, source } => {
                write!(f, "could not read parameter {parameter}: {source}")
            }
            Self::SizeOverflow { parameter } => write!(
                f,
                "parameter {parameter} dimensions overflow the address space"
            ),
            Self::Missing(parameter) => {
                write!(f, "checkpoint is missing required parameter {parameter}")
            }
            Self::Shape {
                parameter,
                expected,
                found,
            } => write!(
                f,
                "parameter {parameter} has shape {found:?}, expected {expected:?}"
            ),
            Self::Dtype { parameter } => write!(
                f,
                "parameter {parameter} is not F32; this encoder executes F32 checkpoints"
            ),
            Self::Length {
                parameter,
                expected,
                found,
            } => write!(
                f,
                "parameter {parameter} carries {found} values, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for WeightError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Config(error) => Some(error),
            Self::Package(error) => Some(error),
            Self::Read { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<ConfigError> for WeightError {
    fn from(error: ConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<PackageError> for WeightError {
    fn from(error: PackageError) -> Self {
        Self::Package(error)
    }
}

struct Reader<'a> {
    weights: LocalWeightSet<'a>,
}

impl<'a> Reader<'a> {
    fn new(package: &'a LocalModelPackage) -> Result<Self, WeightError> {
        Ok(Self {
            weights: package.weight_set_resident(1)?,
        })
    }

    fn dense(&mut self, prefix: &str, input: usize, output: usize) -> Result<Dense, WeightError> {
        Ok(Dense {
            input,
            output,
            weight: self.tensor(&format!("{prefix}.weight"), &[output, input])?,
            bias: self.tensor(&format!("{prefix}.bias"), &[output])?,
        })
    }

    fn tensor(&mut self, parameter: &str, expected: &[usize]) -> Result<Vec<f32>, WeightError> {
        let tensor = self
            .weights
            .tensor(parameter)
            .map_err(|error| match error {
                PackageError::UnknownParameter(_) => WeightError::Missing(parameter.to_owned()),
                other => WeightError::Package(other),
            })?;
        if tensor.scalar_type() != &ribn_foundation::ScalarType::F32 {
            return Err(WeightError::Dtype {
                parameter: parameter.to_owned(),
            });
        }
        if tensor.shape() != expected {
            return Err(WeightError::Shape {
                parameter: parameter.to_owned(),
                expected: expected.to_vec(),
                found: tensor.shape().to_vec(),
            });
        }
        let count = element_count(parameter, expected)?;
        let mut values = Vec::with_capacity(count);
        let mut reader = tensor.reader();
        // Decode directly into the final parameter owner. Only this fixed byte
        // buffer overlaps the output; no encoded tensor/shard cache is retained.
        let mut buffer = [0_u8; 16 * 1024];
        while values.len() < count {
            let bytes = (count - values.len()).min(buffer.len() / 4) * 4;
            reader
                .read_exact(&mut buffer[..bytes])
                .map_err(|source| WeightError::Read {
                    parameter: parameter.to_owned(),
                    source,
                })?;
            values.extend(
                buffer[..bytes]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk)),
            );
        }
        validate_length(parameter, &values, expected)?;
        Ok(values)
    }
}
