//! Request shape, request-local infeasibility and concrete-shape selection.
//!
//! Everything here is host-computable and independent of a CUDA context: which
//! shapes this encoder can execute at all, how many device bytes one request holds
//! from submission until release, and which prefix of a candidate set it accepts.
//! Device execution ([`crate::CudaBertEncoder`]) consumes these decisions instead of
//! re-deriving them, so the accepted range and the byte envelope the shared pool is
//! asked to cover come from one concrete shape rather than two independent budgets.

use std::num::NonZeroUsize;

use ribn_batch::BatchSelection;

use crate::config::{BertConfig, ConfigError};

/// One encoder request: token ids, their segment ids and a per-position key mask.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncoderRequest {
    pub token_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    /// One flag per position; a zero flag removes that position as a key.
    pub attention_mask: Vec<i32>,
}

impl EncoderRequest {
    /// A single-segment request whose every position is attendable.
    #[must_use]
    pub fn single_segment(token_ids: Vec<u32>) -> Self {
        Self {
            token_type_ids: vec![0; token_ids.len()],
            attention_mask: vec![1; token_ids.len()],
            token_ids,
        }
    }

    /// Positions this request occupies.
    #[must_use]
    pub fn sequence(&self) -> usize {
        self.token_ids.len()
    }
}

/// Why this encoder can never execute a request, whatever another owner releases.
///
/// These are request-local defects, not backpressure: releasing device bytes or
/// draining a consumer cannot make them executable. They are reported as
/// [`BatchSelection::Rejected`] and delivered without waiting.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EncoderConstraint {
    /// A request with no positions has no shape to execute.
    EmptySequence,
    /// The request is longer than the model's position embeddings.
    SequenceTooLong {
        sequence: usize,
        max_position_embeddings: usize,
    },
    /// The model permits the length, but this backend cannot execute it.
    BackendSequenceLimit {
        sequence: usize,
        max_sequence: usize,
    },
    /// Token, token-type and mask lengths do not agree.
    LengthMismatch {
        tokens: usize,
        token_types: usize,
        mask: usize,
    },
    /// A token id is outside the model's vocabulary.
    UnknownToken { token: u32, vocabulary: usize },
    /// A token-type id is outside the model's type vocabulary.
    UnknownTokenType { kind: u32, type_vocabulary: usize },
    /// The request's element count does not fit an addressable byte envelope.
    ShapeOverflow { sequence: usize },
}

impl std::fmt::Display for EncoderConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySequence => f.write_str("sequence length must be positive"),
            Self::SequenceTooLong {
                sequence,
                max_position_embeddings,
            } => write!(
                f,
                "sequence length {sequence} exceeds the model's {max_position_embeddings} \
                 position embeddings"
            ),
            Self::BackendSequenceLimit {
                sequence,
                max_sequence,
            } => write!(
                f,
                "sequence length {sequence} exceeds the backend's {max_sequence}-token limit"
            ),
            Self::LengthMismatch {
                tokens,
                token_types,
                mask,
            } => write!(
                f,
                "token, token-type and mask lengths must agree, found {tokens}, {token_types} \
                 and {mask}"
            ),
            Self::UnknownToken { token, vocabulary } => write!(
                f,
                "token id {token} is outside the model's vocabulary of {vocabulary}"
            ),
            Self::UnknownTokenType {
                kind,
                type_vocabulary,
            } => write!(
                f,
                "token type id {kind} is outside the model's type vocabulary of {type_vocabulary}"
            ),
            Self::ShapeOverflow { sequence } => write!(
                f,
                "a sequence of {sequence} positions does not fit an addressable byte envelope"
            ),
        }
    }
}

impl std::error::Error for EncoderConstraint {}

/// Concrete device shape of one request, and the storage it covers.
///
/// The envelope is the whole request's device footprint, not only the result: the
/// staging buffers, every activation the forward pass keeps alive, and the two
/// outputs. A shared pool must cover all of it before the work is admitted, because
/// none of it is reusable until the request is released.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncoderShape {
    sequence: usize,
    hidden: usize,
    heads: usize,
    hidden_values: usize,
    intermediate_values: usize,
    device_bytes: u64,
}

impl EncoderShape {
    #[must_use]
    pub const fn sequence(self) -> usize {
        self.sequence
    }

    #[must_use]
    pub const fn hidden(self) -> usize {
        self.hidden
    }

    #[must_use]
    pub const fn heads(self) -> usize {
        self.heads
    }

    /// `[sequence, hidden]` value count of every activation row buffer.
    #[must_use]
    pub const fn hidden_values(self) -> usize {
        self.hidden_values
    }

    /// `[sequence, intermediate]` value count of the feed-forward buffer.
    #[must_use]
    pub const fn intermediate_values(self) -> usize {
        self.intermediate_values
    }

    /// Bytes this request holds on the device from submission until release.
    ///
    /// `EncoderSubmission::device_bytes` reports the same quantity from the
    /// allocations that actually exist, so a device test can prove the prediction
    /// the pool is charged for equals the storage it covers.
    #[must_use]
    pub const fn device_bytes(self) -> u64 {
        self.device_bytes
    }
}

/// Host-prepared request geometry bound to one backend's sequence limit.
/// The model configuration remains unchanged; every accepted shape uses the same
/// immutable intersection of model and backend limits.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderPreparation {
    config: BertConfig,
    max_sequence: usize,
}

impl EncoderPreparation {
    /// Bind validated model geometry to a positive backend execution limit.
    ///
    /// # Errors
    /// Returns [`ConfigError`] for invalid model geometry or numerical policy.
    pub fn new(
        config: BertConfig,
        backend_max_sequence: NonZeroUsize,
    ) -> Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self {
            max_sequence: config
                .max_position_embeddings
                .min(backend_max_sequence.get()),
            config,
        })
    }

    #[must_use]
    pub const fn config(&self) -> &BertConfig {
        &self.config
    }

    /// Effective executable sequence limit, not a rewritten model position table.
    #[must_use]
    pub const fn max_sequence(&self) -> usize {
        self.max_sequence
    }

    /// Concrete storage shape; rejects model/backend limits before byte arithmetic.
    ///
    /// # Errors
    /// Returns a permanent request-local constraint for an unsupported sequence.
    pub fn shape(&self, sequence: usize) -> Result<EncoderShape, EncoderConstraint> {
        if sequence == 0 {
            return Err(EncoderConstraint::EmptySequence);
        }
        if sequence > self.config.max_position_embeddings {
            return Err(EncoderConstraint::SequenceTooLong {
                sequence,
                max_position_embeddings: self.config.max_position_embeddings,
            });
        }
        if sequence > self.max_sequence {
            return Err(EncoderConstraint::BackendSequenceLimit {
                sequence,
                max_sequence: self.max_sequence,
            });
        }
        let hidden_values = sequence.checked_mul(self.config.hidden_size);
        let intermediate_values = sequence.checked_mul(self.config.intermediate_size);
        let (Some(hidden_values), Some(intermediate_values)) = (hidden_values, intermediate_values)
        else {
            return Err(EncoderConstraint::ShapeOverflow { sequence });
        };
        // Staging plus every activation the forward pass keeps alive: three id vectors,
        // six `[sequence, hidden]` float buffers, the `[sequence, intermediate]`
        // feed-forward buffer, and the two `[hidden]` outputs.
        let device_bytes = (|| {
            let float_values = hidden_values
                .checked_mul(6)?
                .checked_add(intermediate_values)?
                .checked_add(self.config.hidden_size.checked_mul(2)?)?;
            let elements = float_values.checked_add(sequence.checked_mul(3)?)?;
            u64::try_from(elements.checked_mul(size_of::<f32>())?).ok()
        })();
        let Some(device_bytes) = device_bytes else {
            return Err(EncoderConstraint::ShapeOverflow { sequence });
        };
        Ok(EncoderShape {
            sequence,
            hidden: self.config.hidden_size,
            heads: self.config.num_attention_heads,
            hidden_values,
            intermediate_values,
            device_bytes,
        })
    }

    /// Validate contents and compute the same shape used by selection and enqueue.
    ///
    /// # Errors
    /// Returns the first shape or input constraint this request violates.
    pub fn shape_of(&self, request: &EncoderRequest) -> Result<EncoderShape, EncoderConstraint> {
        let shape = self.shape(request.sequence())?;
        if request.token_type_ids.len() != request.token_ids.len()
            || request.attention_mask.len() != request.token_ids.len()
        {
            return Err(EncoderConstraint::LengthMismatch {
                tokens: request.token_ids.len(),
                token_types: request.token_type_ids.len(),
                mask: request.attention_mask.len(),
            });
        }
        for token in &request.token_ids {
            if usize::try_from(*token).map_or(true, |token| token >= self.config.vocab_size) {
                return Err(EncoderConstraint::UnknownToken {
                    token: *token,
                    vocabulary: self.config.vocab_size,
                });
            }
        }
        for kind in &request.token_type_ids {
            if usize::try_from(*kind).map_or(true, |kind| kind >= self.config.type_vocab_size) {
                return Err(EncoderConstraint::UnknownTokenType {
                    kind: *kind,
                    type_vocabulary: self.config.type_vocab_size,
                });
            }
        }
        Ok(shape)
    }

    /// Select the executable FIFO prefix, or reject an infeasible head locally.
    /// A defect behind a healthy prefix stops selection until it becomes the head.
    /// Empty input selects zero; the batching runtime supplies nonempty candidates.
    #[must_use]
    pub fn select(&self, candidates: &[&EncoderRequest]) -> BatchSelection<EncoderConstraint> {
        let mut items = 0_usize;
        let mut defect = None;
        for candidate in candidates {
            match self.shape_of(candidate) {
                Ok(_) => items += 1,
                Err(constraint) => {
                    defect = Some(constraint);
                    break;
                }
            }
        }
        // A defect in the head is permanent and reported now; the same defect further
        // along the candidate set only stops the accepted range.
        match defect {
            Some(constraint) if items == 0 => BatchSelection::Rejected(constraint),
            _ => BatchSelection::Ready { items },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BertConfig {
        BertConfig {
            vocab_size: 8,
            hidden_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 3,
            intermediate_size: 20,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            layer_norm_eps: 1.0e-12,
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn backend_sequence_limit_rejects_before_accepting_a_healthy_peer() {
        let limit = engine_nvidia::CudaEncoderOps::MAX_ATTENTION_SEQUENCE.get();
        let config = BertConfig {
            max_position_embeddings: limit + 1,
            ..config()
        };
        let preparation = EncoderPreparation::new(
            config,
            engine_nvidia::CudaEncoderOps::MAX_ATTENTION_SEQUENCE,
        )
        .unwrap();
        assert_eq!(preparation.config().max_position_embeddings, limit + 1);
        assert_eq!(preparation.max_sequence(), limit);
        let oversized = EncoderRequest::single_segment(vec![1; limit + 1]);
        let healthy = EncoderRequest::single_segment(vec![1; limit]);
        assert_eq!(
            preparation.select(&[&oversized, &healthy]),
            BatchSelection::Rejected(EncoderConstraint::BackendSequenceLimit {
                sequence: limit + 1,
                max_sequence: limit,
            })
        );
        assert_eq!(
            preparation.select(&[&healthy, &oversized]),
            BatchSelection::Ready { items: 1 }
        );
        assert!(preparation.shape(limit).is_ok());
    }

    #[test]
    fn envelope_covers_every_buffer_the_request_keeps() {
        let preparation = EncoderPreparation::new(config(), NonZeroUsize::MAX).unwrap();
        let shape = preparation.shape(4).expect("shape");
        assert_eq!(shape.hidden_values(), 48);
        assert_eq!(shape.intermediate_values(), 80);
        // six activation row buffers plus the feed-forward buffer plus two outputs,
        // all f32, plus three id vectors.
        let expected = (6 * 48 + 80 + 2 * 12 + 3 * 4) * size_of::<f32>();
        assert_eq!(shape.device_bytes(), u64::try_from(expected).unwrap());
        assert_eq!(
            preparation
                .shape_of(&EncoderRequest::single_segment(vec![1, 2, 3, 4]))
                .expect("envelope")
                .device_bytes(),
            shape.device_bytes()
        );
    }

    #[test]
    fn the_envelope_grows_with_the_sequence_and_not_with_the_batch() {
        let preparation = EncoderPreparation::new(config(), NonZeroUsize::MAX).unwrap();
        let short = preparation
            .shape_of(&EncoderRequest::single_segment(vec![1]))
            .expect("short")
            .device_bytes();
        let long = preparation
            .shape_of(&EncoderRequest::single_segment(vec![1; 8]))
            .expect("long")
            .device_bytes();
        assert!(short < long);
        // A concrete shape, not a per-row constant.
        assert_eq!(
            long,
            u64::try_from((6 * 96 + 160 + 2 * 12 + 3 * 8) * size_of::<f32>()).unwrap()
        );
    }

    #[test]
    fn a_request_defect_is_reported_as_a_permanent_constraint() {
        let preparation = EncoderPreparation::new(config(), NonZeroUsize::MAX).unwrap();
        assert_eq!(
            preparation.shape_of(&EncoderRequest::single_segment(Vec::new())),
            Err(EncoderConstraint::EmptySequence)
        );
        assert_eq!(
            preparation.shape_of(&EncoderRequest::single_segment(vec![1; 17])),
            Err(EncoderConstraint::SequenceTooLong {
                sequence: 17,
                max_position_embeddings: 16
            })
        );
        assert_eq!(
            preparation.shape_of(&EncoderRequest::single_segment(vec![99])),
            Err(EncoderConstraint::UnknownToken {
                token: 99,
                vocabulary: 8
            })
        );
        let mismatched = EncoderRequest {
            token_ids: vec![1, 2],
            token_type_ids: vec![0],
            attention_mask: vec![1, 1],
        };
        assert_eq!(
            preparation.shape_of(&mismatched),
            Err(EncoderConstraint::LengthMismatch {
                tokens: 2,
                token_types: 1,
                mask: 2
            })
        );
    }

    #[test]
    fn selection_accepts_the_executable_prefix_and_rejects_a_defective_head() {
        let preparation = EncoderPreparation::new(config(), NonZeroUsize::MAX).unwrap();
        let healthy = [EncoderRequest::single_segment(vec![1, 2, 3])];
        let candidates = healthy.iter().collect::<Vec<_>>();
        assert_eq!(
            preparation.select(&candidates),
            BatchSelection::Ready { items: 1 }
        );
        assert_eq!(preparation.select(&[]), BatchSelection::Ready { items: 0 });

        let long = EncoderRequest::single_segment(vec![1; 17]);
        let candidates = vec![&long];
        assert_eq!(
            preparation.select(&candidates),
            BatchSelection::Rejected(EncoderConstraint::SequenceTooLong {
                sequence: 17,
                max_position_embeddings: 16
            })
        );
        let unknown = EncoderRequest::single_segment(vec![99]);
        let candidates = vec![&unknown];
        assert_eq!(
            preparation.select(&candidates),
            BatchSelection::Rejected(EncoderConstraint::UnknownToken {
                token: 99,
                vocabulary: 8
            })
        );

        // A defect behind the head stops the range instead of rejecting work the
        // encoder can execute: the defect becomes the head once the prefix drains.
        let short = EncoderRequest::single_segment(vec![1]);
        let candidates = vec![&short, &long, &unknown];
        assert_eq!(
            preparation.select(&candidates),
            BatchSelection::Ready { items: 1 }
        );
    }
}
