//! Host materialization of the validated Gemma 4 BF16 parameter inventory.

use crate::{Gemma4Error, Gemma4Source, LayerParameter, ParameterId};
use std::collections::BTreeMap;
use std::io::Read;

/// One prepared host parameter, preserving its native BF16 encoding and shape.
pub struct CpuParameter {
    shape: Box<[usize]>,
    values: Box<[u16]>,
}

impl CpuParameter {
    /// Construct a prepared parameter from host BF16 values for synthetic use.
    /// # Errors
    /// Rejects shapes whose element count differs from the supplied values.
    pub fn from_parts(shape: &[usize], values: &[u16]) -> Result<Self, Gemma4Error> {
        let count = shape.iter().try_fold(1_usize, |product, &dimension| {
            product.checked_mul(dimension)
        });
        if count != Some(values.len()) {
            return Err(Gemma4Error::new(
                "CPU parameter",
                "shape and value count disagree",
            ));
        }
        Ok(Self {
            shape: shape.into(),
            values: values.into(),
        })
    }

    /// Materialize an exact BF16 source with bounded scratch space.
    pub(crate) fn read(shape: &[usize], mut reader: impl Read) -> Result<Self, Gemma4Error> {
        let count = shape
            .iter()
            .try_fold(1_usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| Gemma4Error::new("CPU weight", "shape overflows"))?;
        let length = count
            .checked_mul(2)
            .ok_or_else(|| Gemma4Error::new("CPU weight", "encoded length overflows"))?;
        let mut values = crate::cpu::try_vector(count, 0_u16)?;
        let mut scratch = crate::cpu::try_vector(64 * 1024, 0_u8)?;
        let mut written = 0;
        while written < length {
            let want = (length - written).min(scratch.len());
            reader
                .read_exact(&mut scratch[..want])
                .map_err(|e| Gemma4Error::new("CPU weight", e.to_string()))?;
            for (index, pair) in scratch[..want].as_chunks::<2>().0.iter().enumerate() {
                values[written / 2 + index] = u16::from_le_bytes(*pair);
            }
            written += want;
        }
        if reader
            .read(&mut [0_u8; 1])
            .map_err(|e| Gemma4Error::new("CPU weight", e.to_string()))?
            != 0
        {
            return Err(Gemma4Error::new(
                "CPU weight",
                "source has trailing bytes beyond its declared shape",
            ));
        }
        Ok(Self {
            shape: shape.into(),
            values: values.into_boxed_slice(),
        })
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn values(&self) -> &[u16] {
        &self.values
    }
}

/// Every active text parameter in host memory, including one tied output head.
///
/// Loading reads each retained source reader once, checks the exact encoded length
/// and refuses trailing bytes. This is a synchronous host snapshot, not a device
/// grant or an executable snapshot of a mutable package.
pub struct Gemma4CpuWeights {
    parameters: BTreeMap<ParameterId, CpuParameter>,
    bytes: u64,
    config: crate::Gemma4Config,
}

impl Gemma4CpuWeights {
    /// Read the complete active inventory into host BF16 buffers.
    /// # Errors
    /// Rejects length overflow, short sources, trailing bytes and allocation failures.
    pub fn load(source: &Gemma4Source) -> Result<Self, Gemma4Error> {
        let mut parameters = BTreeMap::new();
        let mut bytes = 0_u64;
        for (id, stored) in source.parameters() {
            let spec = stored.spec();
            let length = spec.byte_len()?;
            parameters.insert(*id, CpuParameter::read(&spec.shape, stored.reader())?);
            bytes =
                bytes
                    .checked_add(u64::try_from(length).map_err(|_| {
                        Gemma4Error::new("CPU weight", "encoded length exceeds u64")
                    })?)
                    .ok_or_else(|| Gemma4Error::new("CPU weight", "total bytes overflow"))?;
        }
        let expected = u64::try_from(source.encoded_bytes())
            .map_err(|_| Gemma4Error::new("CPU weight", "source bytes exceed u64"))?;
        if bytes != expected {
            return Err(Gemma4Error::new(
                "CPU weight",
                "materialized bytes differ from the validated inventory",
            ));
        }
        Ok(Self {
            parameters,
            bytes,
            config: source.config().clone(),
        })
    }

    /// Source configuration this inventory was validated against.
    #[must_use]
    pub const fn config(&self) -> &crate::Gemma4Config {
        &self.config
    }

    #[must_use]
    pub fn parameter(&self, id: ParameterId) -> Option<&CpuParameter> {
        self.parameters.get(&id)
    }

    /// Every active parameter in identifier order.
    pub fn parameters(&self) -> impl Iterator<Item = (ParameterId, &CpuParameter)> {
        self.parameters
            .iter()
            .map(|(&id, parameter)| (id, parameter))
    }

    /// Decoder-layer parameter; inactive shared-tail roles are absent by design.
    /// # Errors
    /// Rejects an unknown layer/role instead of substituting another owner's bytes.
    pub fn layer(
        &self,
        layer: usize,
        parameter: LayerParameter,
    ) -> Result<&CpuParameter, Gemma4Error> {
        self.parameter(ParameterId::Layer { layer, parameter })
            .ok_or_else(|| {
                Gemma4Error::new(
                    "CPU weight",
                    format!("layer {layer} has no active {parameter:?} parameter"),
                )
            })
    }

    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::CpuParameter;

    #[test]
    fn bf16_reader_preserves_scalar_rank_and_crosses_scratch_boundaries() {
        let scalar = CpuParameter::read(&[], &[0x80, 0xbe][..]).unwrap();
        assert!(scalar.shape().is_empty());
        assert_eq!(scalar.values(), &[0xbe80]);
        for bytes in [&[0x80][..], &[0x80, 0xbe, 0x00][..]] {
            assert!(CpuParameter::read(&[], bytes).is_err());
        }
        assert!(CpuParameter::read(&[usize::MAX, 2], &[][..]).is_err());
        let words: Vec<u16> = (0..32769).collect();
        let bytes: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
        let parameter = CpuParameter::read(&[words.len()], bytes.as_slice()).unwrap();
        assert_eq!(parameter.values(), words);
    }
}
