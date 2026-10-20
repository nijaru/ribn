//! Exact host GELU specialization for the finite BF16 input domain.

use crate::Gemma4Error;
use crate::cpu::{gelu_tanh, try_vector};

const ENCODINGS: usize = 1 << 16;

pub(super) struct GeluTable {
    values: Box<[u16]>,
}

impl GeluTable {
    pub(super) fn new() -> Result<Self, Gemma4Error> {
        let mut inputs = try_vector(ENCODINGS, 0_u16)?;
        for (slot, bits) in inputs.iter_mut().zip(0..=u16::MAX) {
            if finite(bits) {
                *slot = bits;
            }
        }
        // Keep the CPU oracle's algebra, constants, tanh and BF16 rounding in one
        // place. Non-finite slots use zero only during preparation; apply rejects
        // their original encodings before indexing, preserving the input contract.
        let values = gelu_tanh(&inputs)?.into_boxed_slice();
        Ok(Self { values })
    }

    pub(super) fn apply(&self, input: &[u16]) -> Result<Vec<u16>, Gemma4Error> {
        let mut output = try_vector(input.len(), 0_u16)?;
        for (slot, &bits) in output.iter_mut().zip(input) {
            if !finite(bits) {
                return Err(Gemma4Error::new("Metal GELU", "input must be finite"));
            }
            *slot = self.values[usize::from(bits)];
        }
        Ok(output)
    }
}

fn finite(bits: u16) -> bool {
    bits & 0x7f80 != 0x7f80
}

#[cfg(test)]
mod tests {
    use super::GeluTable;
    use crate::cpu::gelu_tanh;

    #[test]
    fn table_matches_reference_for_every_encoding() {
        let table = GeluTable::new().unwrap();
        let finite: Vec<_> = (0..=u16::MAX)
            .filter(|&bits| f32::from_bits(u32::from(bits) << 16).is_finite())
            .collect();
        let actual = table.apply(&finite).unwrap();
        assert_eq!(actual.len(), finite.len());
        for (&bits, actual) in finite.iter().zip(actual) {
            let expected = gelu_tanh(&[bits]).unwrap()[0];
            assert!(
                f32::from_bits(u32::from(actual) << 16).is_finite(),
                "BF16 input {bits:#06x}"
            );
            assert_eq!(actual, expected, "BF16 input {bits:#06x}");
        }
        for bits in 0..=u16::MAX {
            if !f32::from_bits(u32::from(bits) << 16).is_finite() {
                let input = [0x3f80, bits, 0xbf80];
                assert!(gelu_tanh(&input).is_err());
                assert!(table.apply(&input).is_err(), "BF16 input {bits:#06x}");
            }
        }
        assert!(table.apply(&[]).unwrap().is_empty());
        let input = [0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f7f, 0xff7f];
        assert_eq!(table.apply(&input).unwrap(), gelu_tanh(&input).unwrap());
    }
}
