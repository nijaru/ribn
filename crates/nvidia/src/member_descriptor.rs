//! Wire layout for one decode step's continuation addresses. No device ownership lives here.
use std::ops::Range;

use crate::MAX_BATCH_MEMBERS;

/// Three header words, followed by member-major records. Each record contains
/// table/position/page size/capacity, KV pointer pairs, then recurrent pointer pairs.
/// All words are u64, so CUDA can read addresses without struct ABI assumptions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DescriptorLayout {
    pub members: usize,
    pub kv_layers: usize,
    pub recurrent_layers: usize,
}

impl DescriptorLayout {
    pub fn new(
        members: usize,
        kv_layers: usize,
        recurrent_layers: usize,
    ) -> Result<Self, &'static str> {
        if members == 0 || members > MAX_BATCH_MEMBERS {
            return Err("descriptor member count is outside the prepared lane bound");
        }
        // State declarations use u16 layer counts; bound the packet before allocating it.
        if kv_layers > usize::from(u16::MAX)
            || recurrent_layers > usize::from(u16::MAX)
            || kv_layers + recurrent_layers == 0
        {
            return Err("descriptor requires representable continuation layer counts");
        }
        Ok(Self {
            members,
            kv_layers,
            recurrent_layers,
        })
    }

    pub const fn member_words(self) -> usize {
        4 + 2 * (self.kv_layers + self.recurrent_layers)
    }

    pub const fn words(self) -> usize {
        3 + self.members * self.member_words()
    }

    pub fn member_range(self, member: usize) -> Option<Range<usize>> {
        (member < self.members).then(|| {
            let start = 3 + member * self.member_words();
            start..start + self.member_words()
        })
    }
}

pub(crate) struct DescriptorPacket {
    pub layout: DescriptorLayout,
    words: Vec<u64>,
}

impl DescriptorPacket {
    pub fn new(layout: DescriptorLayout) -> Self {
        let mut words = vec![0; layout.words()];
        words[..3].copy_from_slice(&[
            layout.members as u64,
            layout.kv_layers as u64,
            layout.recurrent_layers as u64,
        ]);
        Self { layout, words }
    }

    pub fn member_mut(&mut self, member: usize) -> Option<&mut [u64]> {
        self.words.get_mut(self.layout.member_range(member)?)
    }

    /// Reject incomplete records before the packet can reach a kernel. Allocation
    /// size, dtype, context and page-table validity belong to the CUDA state owner.
    pub fn validated_words(&self) -> Result<&[u64], &'static str> {
        for member in 0..self.layout.members {
            let row = &self.words[self.layout.member_range(member).expect("bounded member")];
            if self.layout.kv_layers > 0 {
                let [table, position, page_tokens, capacity] = row[..4] else {
                    unreachable!()
                };
                if table == 0 || !table.is_multiple_of(4)
                    || page_tokens == 0 || capacity == 0
                    || !capacity.is_multiple_of(page_tokens) || position >= capacity
                    // Existing attention kernels index tokens/pages with signed ints.
                    || capacity > i32::MAX as u64
                {
                    return Err("invalid descriptor page geometry, table or position");
                }
            } else if row[..4] != [0; 4] {
                return Err("descriptor without KV must not carry page metadata");
            }
            let kv_end = 4 + 2 * self.layout.kv_layers;
            if row[4..kv_end]
                .iter()
                .any(|&pointer| pointer == 0 || !pointer.is_multiple_of(2))
                || row[kv_end..]
                    .iter()
                    .any(|&pointer| pointer == 0 || !pointer.is_multiple_of(4))
            {
                return Err("descriptor contains missing or misaligned continuation addresses");
            }
        }
        Ok(&self.words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_heterogeneous_members_without_overlap_or_slot_reordering() {
        let layout = DescriptorLayout::new(2, 2, 1).unwrap();
        let mut packet = DescriptorPacket::new(layout);
        let first = [
            0x1000, 15, 16, 32, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000,
        ];
        let second = [
            0x8000, 32, 16, 64, 0x9000, 0xa000, 0xb000, 0xc000, 0xd000, 0xe000,
        ];
        packet.member_mut(0).unwrap().copy_from_slice(&first);
        packet.member_mut(1).unwrap().copy_from_slice(&second);
        let words = packet.validated_words().unwrap();
        assert_eq!(&words[..3], &[2, 2, 1]);
        assert_eq!(&words[3..13], &first);
        assert_eq!(&words[13..23], &second);
        assert!(packet.member_mut(2).is_none());
        assert_eq!(layout.words() * size_of::<u64>(), 184);
    }

    #[test]
    fn rejects_incomplete_packets_and_invalid_page_reach() {
        let mut packet = DescriptorPacket::new(DescriptorLayout::new(1, 1, 1).unwrap());
        assert!(packet.validated_words().is_err());
        let valid = [0x1000, 31, 16, 32, 0x2000, 0x3000, 0x4000, 0x5000];
        for (index, bad) in [
            (0, 0),
            (0, 3),
            (1, 32),
            (2, 0),
            (3, 31),
            (3, 1 << 32),
            (4, 0),
            (5, 3),
            (6, 2),
            (7, 0),
        ] {
            packet.member_mut(0).unwrap().copy_from_slice(&valid);
            packet.member_mut(0).unwrap()[index] = bad;
            assert!(
                packet.validated_words().is_err(),
                "index {index} value {bad}"
            );
        }
        packet.member_mut(0).unwrap().copy_from_slice(&valid);
        assert!(packet.validated_words().is_ok());
    }

    #[test]
    fn bounds_packet_and_supports_single_family_models() {
        for (members, kv, recurrent) in [
            (0, 1, 0),
            (MAX_BATCH_MEMBERS + 1, 0, 1),
            (1, 0, 0),
            (1, 65536, 0),
            (1, 0, usize::MAX),
        ] {
            assert!(DescriptorLayout::new(members, kv, recurrent).is_err());
        }
        let mut recurrent = DescriptorPacket::new(DescriptorLayout::new(1, 0, 1).unwrap());
        recurrent
            .member_mut(0)
            .unwrap()
            .copy_from_slice(&[0, 0, 0, 0, 4, 8]);
        assert!(recurrent.validated_words().is_ok());
        recurrent.member_mut(0).unwrap()[1] = 1;
        assert!(recurrent.validated_words().is_err());
        let mut kv = DescriptorPacket::new(DescriptorLayout::new(1, 1, 0).unwrap());
        kv.member_mut(0)
            .unwrap()
            .copy_from_slice(&[4, 0, 16, 16, 2, 6]);
        assert!(kv.validated_words().is_ok());
    }
}
