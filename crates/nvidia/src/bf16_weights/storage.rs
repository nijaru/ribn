use std::{collections::HashSet, io::Read, ops::Range, sync::Arc};

use engine_core::{DataType, WeightTensorSpec};
use ribn_foundation::{AllocationId, BytePool};

use crate::cuda_allocation::{Allocation, CudaBf16Error, Transport};

type Result<T> = std::result::Result<T, CudaBf16Error>;

struct Slot {
    spec: WeightTensorSpec,
    range: Range<u64>,
    uploaded: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Loading,
    Ready,
    Poisoned,
    Closed,
}

pub(super) struct Storage<T: Transport> {
    allocation: Option<Arc<Allocation<T>>>,
    slots: Vec<Slot>,
    bytes: u64,
    allocation_id: AllocationId,
    state: State,
}

fn layout(specs: Vec<WeightTensorSpec>, staging_bytes: usize) -> Result<(Vec<Slot>, u64)> {
    if specs.is_empty() || staging_bytes == 0 {
        return Err(CudaBf16Error::Invalid("empty set or zero staging capacity"));
    }
    let mut names = HashSet::with_capacity(specs.len());
    let mut offset = 0_u64;
    let mut slots = Vec::with_capacity(specs.len());
    for spec in specs {
        if spec.dtype() != DataType::BF16 {
            return Err(CudaBf16Error::Invalid("only native BF16 is accepted"));
        }
        if !names.insert(spec.name().to_owned()) {
            return Err(CudaBf16Error::Invalid("duplicate tensor name"));
        }
        let bytes = spec
            .byte_size()
            .ok_or(CudaBf16Error::Invalid("tensor byte overflow"))?;
        let end = offset
            .checked_add(bytes)
            .ok_or(CudaBf16Error::Invalid("byte sum overflow"))?;
        // Include trailing padding on EVERY tensor (including the last). Logical
        // ranges exclude padding, which is never read or published as tensor data.
        let padded_end = end
            .checked_add(15)
            .ok_or(CudaBf16Error::Invalid("alignment overflow"))?
            & !15;
        usize::try_from(padded_end)
            .map_err(|_| CudaBf16Error::Invalid("allocation exceeds usize"))?;
        slots.push(Slot {
            spec,
            range: offset..end,
            uploaded: false,
        });
        offset = padded_end;
    }
    Ok((slots, offset))
}

impl<T: Transport> Storage<T> {
    pub(super) fn allocate(
        specs: Vec<WeightTensorSpec>,
        transport: T,
        pool: &Arc<BytePool>,
        staging_bytes: usize,
    ) -> Result<Self> {
        let (slots, bytes) = layout(specs, staging_bytes)?;
        let allocation = Allocation::allocate(bytes, transport, pool, staging_bytes)?;
        Ok(Self {
            bytes: allocation.allocated_bytes(),
            allocation_id: allocation.allocation_id(),
            allocation: Some(allocation),
            slots,
            state: State::Loading,
        })
    }

    pub(super) fn allocated_bytes(&self) -> u64 {
        self.bytes
    }
    pub(super) fn allocation_id(&self) -> AllocationId {
        self.allocation_id
    }
    pub(super) fn spec(&self, slot: usize) -> Option<&WeightTensorSpec> {
        self.slots.get(slot).map(|s| &s.spec)
    }
    pub(super) fn tensor_range(&self, slot: usize) -> Option<Range<u64>> {
        self.slots.get(slot).map(|s| s.range.clone())
    }
    pub(super) fn ready(&self) -> bool {
        self.state == State::Ready
    }

    fn check_open(&self) -> Result<()> {
        match self.state {
            State::Closed => Err(CudaBf16Error::Closed),
            State::Poisoned => Err(CudaBf16Error::Poisoned),
            State::Loading | State::Ready => Ok(()),
        }
    }

    pub(crate) fn access(&self, slot: usize) -> Result<(Arc<Allocation<T>>, Range<u64>)> {
        self.check_open()?;
        if !self.ready() {
            return Err(CudaBf16Error::NotReady);
        }
        let range = self
            .tensor_range(slot)
            .ok_or(CudaBf16Error::InvalidSlot(slot))?;
        let allocation = self.allocation.as_ref().expect("open owner has allocation");
        // Retained operation/error ownership forbids another synchronous use, not
        // only mutation or retirement. No concurrent-reader contract is selected.
        if Arc::strong_count(allocation) != 1 {
            return Err(CudaBf16Error::InUse);
        }
        allocation.pointer()?;
        Ok((Arc::clone(allocation), range))
    }

    pub(super) fn upload(&mut self, slot: usize, reader: &mut impl Read) -> Result<()> {
        self.check_open()?;
        let allocation = Arc::get_mut(self.allocation.as_mut().expect("open owner has allocation"))
            .ok_or(CudaBf16Error::InUse)?;
        let tensor = self
            .slots
            .get(slot)
            .ok_or(CudaBf16Error::InvalidSlot(slot))?;
        if tensor.uploaded {
            return Err(CudaBf16Error::AlreadyUploaded(slot));
        }
        let range = tensor.range.clone();
        // Poison BEFORE any user reader/FFI call: errors or unwinding cannot publish
        // a partial set. Only a complete successful upload restores loading/ready.
        self.state = State::Poisoned;
        allocation.write_exact(range.start, range.end - range.start, reader)?;
        self.slots[slot].uploaded = true;
        self.state = if self.slots.iter().all(|s| s.uploaded) {
            State::Ready
        } else {
            State::Loading
        };
        Ok(())
    }

    pub(super) fn read_chunk(
        &mut self,
        slot: usize,
        byte_offset: u64,
        len: usize,
    ) -> Result<Vec<u8>> {
        self.check_open()?;
        if !self.ready() {
            return Err(CudaBf16Error::NotReady);
        }
        let allocation = Arc::get_mut(self.allocation.as_mut().expect("open owner has allocation"))
            .ok_or(CudaBf16Error::InUse)?;
        let tensor = self
            .slots
            .get(slot)
            .ok_or(CudaBf16Error::InvalidSlot(slot))?;
        let end = byte_offset
            .checked_add(len as u64)
            .ok_or(CudaBf16Error::Invalid("read offset overflow"))?;
        if end > tensor.range.end - tensor.range.start {
            return Err(CudaBf16Error::Invalid("read exceeds tensor"));
        }
        let offset = tensor.range.start + byte_offset;
        allocation.check_read(offset, len)?;
        if len == 0 {
            return Ok(Vec::new());
        }
        self.state = State::Poisoned;
        let bytes = allocation.read_chunk(offset, len)?;
        self.state = State::Ready;
        Ok(bytes)
    }

    pub(super) fn close(&mut self) -> Result<()> {
        let Some(allocation) = self.allocation.as_mut() else {
            return Ok(());
        };
        // Borrow refusal is not a physical failure and cannot invalidate the set.
        let allocation = Arc::get_mut(allocation).ok_or(CudaBf16Error::InUse)?;
        self.state = State::Poisoned;
        allocation.close()?;
        self.allocation = None;
        self.state = State::Closed;
        Ok(())
    }
}

// Default Drop relinquishes only this Arc. Retained device access owns retirement
// independently of the weight root; Allocation's defensive Drop runs on the last Arc.

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
