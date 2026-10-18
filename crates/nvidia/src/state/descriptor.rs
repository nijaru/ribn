//! Borrows and event tracking for indirect continuation access during one enqueue.
use cudarc::driver::{DevicePtr, DevicePtrMut, SyncOnDrop};

use super::{CudaHybridState, CudaSlice, CudaStream};
use crate::decode::CudaDecodeError;
use crate::member_descriptor::{DescriptorLayout, DescriptorPacket};
use std::sync::Arc;

/// Depth-one lane workspace. Explicitly finish publication before reusing the
/// pageable host packet: cudarc's ordinary `HostSlice` implementations do not wait.
/// Failed upload/drain retains this packet; Drop retries or quarantines it.
pub(crate) struct DescriptorWorkspace {
    packet: Option<DescriptorPacket>,
    device: CudaSlice<u64>,
    stream: Arc<CudaStream>,
    upload_pending: bool,
}

/// Exclusive state borrows and cudarc access records outlive every indirect
/// launch, including a partially enqueued step. Dropping records writes *after*
/// those launches, so reuse/free on another stream waits on the actual access.
/// The serving executor still owns allocation/retirement after enqueue returns.
pub(crate) struct BoundDescriptor<'a> {
    pub device: &'a CudaSlice<u64>,
    pub layout: DescriptorLayout,
    pub stream: &'a Arc<CudaStream>,
    pub max_capacity: usize,
    _access: Vec<SyncOnDrop<'a>>,
}

impl DescriptorWorkspace {
    pub fn new(
        stream: &Arc<CudaStream>,
        layout: DescriptorLayout,
    ) -> Result<Self, CudaDecodeError> {
        let packet = DescriptorPacket::new(layout);
        let device = stream
            .alloc_zeros::<u64>(layout.words())
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
        Ok(Self {
            packet: Some(packet),
            device,
            stream: Arc::clone(stream),
            upload_pending: false,
        })
    }

    /// The caller validates the complete model state, positions, dtype and
    /// context before binding. This function additionally checks the exact
    /// descriptor shape before taking addresses; no state-changing work launches.
    pub fn bind<'a>(
        &'a mut self,
        states: &'a mut [&mut CudaHybridState],
        positions: &[u32],
        stream: &'a Arc<CudaStream>,
    ) -> Result<BoundDescriptor<'a>, CudaDecodeError> {
        if !Arc::ptr_eq(stream, &self.stream) {
            return Err(invalid("descriptor must be bound on its owning stream"));
        }
        if self.upload_pending {
            self.stream
                .synchronize()
                .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
            self.upload_pending = false;
        }
        let packet = self
            .packet
            .as_mut()
            .expect("live descriptor workspace owns its packet");
        let layout = packet.layout;
        if states.len() != layout.members || positions.len() != layout.members {
            return Err(invalid(
                "descriptor state/position count does not match lane",
            ));
        }
        for state in states.iter() {
            if state.kv.as_ref().map_or(0, |kv| kv.keys.len()) != layout.kv_layers
                || state.recurrent.as_ref().map_or(0, |rec| rec.matrix.len())
                    != layout.recurrent_layers
            {
                return Err(invalid("descriptor state layer counts do not match lane"));
            }
        }
        let max_capacity = states
            .iter()
            .filter_map(|state| {
                state
                    .kv
                    .as_ref()
                    .map(|kv| kv.spec.capacity_tokens() as usize)
            })
            .max()
            .unwrap_or(0);
        let mut access = Vec::with_capacity(
            layout.members * (1 + 2 * (layout.kv_layers + layout.recurrent_layers)),
        );
        for (member, (state, &position)) in states.iter_mut().zip(positions).enumerate() {
            let row = packet.member_mut(member).expect("validated member count");
            row.fill(0);
            if let Some(kv) = state.kv.as_mut() {
                let (table, record) = kv.table_device.device_ptr(stream);
                access.push(record);
                row[..4].copy_from_slice(&[
                    table,
                    u64::from(position),
                    u64::from(kv.spec.page_tokens()),
                    u64::from(kv.spec.capacity_tokens()),
                ]);
                for (slot, (keys, values)) in kv.keys.iter_mut().zip(&mut kv.values).enumerate() {
                    let keys = keys
                        .as_f16_mut()
                        .ok_or_else(|| invalid("descriptor KV must be F16"))?;
                    let values = values
                        .as_f16_mut()
                        .ok_or_else(|| invalid("descriptor KV must be F16"))?;
                    let (key_pointer, key_record) = keys.device_ptr_mut(stream);
                    let (value_pointer, value_record) = values.device_ptr_mut(stream);
                    access.extend([key_record, value_record]);
                    row[4 + slot * 2..6 + slot * 2].copy_from_slice(&[key_pointer, value_pointer]);
                }
            }
            if let Some(rec) = state.recurrent.as_mut() {
                let start = 4 + 2 * layout.kv_layers;
                for (slot, (matrix, conv)) in
                    rec.matrix.iter_mut().zip(&mut rec.convolution).enumerate()
                {
                    let matrix = matrix
                        .as_f32_mut()
                        .ok_or_else(|| invalid("descriptor matrix must be F32"))?;
                    let conv = conv
                        .as_f32_mut()
                        .ok_or_else(|| invalid("descriptor history must be F32"))?;
                    let (matrix_pointer, matrix_record) = matrix.device_ptr_mut(stream);
                    let (conv_pointer, conv_record) = conv.device_ptr_mut(stream);
                    access.extend([matrix_record, conv_record]);
                    row[start + slot * 2..start + slot * 2 + 2]
                        .copy_from_slice(&[matrix_pointer, conv_pointer]);
                }
            }
        }
        let words = packet.validated_words().map_err(invalid)?;
        self.upload_pending = true;
        stream
            .memcpy_htod(words, &mut self.device)
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
        // Initial eager candidate deliberately pays one publication wait. This
        // avoids a second staging/retirement protocol before overlap or graphs
        // exist, and makes the host packet immutable for the whole H2D copy.
        stream
            .synchronize()
            .map_err(|error| CudaDecodeError::Driver(error.to_string()))?;
        self.upload_pending = false;
        Ok(BoundDescriptor {
            device: &self.device,
            layout,
            stream,
            max_capacity,
            _access: access,
        })
    }
}

impl Drop for DescriptorWorkspace {
    fn drop(&mut self) {
        if self.upload_pending && self.stream.synchronize().is_err() {
            // The copy may still read host memory. A failed drain cannot turn
            // Drop into proof of completion; retain this bounded packet.
            if let Some(packet) = self.packet.take() {
                std::mem::forget(packet);
            }
        }
    }
}

fn invalid(message: &str) -> CudaDecodeError {
    CudaDecodeError::InvalidPlan(message.to_owned())
}
