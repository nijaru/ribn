//! Checked command preparation and synchronous completion before publication.

use super::execution::Execution;
use super::{MetalBf16Error, MetalBf16Executor};
use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::objc::{Message, rc::autoreleasepool, runtime::Sel};
use metal::{
    CommandBuffer, CommandBufferRef, CommandQueue, ComputeCommandEncoder, ComputeCommandEncoderRef,
    DeviceRef, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLComputeCommandEncoder,
};
use std::{ops::Deref, sync::MutexGuard};

pub(super) struct NativeCommand<'a> {
    buffer: CommandBuffer,
    owner: MutexGuard<'a, Execution>,
}

impl Deref for NativeCommand<'_> {
    type Target = CommandBufferRef;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl NativeCommand<'_> {
    pub fn submit(&mut self, operation: &str) -> Result<(), MetalBf16Error> {
        self.owner.begin_native(&self.buffer);
        self.buffer.commit();
        self.buffer.wait_until_completed();
        let status = self.buffer.status();
        if !self.owner.finish_native(status) {
            return Err(MetalBf16Error::Execution(
                "native wait did not establish terminal completion".into(),
            ));
        }
        if status == MTLCommandBufferStatus::Completed {
            Ok(())
        } else {
            Err(MetalBf16Error::Execution(format!(
                "{operation}: {status:?}"
            )))
        }
    }
}

impl MetalBf16Executor {
    pub(super) fn command_queue(device: &DeviceRef) -> Result<CommandQueue, MetalBf16Error> {
        // SAFETY: nullable newCommandQueue returns a +1 owned reference.
        let pointer: *mut MTLCommandQueue =
            unsafe { device.send_message(Sel::register("newCommandQueue"), ()) }
                .map_err(|error| MetalBf16Error::Allocation(format!("command queue: {error}")))?;
        if pointer.is_null() {
            return Err(MetalBf16Error::Allocation(
                "command queue returned nil".into(),
            ));
        }
        // SAFETY: non-null, newly owned MTLCommandQueue reference.
        Ok(unsafe { CommandQueue::from_ptr(pointer) })
    }

    pub(super) fn command(&self) -> Result<NativeCommand<'_>, MetalBf16Error> {
        let owner = self.owner()?;
        // Bound the +0 handle's autorelease even on a Rust worker without a
        // caller pool. The returned +1 owner still lives through terminal wait.
        let buffer = autoreleasepool(|| {
            // SAFETY: commandBuffer returns a nullable autoreleased reference. The
            // ordinary (retained-references) buffer keeps encoded resources alive.
            let pointer: *mut MTLCommandBuffer =
                unsafe { self.queue.send_message(Sel::register("commandBuffer"), ()) }.map_err(
                    |error| MetalBf16Error::Allocation(format!("command buffer: {error}")),
                )?;
            if pointer.is_null() {
                return Err(MetalBf16Error::Allocation(
                    "command buffer returned nil".into(),
                ));
            }
            // SAFETY: retain the checked +0 reference before this pool drains.
            Ok(unsafe { CommandBufferRef::from_ptr(pointer) }.to_owned())
        })?;
        Ok(NativeCommand { buffer, owner })
    }

    pub(super) fn compute_encoder(
        command: &CommandBufferRef,
    ) -> Result<ComputeCommandEncoder, MetalBf16Error> {
        autoreleasepool(|| {
            // SAFETY: computeCommandEncoder returns a nullable +0 serial encoder.
            let pointer: *mut MTLComputeCommandEncoder =
                unsafe { command.send_message(Sel::register("computeCommandEncoder"), ()) }
                    .map_err(|error| {
                        MetalBf16Error::Allocation(format!("compute encoder: {error}"))
                    })?;
            if pointer.is_null() {
                return Err(MetalBf16Error::Allocation(
                    "compute encoder returned nil".into(),
                ));
            }
            // SAFETY: retain the checked +0 reference before this pool drains.
            Ok(unsafe { ComputeCommandEncoderRef::from_ptr(pointer) }.to_owned())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{Arc, mpsc},
        time::Duration,
    };

    #[test]
    fn failure_settlement_queries_a_gated_native_command_without_reopening() {
        let executor = match MetalBf16Executor::new() {
            Ok(executor) => Arc::new(executor),
            Err(MetalBf16Error::Device) => return,
            Err(error) => panic!("native preparation: {error}"),
        };
        // A concurrent active call must not make failure settlement wait for its
        // mutex, even before it submits device work. Release before joining so a
        // blocking regression fails this test rather than stranding the thread.
        let active = executor.command().unwrap();
        let (sent, received) = mpsc::channel();
        let owner = Arc::clone(&executor);
        let query = std::thread::spawn(move || {
            sent.send(owner.retire_failure().is_err()).unwrap();
        });
        let early = received.recv_timeout(Duration::from_secs(2));
        drop(active);
        query.join().unwrap();
        assert!(
            early.is_ok_and(|rejected| rejected),
            "settlement must refuse a busy execution owner without waiting"
        );
        executor.check_healthy().unwrap();

        let gate = executor.device.new_shared_event();
        gate.set_signaled_value(0);
        let pending = {
            let mut command = executor.command().unwrap();
            command.encode_wait_for_event(&gate, 1);
            command.owner.begin_native(&command.buffer);
            command.owner.fault();
            command.buffer.commit();
            command.buffer.clone()
        };
        let (sent, received) = mpsc::channel();
        let owner = Arc::clone(&executor);
        let query = std::thread::spawn(move || {
            sent.send(owner.retire_failure().is_err()).unwrap();
        });
        let early = received.recv_timeout(Duration::from_secs(2));
        // Release the real GPU gate even if a regression made retirement wait.
        gate.set_signaled_value(1);
        pending.wait_until_completed();
        query.join().unwrap();
        assert!(
            early.unwrap(),
            "query must reject before the GPU gate opens"
        );
        executor.retire_failure().unwrap();
        executor.retire_failure().unwrap();
        assert!(
            executor.check_healthy().is_err(),
            "retirement never reopens a failed owner"
        );
    }
}
