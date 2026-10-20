//! Checked command preparation and synchronous completion before publication.

use super::{MetalBf16Error, MetalBf16Executor};
use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::objc::{Message, runtime::Sel};
use metal::{
    CommandBuffer, CommandBufferRef, CommandQueue, ComputeCommandEncoder, ComputeCommandEncoderRef,
    DeviceRef, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLComputeCommandEncoder,
};

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

    pub(super) fn command(&self) -> Result<CommandBuffer, MetalBf16Error> {
        // SAFETY: commandBuffer returns a nullable autoreleased reference. The
        // ordinary (retained-references) buffer keeps encoded resources alive.
        let pointer: *mut MTLCommandBuffer =
            unsafe { self.queue.send_message(Sel::register("commandBuffer"), ()) }
                .map_err(|error| MetalBf16Error::Allocation(format!("command buffer: {error}")))?;
        if pointer.is_null() {
            return Err(MetalBf16Error::Allocation(
                "command buffer returned nil".into(),
            ));
        }
        // SAFETY: retain the checked +0 reference before its autorelease pool ends.
        Ok(unsafe { CommandBufferRef::from_ptr(pointer) }.to_owned())
    }

    pub(super) fn compute_encoder(
        command: &CommandBufferRef,
    ) -> Result<ComputeCommandEncoder, MetalBf16Error> {
        // SAFETY: computeCommandEncoder returns a nullable +0 serial encoder.
        let pointer: *mut MTLComputeCommandEncoder =
            unsafe { command.send_message(Sel::register("computeCommandEncoder"), ()) }
                .map_err(|error| MetalBf16Error::Allocation(format!("compute encoder: {error}")))?;
        if pointer.is_null() {
            return Err(MetalBf16Error::Allocation(
                "compute encoder returned nil".into(),
            ));
        }
        // SAFETY: retain the checked +0 reference before its autorelease pool ends.
        Ok(unsafe { ComputeCommandEncoderRef::from_ptr(pointer) }.to_owned())
    }

    pub(super) fn submit(
        command: &CommandBufferRef,
        operation: &str,
    ) -> Result<(), MetalBf16Error> {
        command.commit();
        // Metal's waitUntilCompleted establishes terminal completion even when
        // execution was aborted with an error. Locals/borrowed inputs stay live
        // through this wait; status gates validity, not allocation retirement.
        command.wait_until_completed();
        let status = command.status();
        if status != MTLCommandBufferStatus::Completed {
            return Err(MetalBf16Error::Execution(format!(
                "{operation}: {status:?}"
            )));
        }
        Ok(())
    }
}
