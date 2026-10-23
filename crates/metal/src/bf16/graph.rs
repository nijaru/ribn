//! Three prepared BF16 FFN projections. SDK callbacks, not queue fences, retire them.

use super::{MetalBf16Error, MetalLinearShape};
use metal::foreign_types::ForeignType;
use metal::{Buffer, CommandQueue, Device};
use std::ffi::{c_char, c_int, c_void};
use std::ptr::NonNull;

unsafe extern "C" {
    fn ribn_graph_available() -> c_int;
    fn ribn_graph_create(
        device: *mut c_void,
        queue: *mut c_void,
        rows: usize,
        columns: usize,
        inner: usize,
        outputs: usize,
        error: *mut c_char,
        capacity: usize,
    ) -> *mut c_void;
    fn ribn_graph_run(
        context: *mut c_void,
        input: *mut c_void,
        weight0: *mut c_void,
        weight1: *mut c_void,
        output0: *mut c_void,
        output1: *mut c_void,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn ribn_graph_retire(context: *mut c_void, error: *mut c_char, capacity: usize) -> c_int;
    fn ribn_graph_destroy(context: *mut c_void) -> c_int;
}

/// Geometry bound once when preparing the experimental FFN graphs.
#[derive(Clone, Copy, Debug)]
pub(super) struct FfnShape {
    pub hidden: usize,
    pub intermediate: usize,
}

impl FfnShape {
    pub fn pair(self, shape: MetalLinearShape) -> Option<GraphIndex> {
        if shape.inner != self.hidden || shape.columns != self.intermediate {
            return None;
        }
        match shape.rows {
            16 => Some(GraphIndex::Pair16),
            64 => Some(GraphIndex::Pair64),
            _ => None,
        }
    }

    pub fn down(self, shape: MetalLinearShape) -> bool {
        shape.rows == 64 && shape.inner == self.intermediate && shape.columns == self.hidden
    }
}

#[derive(Clone, Copy)]
pub(super) enum GraphIndex {
    Pair16,
    Pair64,
    Down64,
}

// These are real allocation owners, installed in Execution before FFI. No borrowed
// buffer, unpublished output or returned tensor wrapper may outlive that envelope.
pub(super) struct GraphBindings {
    pub input: Buffer,
    pub weight0: Buffer,
    pub weight1: Option<Buffer>,
    pub output0: Buffer,
    pub output1: Option<Buffer>,
}

pub(super) enum RunStatus {
    Complete,
    Rejected(MetalBf16Error),
    Faulted(MetalBf16Error),
}

fn message(bytes: &[u8]) -> MetalBf16Error {
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    MetalBf16Error::Execution(String::from_utf8_lossy(&bytes[..end]).into_owned())
}

struct Graph(NonNull<c_void>);
// SAFETY: all calls and destruction are serialized by the executor's execution
// mutex. The SDK executable has no thread affinity; callbacks access only C-owned
// completion state and never enter Rust or acquire this mutex.
unsafe impl Send for Graph {}

impl Graph {
    fn new(
        device: &Device,
        queue: &CommandQueue,
        shape: MetalLinearShape,
        outputs: usize,
    ) -> Result<Self, MetalBf16Error> {
        let mut error = [0; 1024];
        // SAFETY: live native owners, checked dimensions and writable error span.
        // Creation retains native handles but does not submit parameter execution.
        let context = unsafe {
            ribn_graph_create(
                device.as_ptr().cast(),
                queue.as_ptr().cast(),
                shape.rows,
                shape.columns,
                shape.inner,
                outputs,
                error.as_mut_ptr().cast(),
                error.len(),
            )
        };
        Ok(Self(NonNull::new(context).ok_or_else(|| message(&error))?))
    }

    fn run(&mut self, buffers: &GraphBindings) -> RunStatus {
        let mut error = [0; 1024];
        // SAFETY: Execution retains all buffers before entering this call. Inputs
        // are immutable and ready; outputs are private/disjoint. The bridge checks
        // exact byte ranges, device identity and shapes before possible enqueue.
        let status = unsafe {
            ribn_graph_run(
                self.0.as_ptr(),
                buffers.input.as_ptr().cast(),
                buffers.weight0.as_ptr().cast(),
                buffers
                    .weight1
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |b| b.as_ptr().cast()),
                buffers.output0.as_ptr().cast(),
                buffers
                    .output1
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |b| b.as_ptr().cast()),
                error.as_mut_ptr().cast(),
                error.len(),
            )
        };
        match status {
            1 => RunStatus::Complete,
            0 => RunStatus::Rejected(message(&error)),
            // Includes terminal failure, uncertain enqueue and failed cleanup.
            _ => RunStatus::Faulted(message(&error)),
        }
    }

    fn retire(&mut self) -> Result<(), MetalBf16Error> {
        let mut error = [0; 1024];
        // SAFETY: exclusive context; the bridge acquires terminal callback state
        // before releasing anything and never waits or reopens a faulted context.
        if unsafe { ribn_graph_retire(self.0.as_ptr(), error.as_mut_ptr().cast(), error.len()) }
            == 1
        {
            Ok(())
        } else {
            Err(message(&error))
        }
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: unique +1 context; destroy refuses to consume an unresolved
        // context. Execution normally retires first; refusal defensively leaks the
        // C access envelope instead of releasing device-accessible resources.
        let _ = unsafe { ribn_graph_destroy(self.0.as_ptr()) };
    }
}

pub(super) struct FfnGraphs {
    pair16: Graph,
    pair64: Graph,
    down64: Graph,
}

impl FfnGraphs {
    pub fn check_available(device: &Device) -> Result<(), MetalBf16Error> {
        // SAFETY: pure capability query, no arguments, objects or device work.
        if unsafe { ribn_graph_available() } != 1 {
            return Err(MetalBf16Error::Unsupported(
                "experimental BF16 Graph requires macOS 26 or later and a capable build SDK",
            ));
        }
        if device.name() != "Apple M3 Max" {
            return Err(MetalBf16Error::Unsupported(
                "experimental BF16 Graph is qualified only on Apple M3 Max",
            ));
        }
        Ok(())
    }

    pub fn new(
        device: &Device,
        queue: &CommandQueue,
        shape: FfnShape,
    ) -> Result<Self, MetalBf16Error> {
        Self::check_available(device)?;
        Ok(Self {
            pair16: Graph::new(
                device,
                queue,
                MetalLinearShape {
                    rows: 16,
                    columns: shape.intermediate,
                    inner: shape.hidden,
                },
                2,
            )?,
            pair64: Graph::new(
                device,
                queue,
                MetalLinearShape {
                    rows: 64,
                    columns: shape.intermediate,
                    inner: shape.hidden,
                },
                2,
            )?,
            down64: Graph::new(
                device,
                queue,
                MetalLinearShape {
                    rows: 64,
                    columns: shape.hidden,
                    inner: shape.intermediate,
                },
                1,
            )?,
        })
    }

    fn get(&mut self, index: GraphIndex) -> &mut Graph {
        match index {
            GraphIndex::Pair16 => &mut self.pair16,
            GraphIndex::Pair64 => &mut self.pair64,
            GraphIndex::Down64 => &mut self.down64,
        }
    }

    pub fn run(&mut self, index: GraphIndex, bindings: &GraphBindings) -> RunStatus {
        self.get(index).run(bindings)
    }

    pub fn retire(&mut self) -> Result<(), MetalBf16Error> {
        self.pair16.retire()?;
        self.pair64.retire()?;
        self.down64.retire()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetalBf16Executor, MetalProjectionNorm};

    #[test]
    fn prepared_graphs_and_native_tails_preserve_exact_sparse_dots() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let native = MetalBf16Executor::new().expect("native preparation");
        // Unavailable configurations must reject explicit selection, not silently
        // fall back. This branch is not evidence of Graph device execution.
        if device.name() != "Apple M3 Max" || unsafe { ribn_graph_available() } != 1 {
            assert!(matches!(
                MetalBf16Executor::with_ffn_graphs(4, 8),
                Err(MetalBf16Error::Unsupported(_))
            ));
            return;
        }
        let graph = MetalBf16Executor::with_ffn_graphs(4, 8).expect("Graph preparation");
        let left = graph.upload(&[0x3f80, 0, 0, 0].repeat(8)).unwrap();
        let right = graph.upload(&[0x4000, 0, 0, 0].repeat(8)).unwrap();
        let input = graph.upload(&[0x3f80, 0, 0, 0].repeat(16)).unwrap();
        let short = graph.allocate_shared(2).unwrap();
        let output = graph.allocate_shared(16 * 8 * 2).unwrap();
        {
            let mut owner = graph.owner().unwrap();
            let result = owner.run_graph(
                GraphIndex::Pair16,
                GraphBindings {
                    input: input.buffer.clone(),
                    weight0: left.buffer.clone(),
                    weight1: Some(right.buffer.clone()),
                    output0: short,
                    output1: Some(output),
                },
            );
            assert!(
                result.is_err(),
                "undersized output must reject before launch"
            );
            owner
                .healthy()
                .expect("clean rejection preserves owner usability");
        }
        for rows in [7, 16, 32, 64, 65] {
            let input = graph.upload(&[0x3f80, 0, 0, 0].repeat(rows)).unwrap();
            let shape = MetalLinearShape {
                rows,
                columns: 8,
                inner: 4,
            };
            // Mean square 1/4 + epsilon 3/4 = 1: RMS is exactly identity.
            let (a, b) = graph
                .rms_linear_pair(&input, &left, &right, shape, 0.75, None)
                .unwrap();
            assert_eq!(a.as_slice(), vec![0x3f80; rows * 8]);
            assert_eq!(b.as_slice(), vec![0x4000; rows * 8]);
        }
        let input = graph
            .upload(&[0x3f80, 0, 0, 0, 0, 0, 0, 0].repeat(64))
            .unwrap();
        let weight = graph
            .upload(&[0x3f80, 0, 0, 0, 0, 0, 0, 0].repeat(4))
            .unwrap();
        let down = graph
            .project_normalized(
                &input,
                &weight,
                MetalLinearShape {
                    rows: 64,
                    columns: 4,
                    inner: 8,
                },
                MetalProjectionNorm {
                    width: 4,
                    epsilon: 3.0,
                    scale: None,
                    rotary: None,
                },
            )
            .unwrap();
        assert_eq!(down.as_slice(), vec![0x3f00; 64 * 4]);
        // Different owners remain independently usable; selection is not global.
        let value = native.upload(&[0x3f80]).unwrap();
        assert_eq!(native.scale(&value, 2.0).unwrap().as_slice(), &[0x4000]);
    }
}
