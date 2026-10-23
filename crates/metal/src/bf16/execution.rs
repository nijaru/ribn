//! One failure/retirement owner shared by native commands and prepared Graphs.

use super::graph::{FfnGraphs, GraphBindings, GraphIndex, RunStatus};
use super::{MetalBf16Error, MetalBf16Executor};
use metal::{CommandBuffer, CommandQueue, Device, MTLCommandBufferStatus};
use std::sync::{MutexGuard, TryLockError};

enum Pending {
    Native(CommandBuffer),
    Graph(GraphBindings),
}

pub(super) struct Execution {
    // These native owners must survive defensive quarantine, not just Drop of
    // the outer executor. A retained command keeps its encoded resources alive.
    _device: Device,
    _queue: CommandQueue,
    graphs: Option<FfnGraphs>,
    closed: bool,
    pending: Option<Pending>,
}

impl Execution {
    pub fn new(device: &Device, queue: &CommandQueue, graphs: Option<FfnGraphs>) -> Self {
        Self {
            _device: device.clone(),
            _queue: queue.clone(),
            graphs,
            closed: false,
            pending: None,
        }
    }

    pub fn healthy(&self) -> Result<(), MetalBf16Error> {
        if self.closed || self.pending.is_some() {
            Err(MetalBf16Error::Execution(
                "execution owner faulted or pending retirement".into(),
            ))
        } else {
            Ok(())
        }
    }

    pub fn fault(&mut self) {
        self.closed = true;
    }

    pub fn begin_native(&mut self, command: &CommandBuffer) {
        // The command guard acquired a healthy, idle owner and holds its mutex
        // through encoding and terminal observation. Retain BEFORE commit.
        self.pending = Some(Pending::Native(command.clone()));
    }

    pub fn finish_native(&mut self, status: MTLCommandBufferStatus) -> bool {
        let terminal = matches!(
            status,
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        );
        if terminal {
            self.pending = None;
        }
        if status != MTLCommandBufferStatus::Completed {
            self.closed = true;
        }
        terminal
    }

    pub fn run_graph(
        &mut self,
        index: GraphIndex,
        bindings: GraphBindings,
    ) -> Result<(), MetalBf16Error> {
        self.healthy()?;
        let graphs = self
            .graphs
            .as_mut()
            .ok_or(MetalBf16Error::Invalid("FFN graphs were not prepared"))?;
        // Install real owners before the FIRST enqueue-capable FFI call. An
        // error or unwind must not return these buffers to the caller or reuse.
        self.pending = Some(Pending::Graph(bindings));
        let Some(Pending::Graph(bindings)) = &self.pending else {
            unreachable!("pending Graph installed immediately above")
        };
        match graphs.run(index, bindings) {
            RunStatus::Complete => {
                if let Err(error) = graphs.retire() {
                    self.closed = true;
                    return Err(error);
                }
                self.pending = None;
                Ok(())
            }
            RunStatus::Rejected(error) => {
                self.pending = None;
                Err(error)
            }
            RunStatus::Faulted(error) => {
                self.closed = true;
                Err(error)
            }
        }
    }

    pub fn retire(&mut self) -> Result<(), MetalBf16Error> {
        if let Some(Pending::Native(command)) = &self.pending {
            // Failure settlement and Drop QUERY status. A new wait here could
            // hang shutdown behind stalled device work. Normal submit still waits.
            if !matches!(
                command.status(),
                MTLCommandBufferStatus::NotEnqueued
                    | MTLCommandBufferStatus::Completed
                    | MTLCommandBufferStatus::Error
            ) {
                return Err(MetalBf16Error::Execution(
                    "native command not terminal".into(),
                ));
            }
        }
        // This also covers C cleanup failure following a prelaunch rejection.
        // Never infer Graph completion from another queue command or refcounts.
        if let Some(graphs) = &mut self.graphs {
            graphs.retire()?;
        }
        self.pending = None;
        // Physical retirement does not make poisoned continuation usable again.
        Ok(())
    }
}

impl MetalBf16Executor {
    pub(super) fn owner(&self) -> Result<MutexGuard<'_, Execution>, MetalBf16Error> {
        match self.execution.lock() {
            Ok(owner) => {
                owner.healthy()?;
                Ok(owner)
            }
            Err(poisoned) => {
                poisoned.into_inner().fault();
                Err(MetalBf16Error::Execution("execution mutex poisoned".into()))
            }
        }
    }

    /// Reject work or publication after this execution owner has faulted.
    /// # Errors
    /// Returns an execution error for a closed owner or unresolved submission.
    pub fn check_healthy(&self) -> Result<(), MetalBf16Error> {
        drop(self.owner()?);
        Ok(())
    }

    /// Nonblocking physical retirement; never reopens a faulted owner.
    /// # Errors
    /// Unresolved device access remains owned for another explicit cleanup retry.
    pub fn retire_failure(&self) -> Result<(), MetalBf16Error> {
        let mut owner = match self.execution.try_lock() {
            Ok(owner) => owner,
            Err(TryLockError::Poisoned(poisoned)) => {
                let mut owner = poisoned.into_inner();
                owner.fault();
                owner
            }
            Err(TryLockError::WouldBlock) => {
                return Err(MetalBf16Error::Execution(
                    "execution owner is still active".into(),
                ));
            }
        };
        owner.retire()
    }
}
