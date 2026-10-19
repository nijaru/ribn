//! Completion-owned cuBLAS handle and its explicitly charged device workspace.

use crate::{
    bf16_weights::cuda::CudaTransport,
    cuda_allocation::{Allocation, CudaBf16Error},
};
use cudarc::{
    cublas::{result, sys},
    driver::CudaStream,
};
use ribn_foundation::BytePool;
use std::{mem::ManuallyDrop, sync::Arc};

pub(super) fn error(
    operation: &'static str,
    source: impl std::error::Error + Send + Sync + 'static,
) -> CudaBf16Error {
    CudaBf16Error::Transport {
        operation,
        source: Box::new(source),
    }
}

struct Resources {
    handle: Option<sys::cublasHandle_t>,
    destroy_uncertain: bool,
    workspace: Arc<Allocation<CudaTransport>>,
    stream: Arc<CudaStream>,
}

pub(super) struct Blas {
    // Set before entering cuBLAS creation/configuration. No destructor can release
    // the workspace/context under uncertain library use, including unwinding.
    resources: Option<ManuallyDrop<Resources>>,
}
// SAFETY: handle mutation is private to the unique CudaBf16Gemm owner. Shared
// references are only completion guards, which call stream drain, never cuBLAS.
// Explicit close needs unique Arc ownership; final Drop cannot race an operation.
unsafe impl Send for Blas {}
unsafe impl Sync for Blas {}

impl Blas {
    pub(super) fn prepare(
        stream: Arc<CudaStream>,
        pool: &Arc<BytePool>,
        workspace_bytes: usize,
    ) -> Result<Self, CudaBf16Error> {
        if workspace_bytes < 16 * 1024 {
            return Err(CudaBf16Error::Invalid(
                "cuBLAS workspace requires at least 16 KiB",
            ));
        }
        let workspace = Allocation::allocate(
            workspace_bytes as u64,
            CudaTransport {
                stream: Arc::clone(&stream),
            },
            pool,
            1,
        )?;
        Ok(Self {
            resources: Some(ManuallyDrop::new(Resources {
                handle: None,
                destroy_uncertain: false,
                workspace,
                stream,
            })),
        })
    }

    pub(super) fn initialize(&mut self) -> Result<(), CudaBf16Error> {
        let r = self
            .resources
            .as_deref_mut()
            .expect("preparation owns resources");
        r.stream
            .context()
            .bind_to_thread()
            .map_err(|e| error("cuBLAS bind", e))?;
        r.handle = Some(result::create_handle().map_err(|e| error("cuBLAS create", e))?);
        let handle = r.handle.expect("created handle");
        let workspace = r.workspace.pointer()?;
        let workspace = usize::try_from(workspace)
            .map_err(|_| CudaBf16Error::Invalid("cuBLAS workspace pointer exceeds host width"))?;
        if workspace % 256 != 0 {
            return Err(CudaBf16Error::Invalid(
                "cuBLAS workspace must be 256-byte aligned",
            ));
        }
        let workspace_bytes = usize::try_from(r.workspace.allocated_bytes())
            .map_err(|_| CudaBf16Error::Invalid("cuBLAS workspace exceeds host width"))?;
        // SAFETY: handle is captured before any fallible configuration. Workspace
        // and stream/context remain resident even on partial initialization.
        unsafe {
            result::set_stream(handle, r.stream.cu_stream().cast())
                .map_err(|e| error("cuBLAS stream", e))?;
            sys::cublasSetPointerMode_v2(
                handle,
                sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
            )
            .result()
            .map_err(|e| error("cuBLAS pointer mode", e))?;
            sys::cublasSetMathMode(
                handle,
                sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
            )
            .result()
            .map_err(|e| error("cuBLAS math mode", e))?;
            sys::cublasSetWorkspace_v2(handle, workspace as *mut _, workspace_bytes)
                .result()
                .map_err(|e| error("cuBLAS workspace", e))?;
        }
        Ok(())
    }

    pub(super) fn handle(&self) -> Result<sys::cublasHandle_t, CudaBf16Error> {
        self.resources
            .as_deref()
            .and_then(|r| r.handle)
            .ok_or(CudaBf16Error::Closed)
    }
    pub(super) fn stream(&self) -> &Arc<CudaStream> {
        &self.resources.as_deref().expect("live cuBLAS owner").stream
    }
    pub(super) fn close(&mut self) -> Result<(), CudaBf16Error> {
        let Some(r) = self.resources.as_deref_mut() else {
            return Ok(());
        };
        r.stream
            .synchronize()
            .map_err(|e| error("cuBLAS retirement drain", e))?;
        if let Some(handle) = r.handle {
            if r.destroy_uncertain {
                return Err(CudaBf16Error::FreeUncertain);
            }
            r.stream
                .context()
                .bind_to_thread()
                .map_err(|e| error("cuBLAS retirement bind", e))?;
            r.destroy_uncertain = true;
            // SAFETY: exclusive handle owner, after its stream completed. This
            // release is invoked at most once, even on error or unwind.
            unsafe { result::destroy_handle(handle) }.map_err(|e| error("cuBLAS destroy", e))?;
            r.handle = None;
        }
        Arc::get_mut(&mut r.workspace)
            .ok_or(CudaBf16Error::InUse)?
            .close()?;
        drop(self.resources.take().map(ManuallyDrop::into_inner));
        Ok(())
    }
}
impl Drop for Blas {
    fn drop(&mut self) {
        if self.close().is_err() {
            let _quarantined = self.resources.take();
        }
    }
}
