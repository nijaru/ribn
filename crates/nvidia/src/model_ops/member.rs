//! Unqualified member-dimension decode kernels. Constructed only by explicit opt-in.
use super::{
    CudaFunction, CudaModelKernelError, CudaSlice, CudaStream, LaunchConfig, MODEL_OPS_SOURCE,
    PushKernelArg, compile_ptx,
};
use crate::state::descriptor::BoundDescriptor;
use crate::{
    ATTN_HEAD_DIM, ATTN_KV_HEADS, ATTN_Q_HEADS, GDN_HEAD_DIM, GDN_INNER, GDN_K_HEADS, GDN_QKV_DIM,
    GDN_V_HEADS,
};
use std::sync::Arc;

pub(crate) struct MemberOps {
    stream: Arc<CudaStream>,
    conv: CudaFunction,
    gather_qk: CudaFunction,
    state: CudaFunction,
    append: CudaFunction,
    attention: CudaFunction,
}

impl MemberOps {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CudaModelKernelError> {
        let source = format!("{MODEL_OPS_SOURCE}\n{}", include_str!("member.cu"));
        let ptx =
            compile_ptx(source).map_err(|error| CudaModelKernelError::Nvrtc(error.to_string()))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        let load = |name| {
            module
                .load_function(name)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))
        };
        Ok(Self {
            conv: load("gdn_conv_members")?,
            gather_qk: load("gdn_gather_qk_members")?,
            state: load("gdn_state_members")?,
            append: load("kv_append_members")?,
            attention: load("attn_members")?,
            stream,
        })
    }

    fn check(
        &self,
        bound: &BoundDescriptor<'_>,
        slot: u32,
        recurrent: bool,
        buffers: &[(&CudaSlice<f32>, usize)],
    ) -> Result<(), CudaModelKernelError> {
        let layers = if recurrent {
            bound.layout.recurrent_layers
        } else {
            bound.layout.kv_layers
        };
        if slot as usize >= layers {
            return Err(CudaModelKernelError::ShapeOverflow);
        }
        // Access guards record on the binding stream. Merely sharing a context
        // would not order an indirect launch on a different stream.
        if !Arc::ptr_eq(&self.stream, bound.stream)
            || buffers
                .iter()
                .any(|(buffer, _)| buffer.context().as_ref() != self.stream.context().as_ref())
        {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        for &(buffer, length) in buffers {
            if buffer.len() != length {
                return Err(CudaModelKernelError::InputLength {
                    expected: length,
                    actual: buffer.len(),
                });
            }
        }
        Ok(())
    }

    pub fn conv(
        &self,
        bound: &BoundDescriptor<'_>,
        slot: u32,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaModelKernelError> {
        let m = bound.layout.members;
        self.check(
            bound,
            slot,
            true,
            &[
                (input, m * GDN_QKV_DIM),
                (weight, 4 * GDN_QKV_DIM),
                (output, m * GDN_QKV_DIM),
            ],
        )?;
        let channels = kernel_dim(GDN_QKV_DIM)?;
        // SAFETY: bound retains exclusive, validated F32 history allocations and
        // tracks their writes until after this enqueue. Each grid.y owns one row.
        unsafe {
            self.stream
                .launch_builder(&self.conv)
                .arg(bound.device)
                .arg(&slot)
                .arg(input)
                .arg(weight)
                .arg(&mut *output)
                .arg(&channels)
                .launch(config(GDN_QKV_DIM, m, 64)?)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    pub fn gather_qk(
        &self,
        bound: &BoundDescriptor<'_>,
        slot: u32,
        input: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
    ) -> Result<(), CudaModelKernelError> {
        let m = bound.layout.members;
        let width = GDN_K_HEADS * GDN_HEAD_DIM;
        self.check(
            bound,
            slot,
            true,
            &[(input, m * GDN_QKV_DIM), (q, m * width), (k, m * width)],
        )?;
        let channels = kernel_dim(GDN_QKV_DIM)?;
        let width_u32 = kernel_dim(width)?;
        // SAFETY: distinct scratch outputs, exact row lengths, no arithmetic changes.
        unsafe {
            self.stream
                .launch_builder(&self.gather_qk)
                .arg(input)
                .arg(&mut *q)
                .arg(&mut *k)
                .arg(&channels)
                .arg(&width_u32)
                .launch(config(width, m, 128)?)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one GDN state update's concrete scratch buffers"
    )]
    pub fn state(
        &self,
        bound: &BoundDescriptor<'_>,
        slot: u32,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        conv: &CudaSlice<f32>,
        decay: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaModelKernelError> {
        let m = bound.layout.members;
        self.check(
            bound,
            slot,
            true,
            &[
                (q, m * GDN_K_HEADS * GDN_HEAD_DIM),
                (k, m * GDN_K_HEADS * GDN_HEAD_DIM),
                (conv, m * GDN_QKV_DIM),
                (decay, m * GDN_V_HEADS),
                (beta, m * GDN_V_HEADS),
                (output, m * GDN_INNER),
            ],
        )?;
        let vh = kernel_dim(GDN_V_HEADS)?;
        let kh = kernel_dim(GDN_K_HEADS)?;
        let dim = kernel_dim(GDN_HEAD_DIM)?;
        let offset = kernel_dim(2 * GDN_K_HEADS * GDN_HEAD_DIM)?;
        // SAFETY: bound retains all matrices exclusively; the model validates
        // matrix sizes before binding. Members update disjoint allocations.
        unsafe {
            self.stream
                .launch_builder(&self.state)
                .arg(bound.device)
                .arg(&slot)
                .arg(q)
                .arg(k)
                .arg(conv)
                .arg(decay)
                .arg(beta)
                .arg(&mut *output)
                .arg(&vh)
                .arg(&kh)
                .arg(&dim)
                .arg(&offset)
                .launch(config(GDN_INNER, m, 128)?)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    pub fn append(
        &self,
        bound: &BoundDescriptor<'_>,
        slot: u32,
        keys: &CudaSlice<f32>,
        values: &CudaSlice<f32>,
    ) -> Result<(), CudaModelKernelError> {
        let m = bound.layout.members;
        let width = ATTN_KV_HEADS * ATTN_HEAD_DIM;
        self.check(
            bound,
            slot,
            false,
            &[(keys, m * width), (values, m * width)],
        )?;
        let heads = kernel_dim(ATTN_KV_HEADS)?;
        let dim = kernel_dim(ATTN_HEAD_DIM)?;
        // SAFETY: validated positions and owned page tables address each member's
        // exclusive F16 caches. No table version may change during this borrow.
        unsafe {
            self.stream
                .launch_builder(&self.append)
                .arg(bound.device)
                .arg(&slot)
                .arg(keys)
                .arg(values)
                .arg(&heads)
                .arg(&dim)
                .launch(config(width, m, 128)?)
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one paged attention operation's concrete scratch buffers"
    )]
    pub fn attention(
        &self,
        bound: &BoundDescriptor<'_>,
        slot: u32,
        q: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        scores: &mut CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaModelKernelError> {
        let m = bound.layout.members;
        let width = ATTN_Q_HEADS * ATTN_HEAD_DIM;
        self.check(
            bound,
            slot,
            false,
            &[(q, m * width), (gate, m * 2 * width), (output, m * width)],
        )?;
        if scores.context().as_ref() != self.stream.context().as_ref() {
            return Err(CudaModelKernelError::ContextMismatch);
        }
        let required = m
            .checked_mul(ATTN_Q_HEADS)
            .and_then(|v| v.checked_mul(bound.max_capacity))
            .ok_or(CudaModelKernelError::ShapeOverflow)?;
        if scores.len() < required {
            return Err(CudaModelKernelError::InputLength {
                expected: required,
                actual: scores.len(),
            });
        }
        let stride =
            u32::try_from(bound.max_capacity).map_err(|_| CudaModelKernelError::ShapeOverflow)?;
        let qh = kernel_dim(ATTN_Q_HEADS)?;
        let kh = kernel_dim(ATTN_KV_HEADS)?;
        let dim = kernel_dim(ATTN_HEAD_DIM)?;
        // SAFETY: each member has its own q/gate/output and score row, and
        // bound retains tables/caches until all indirect accesses are enqueued.
        unsafe {
            self.stream
                .launch_builder(&self.attention)
                .arg(bound.device)
                .arg(&slot)
                .arg(q)
                .arg(gate)
                .arg(&mut *scores)
                .arg(&mut *output)
                .arg(&stride)
                .arg(&qh)
                .arg(&kh)
                .arg(&dim)
                .launch(LaunchConfig {
                    grid_dim: ((kernel_dim(ATTN_Q_HEADS)?).div_ceil(4), kernel_dim(m)?, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|error| CudaModelKernelError::Driver(error.to_string()))?;
        }
        Ok(())
    }
}

fn config(
    elements: usize,
    members: usize,
    threads: u32,
) -> Result<LaunchConfig, CudaModelKernelError> {
    Ok(LaunchConfig {
        grid_dim: (
            kernel_dim(elements)?.div_ceil(threads),
            kernel_dim(members)?,
            1,
        ),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn kernel_dim(value: usize) -> Result<u32, CudaModelKernelError> {
    u32::try_from(value).map_err(|_| CudaModelKernelError::ShapeOverflow)
}
