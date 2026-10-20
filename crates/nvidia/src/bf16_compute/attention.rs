//! Causal grouped-query BF16 attention on completion-owned storage.
//!
//! Scores and softmax weights are BF16; softmax and value accumulation are FP32
//! and walk visible keys in ascending order. This is a correctness-first
//! component, not a flash-attention or fused-prefill implementation.
use super::{
    Access, CudaBf16ComputeError, CudaBf16Error, CudaBf16Matrix, Operation, Pending, Retirement,
    drain, error,
};
use cudarc::{
    driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};
use ribn_foundation::BytePool;
use std::sync::Arc;

pub(super) struct AttentionKernels {
    pub(super) stream: Arc<CudaStream>,
    scores: CudaFunction,
    softmax: CudaFunction,
    values: CudaFunction,
}

/// Absolute starts of the query and key spans.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionSpan {
    pub query_start: usize,
    pub key_start: usize,
    /// Rows of the key/value storage actually visible, permitting a cache with
    /// unused capacity beyond the current sequence.
    pub key_sequence: usize,
}

/// One layer's attention geometry and mask policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionGeometry {
    pub heads: usize,
    pub kv_heads: usize,
    pub width: usize,
    /// Sliding window in positions; `None` is full causal attention.
    pub window: Option<usize>,
}

/// Prepared attention with one reusable BF16 score/weight scratch matrix.
///
/// The scratch bound is fixed at construction and charged to the supplied pool.
/// A call validates all shapes, the key span, geometry and scratch capacity before
/// invalidating output; successful return includes stream completion.
pub struct CudaBf16Attention {
    kernels: Option<Arc<AttentionKernels>>,
    scratch: CudaBf16Matrix,
    poisoned: bool,
}

impl CudaBf16Attention {
    /// Compile the kernels and allocate a reusable scratch matrix of
    /// `scratch_entries` BF16 elements.
    /// # Errors
    /// Rejects a per-thread pseudo-stream, zero scratch, compilation, module or
    /// allocation failures.
    pub fn new(
        stream: Arc<CudaStream>,
        pool: &Arc<BytePool>,
        scratch_entries: usize,
    ) -> Result<Self, CudaBf16ComputeError> {
        if stream.cu_stream().addr() == 0x2 {
            return Err(CudaBf16Error::Invalid("attention requires a stable stream").into());
        }
        if scratch_entries == 0 {
            return Err(CudaBf16Error::Invalid("attention scratch must be positive").into());
        }
        let ptx = compile_ptx(concat!(
            include_str!("bf16.cuh"),
            include_str!("attention.cu")
        ))
        .map_err(|e| error("BF16 attention compilation", e))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| error("BF16 attention module", e))?;
        let load = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| error("BF16 attention function", e))
        };
        let kernels = AttentionKernels {
            stream,
            scores: load("bf16_attention_scores")?,
            softmax: load("bf16_attention_softmax")?,
            values: load("bf16_attention_values")?,
        };
        let scratch =
            CudaBf16Matrix::allocate(1, scratch_entries, Arc::clone(&kernels.stream), pool, 1)?;
        Ok(Self {
            kernels: Some(Arc::new(kernels)),
            scratch,
            poisoned: false,
        })
    }

    #[must_use]
    pub fn scratch_entries(&self) -> usize {
        self.scratch.shape().1
    }

    /// Attend `query` positions against the supplied key/value span.
    ///
    /// `query` is `[sequence, heads × width]`, keys/values
    /// `[key_sequence, kv_heads × width]`. Positions are absolute; the caller
    /// supplies every key the cache policy requires. Containment is checked, not
    /// coverage, mirroring the host contract.
    /// # Errors
    /// Rejects invalid geometry/spans, mismatched or unready buffers, insufficient
    /// scratch, foreign streams and failed enqueue/completion.
    #[expect(
        clippy::too_many_lines,
        reason = "one validation-and-launch sequence whose checks must stay together"
    )]
    pub fn attend_into(
        &mut self,
        query: &CudaBf16Matrix,
        keys: &CudaBf16Matrix,
        values: &CudaBf16Matrix,
        geometry: AttentionGeometry,
        span: AttentionSpan,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        let kernels = self.kernels.as_ref().ok_or(CudaBf16Error::Closed)?;
        if self.poisoned {
            return Err(CudaBf16Error::Poisoned.into());
        }
        query.check_open()?;
        keys.check_open()?;
        values.check_open()?;
        self.scratch.check_open()?;
        output.check_open()?;
        if !query.ready() || !keys.ready() || !values.ready() {
            return Err(CudaBf16Error::NotReady.into());
        }
        let AttentionSpan {
            query_start,
            key_start,
            key_sequence,
        } = span;
        let AttentionGeometry {
            heads,
            kv_heads,
            width,
            window,
        } = geometry;
        if heads == 0 || kv_heads == 0 || width == 0 || !heads.is_multiple_of(kv_heads) {
            return Err(
                CudaBf16Error::Invalid("attention requires positive whole head groups").into(),
            );
        }
        if window == Some(0) {
            return Err(CudaBf16Error::Invalid("attention window must be positive").into());
        }
        let sequence = query.shape.rows;
        let head_step = heads
            .checked_mul(width)
            .ok_or(CudaBf16Error::Invalid("attention head stride overflow"))?;
        let kv_step = kv_heads
            .checked_mul(width)
            .ok_or(CudaBf16Error::Invalid("attention key stride overflow"))?;
        if sequence == 0
            || key_sequence == 0
            || key_sequence > keys.shape.rows
            || key_sequence > values.shape.rows
            || query.shape.cols != head_step
            || keys.shape.cols != kv_step
            || values.shape != keys.shape
            || output.shape != query.shape
        {
            return Err(CudaBf16Error::Invalid(
                "attention buffers disagree with the requested geometry",
            )
            .into());
        }
        let last_query = query_start
            .checked_add(sequence)
            .ok_or(CudaBf16Error::Invalid("attention query positions overflow"))?;
        let last_key = key_start
            .checked_add(key_sequence)
            .ok_or(CudaBf16Error::Invalid("attention key positions overflow"))?;
        if query_start < key_start || last_query > last_key {
            return Err(CudaBf16Error::Invalid(
                "attention queries must not precede or outrun the key span",
            )
            .into());
        }
        let needed = sequence
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(key_sequence))
            .ok_or(CudaBf16Error::Invalid("attention scratch count overflow"))?;
        if needed > self.scratch.shape.cols {
            return Err(CudaBf16Error::Invalid("attention scratch is too small").into());
        }
        if !Arc::ptr_eq(&kernels.stream, &query.allocation.transport().stream)
            || !Arc::ptr_eq(&kernels.stream, &keys.allocation.transport().stream)
            || !Arc::ptr_eq(&kernels.stream, &values.allocation.transport().stream)
            || !Arc::ptr_eq(&kernels.stream, &self.scratch.allocation.transport().stream)
            || !Arc::ptr_eq(&kernels.stream, &output.allocation.transport().stream)
        {
            return Err(CudaBf16Error::Invalid("attention requires one associated stream").into());
        }
        let sequence_u32 = u32::try_from(sequence)
            .map_err(|_| CudaBf16Error::Invalid("attention sequence exceeds u32"))?;
        let key_sequence_u32 = u32::try_from(key_sequence)
            .map_err(|_| CudaBf16Error::Invalid("attention key sequence exceeds u32"))?;
        let heads_u32 = u32::try_from(heads)
            .map_err(|_| CudaBf16Error::Invalid("attention heads exceed u32"))?;
        let kv_heads_u32 = u32::try_from(kv_heads)
            .map_err(|_| CudaBf16Error::Invalid("attention KV heads exceed u32"))?;
        let width_u32 = u32::try_from(width)
            .map_err(|_| CudaBf16Error::Invalid("attention width exceeds u32"))?;
        let query_start_u32 = u32::try_from(query_start)
            .map_err(|_| CudaBf16Error::Invalid("attention query start exceeds u32"))?;
        let key_start_u32 = u32::try_from(key_start)
            .map_err(|_| CudaBf16Error::Invalid("attention key start exceeds u32"))?;
        let window_i32 = match window {
            None => 0,
            Some(window) => i32::try_from(window)
                .map_err(|_| CudaBf16Error::Invalid("attention window exceeds i32"))?,
        };
        let scores_total = u32::try_from(needed)
            .map_err(|_| CudaBf16Error::Invalid("attention scratch exceeds u32"))?;
        let softmax_total = sequence_u32
            .checked_mul(heads_u32)
            .ok_or(CudaBf16Error::Invalid("attention softmax count overflow"))?;
        let values_total = softmax_total
            .checked_mul(width_u32)
            .ok_or(CudaBf16Error::Invalid("attention value count overflow"))?;
        let query_pointer = query.allocation.pointer()?;
        let key_pointer = keys.allocation.pointer()?;
        let value_pointer = values.allocation.pointer()?;
        let scratch_pointer = self.scratch.allocation.pointer()?;
        let output_pointer = output.allocation.pointer()?;

        let pending = Pending::new(
            Access::Attention {
                operation: Operation::Attention {
                    program: Arc::clone(kernels),
                },
                _query: Arc::clone(&query.allocation),
                _keys: Arc::clone(&keys.allocation),
                _values: Arc::clone(&values.allocation),
                _scratch: Arc::clone(&self.scratch.allocation),
                _output: Arc::clone(&output.allocation),
            },
            drain,
        );
        self.poisoned = true;
        output.state = super::MatrixState::Poisoned;

        let enqueue = (|| -> Result<(), CudaBf16Error> {
            let mut launch = kernels.stream.launch_builder(&kernels.scores);
            launch
                .arg(&query_pointer)
                .arg(&key_pointer)
                .arg(&scratch_pointer)
                .arg(&sequence_u32)
                .arg(&key_sequence_u32)
                .arg(&heads_u32)
                .arg(&kv_heads_u32)
                .arg(&width_u32)
                .arg(&query_start_u32)
                .arg(&key_start_u32)
                .arg(&window_i32)
                .arg(&scores_total);
            // SAFETY: preflight established ready same-stream BF16 buffers and
            // bounded counts; the pending envelope retains every accessed
            // allocation and the program before this launch.
            unsafe { launch.launch(grid(scores_total)) }
                .map_err(|e| error("BF16 attention score enqueue", e))?;

            let mut launch = kernels.stream.launch_builder(&kernels.softmax);
            launch
                .arg(&scratch_pointer)
                .arg(&sequence_u32)
                .arg(&key_sequence_u32)
                .arg(&heads_u32)
                .arg(&query_start_u32)
                .arg(&key_start_u32)
                .arg(&window_i32);
            // SAFETY: as the score launch; the same retained scratch is the only
            // buffer accessed.
            unsafe { launch.launch(grid(softmax_total)) }
                .map_err(|e| error("BF16 attention softmax enqueue", e))?;

            let mut launch = kernels.stream.launch_builder(&kernels.values);
            launch
                .arg(&scratch_pointer)
                .arg(&value_pointer)
                .arg(&output_pointer)
                .arg(&sequence_u32)
                .arg(&key_sequence_u32)
                .arg(&heads_u32)
                .arg(&kv_heads_u32)
                .arg(&width_u32)
                .arg(&query_start_u32)
                .arg(&key_start_u32)
                .arg(&window_i32)
                .arg(&values_total);
            // SAFETY: as the score launch; query is not read by this stage.
            unsafe { launch.launch(grid(values_total)) }
                .map_err(|e| error("BF16 attention value enqueue", e))?;
            Ok(())
        })();

        let mut pending = pending;
        if let Err(cause) = enqueue.and_then(|()| pending.retire()) {
            return Err(CudaBf16ComputeError {
                cause,
                retirement: Some(Retirement::Execution(pending)),
            });
        }
        output.state = super::MatrixState::Ready;
        self.poisoned = false;
        Ok(())
    }

    /// Drain, relinquish compiled code and release the scratch grant.
    /// # Errors
    /// Rejects retained accesses or a failed associated-stream drain.
    pub fn close(&mut self) -> Result<(), CudaBf16Error> {
        if let Some(kernels) = self.kernels.as_mut() {
            let kernels = Arc::get_mut(kernels).ok_or(CudaBf16Error::InUse)?;
            kernels
                .stream
                .synchronize()
                .map_err(|e| error("BF16 attention close", e))?;
        }
        self.kernels = None;
        self.scratch.close()
    }
}

fn grid(count: u32) -> LaunchConfig {
    const THREADS: u32 = 256;
    LaunchConfig {
        grid_dim: (count.div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}
