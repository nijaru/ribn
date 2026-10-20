//! Elementwise BF16 layer math on the same completion-owned storage as linear,
//! RMS and rotary execution. Model ordering stays in the model crate.
use super::{
    Access, CudaBf16ComputeError, CudaBf16Error, CudaBf16Matrix, Operation, Pending, Retirement,
    drain, error,
};
use cudarc::{
    driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};
use std::sync::Arc;

pub(super) struct ElementwiseKernels {
    pub(super) stream: Arc<CudaStream>,
    add: CudaFunction,
    multiply: CudaFunction,
    scale: CudaFunction,
    gelu: CudaFunction,
    softcap: CudaFunction,
    ple_combine: CudaFunction,
}

/// Prepared elementwise BF16 operations with one associated stream.
///
/// Every operation validates whole equal-shaped, ready operands and rounds each
/// output once to BF16 unless the operation documents an intermediate rounding
/// policy (soft cap). A successful call includes stream completion. No per-call
/// allocation or workspace is used.
pub struct CudaBf16Elementwise {
    kernels: Option<Arc<ElementwiseKernels>>,
    poisoned: bool,
}

impl CudaBf16Elementwise {
    /// Compile and load all elementwise kernels on one stable stream.
    /// # Errors
    /// Rejects per-thread pseudo-streams and compilation/module-loading failures.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CudaBf16ComputeError> {
        if stream.cu_stream().addr() == 0x2 {
            return Err(CudaBf16Error::Invalid("elementwise requires a stable stream").into());
        }
        let ptx = compile_ptx(concat!(
            include_str!("bf16.cuh"),
            include_str!("elementwise.cu")
        ))
        .map_err(|e| error("BF16 elementwise compilation", e))?;
        let module = stream
            .context()
            .load_module(ptx)
            .map_err(|e| error("BF16 elementwise module", e))?;
        let load = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| error("BF16 elementwise function", e))
        };
        Ok(Self {
            kernels: Some(Arc::new(ElementwiseKernels {
                stream,
                add: load("bf16_add")?,
                multiply: load("bf16_multiply")?,
                scale: load("bf16_scale")?,
                gelu: load("bf16_gelu_tanh")?,
                softcap: load("bf16_softcap")?,
                ple_combine: load("bf16_ple_combine")?,
            })),
            poisoned: false,
        })
    }

    /// BF16 `left + right` with one final rounding.
    /// # Errors
    /// Rejects mismatched, unready, in-use or foreign-stream operands.
    pub fn add_into(
        &mut self,
        left: &CudaBf16Matrix,
        right: &CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.binary_into(Binary::Add, left, right, output)
    }

    /// BF16 `left * right` with one final rounding.
    /// # Errors
    /// As [`Self::add_into`].
    pub fn multiply_into(
        &mut self,
        left: &CudaBf16Matrix,
        right: &CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.binary_into(Binary::Multiply, left, right, output)
    }

    /// Gemma per-layer embedding combine: BF16 add, then BF16 multiply by
    /// `1/sqrt(2)`; both steps round.
    /// # Errors
    /// As [`Self::add_into`].
    pub fn ple_combine_into(
        &mut self,
        context: &CudaBf16Matrix,
        token: &CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.binary_into(Binary::PleCombine, context, token, output)
    }

    /// BF16 `input * factor` with one final rounding.
    /// # Errors
    /// Rejects a non-finite factor or invalid operands.
    pub fn scale_into(
        &mut self,
        input: &CudaBf16Matrix,
        factor: f32,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        if !factor.is_finite() {
            return Err(CudaBf16Error::Invalid("elementwise scale factor must be finite").into());
        }
        self.scalar_into(Scalar::Scale(factor), input, output)
    }

    /// Gemma tanh-approximated GELU in FP32, then one BF16 rounding.
    /// # Errors
    /// As [`Self::add_into`] without the second operand.
    pub fn gelu_tanh_into(
        &mut self,
        input: &CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        self.unary_into(Unary::Gelu, input, output)
    }

    /// Final logit soft cap with the deliberate per-step BF16 rounding policy:
    /// divide, tanh and multiply each round.
    /// # Errors
    /// Rejects a non-positive/non-finite cap or invalid operands.
    pub fn softcap_into(
        &mut self,
        input: &CudaBf16Matrix,
        cap: f32,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        if !cap.is_finite() || cap <= 0.0 {
            return Err(
                CudaBf16Error::Invalid("elementwise soft cap must be finite and positive").into(),
            );
        }
        self.scalar_into(Scalar::Softcap(cap), input, output)
    }

    fn binary_into(
        &mut self,
        operation: Binary,
        left: &CudaBf16Matrix,
        right: &CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        let kernels = self.kernels.as_ref().ok_or(CudaBf16Error::Closed)?;
        if self.poisoned {
            return Err(CudaBf16Error::Poisoned.into());
        }
        left.check_open()?;
        right.check_open()?;
        output.check_open()?;
        if !left.ready() || !right.ready() {
            return Err(CudaBf16Error::NotReady.into());
        }
        if left.shape != right.shape {
            return Err(
                CudaBf16Error::Invalid("elementwise operands have different shapes").into(),
            );
        }
        let count = left.shape.validate_elementwise(output.shape)?;
        if !Arc::ptr_eq(&kernels.stream, &left.allocation.transport().stream)
            || !Arc::ptr_eq(&kernels.stream, &right.allocation.transport().stream)
            || !Arc::ptr_eq(&kernels.stream, &output.allocation.transport().stream)
        {
            return Err(
                CudaBf16Error::Invalid("elementwise requires one associated stream").into(),
            );
        }
        let x = left.allocation.pointer()?;
        let z = right.allocation.pointer()?;
        let y = output.allocation.pointer()?;
        let function = match operation {
            Binary::Add => &kernels.add,
            Binary::Multiply => &kernels.multiply,
            Binary::PleCombine => &kernels.ple_combine,
        };
        let pending = Pending::new(
            Access::Elementwise {
                operation: Operation::Elementwise {
                    program: Arc::clone(kernels),
                },
                _inputs: vec![Arc::clone(&left.allocation), Arc::clone(&right.allocation)],
                _output: Arc::clone(&output.allocation),
            },
            drain,
        );
        self.poisoned = true;
        output.state = super::MatrixState::Poisoned;
        let mut launch = kernels.stream.launch_builder(function);
        launch.arg(&x).arg(&z).arg(&y).arg(&count);
        // SAFETY: whole-call validation establishes ready equal BF16 buffers on
        // this stream with bounded count. The pending envelope retains the
        // program, stream/context and every accessed allocation before enqueue.
        let enqueue = unsafe { launch.launch(grid(count)) }
            .map(|_| ())
            .map_err(|e| error("BF16 elementwise enqueue", e));
        self.finish(pending, enqueue, output)
    }

    fn unary_into(
        &mut self,
        operation: Unary,
        input: &CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        let kernels = self.kernels.as_ref().ok_or(CudaBf16Error::Closed)?;
        if self.poisoned {
            return Err(CudaBf16Error::Poisoned.into());
        }
        input.check_open()?;
        output.check_open()?;
        if !input.ready() {
            return Err(CudaBf16Error::NotReady.into());
        }
        let count = input.shape.validate_elementwise(output.shape)?;
        if !Arc::ptr_eq(&kernels.stream, &input.allocation.transport().stream)
            || !Arc::ptr_eq(&kernels.stream, &output.allocation.transport().stream)
        {
            return Err(
                CudaBf16Error::Invalid("elementwise requires one associated stream").into(),
            );
        }
        let x = input.allocation.pointer()?;
        let y = output.allocation.pointer()?;
        let function = match operation {
            Unary::Gelu => &kernels.gelu,
        };
        let pending = Pending::new(
            Access::Elementwise {
                operation: Operation::Elementwise {
                    program: Arc::clone(kernels),
                },
                _inputs: vec![Arc::clone(&input.allocation)],
                _output: Arc::clone(&output.allocation),
            },
            drain,
        );
        self.poisoned = true;
        output.state = super::MatrixState::Poisoned;
        let mut launch = kernels.stream.launch_builder(function);
        launch.arg(&x).arg(&y).arg(&count);
        // SAFETY: as binary launch, with one validated ready input.
        let enqueue = unsafe { launch.launch(grid(count)) }
            .map(|_| ())
            .map_err(|e| error("BF16 elementwise enqueue", e));
        self.finish(pending, enqueue, output)
    }

    fn scalar_into(
        &mut self,
        operation: Scalar,
        input: &CudaBf16Matrix,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        let kernels = self.kernels.as_ref().ok_or(CudaBf16Error::Closed)?;
        if self.poisoned {
            return Err(CudaBf16Error::Poisoned.into());
        }
        input.check_open()?;
        output.check_open()?;
        if !input.ready() {
            return Err(CudaBf16Error::NotReady.into());
        }
        let count = input.shape.validate_elementwise(output.shape)?;
        if !Arc::ptr_eq(&kernels.stream, &input.allocation.transport().stream)
            || !Arc::ptr_eq(&kernels.stream, &output.allocation.transport().stream)
        {
            return Err(
                CudaBf16Error::Invalid("elementwise requires one associated stream").into(),
            );
        }
        let x = input.allocation.pointer()?;
        let y = output.allocation.pointer()?;
        let (function, parameter) = match operation {
            Scalar::Scale(factor) => (&kernels.scale, factor),
            Scalar::Softcap(cap) => (&kernels.softcap, cap),
        };
        let pending = Pending::new(
            Access::Elementwise {
                operation: Operation::Elementwise {
                    program: Arc::clone(kernels),
                },
                _inputs: vec![Arc::clone(&input.allocation)],
                _output: Arc::clone(&output.allocation),
            },
            drain,
        );
        self.poisoned = true;
        output.state = super::MatrixState::Poisoned;
        let mut launch = kernels.stream.launch_builder(function);
        launch.arg(&x).arg(&parameter).arg(&y).arg(&count);
        // SAFETY: as binary launch, with one validated ready input and a
        // preflighted finite scalar.
        let enqueue = unsafe { launch.launch(grid(count)) }
            .map(|_| ())
            .map_err(|e| error("BF16 elementwise enqueue", e));
        self.finish(pending, enqueue, output)
    }

    fn finish(
        &mut self,
        mut pending: Pending<Access>,
        enqueue: Result<(), CudaBf16Error>,
        output: &mut CudaBf16Matrix,
    ) -> Result<(), CudaBf16ComputeError> {
        if let Err(cause) = enqueue.and_then(|()| pending.retire()) {
            return Err(CudaBf16ComputeError {
                cause,
                retirement: Some(Retirement::Execution(pending)),
            });
        }
        output.state = super::MatrixState::Ready(output.shape.rows);
        self.poisoned = false;
        Ok(())
    }

    /// Drain and relinquish compiled code references. Driver-managed code unloading
    /// follows the CUDA context; no application allocation is refunded here.
    /// # Errors
    /// Rejects retained access guards or a failed associated-stream drain.
    pub fn close(&mut self) -> Result<(), CudaBf16Error> {
        if let Some(kernels) = self.kernels.as_mut() {
            let kernels = Arc::get_mut(kernels).ok_or(CudaBf16Error::InUse)?;
            kernels
                .stream
                .synchronize()
                .map_err(|e| error("BF16 elementwise close", e))?;
        }
        self.kernels = None;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Binary {
    Add,
    Multiply,
    PleCombine,
}

#[derive(Clone, Copy)]
enum Unary {
    Gelu,
}

#[derive(Clone, Copy)]
enum Scalar {
    Scale(f32),
    Softcap(f32),
}

fn grid(count: u32) -> LaunchConfig {
    const THREADS: u32 = 256;
    LaunchConfig {
        grid_dim: (count.div_ceil(THREADS), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}
