//! Retained construction of the model-wide owner's complete physical working set.
use super::super::{buffers::BufferConstruction, preparation::MODEL_WIDE_PREFILL_LANES};
use super::{
    super::{
        Arc, Buffers, BytePool, CudaBf16Attention, CudaBf16ComputeError, CudaBf16Elementwise,
        CudaBf16Error, CudaBf16Gemm, CudaBf16Matrix, CudaBf16RmsNorm, CudaBf16RotaryApply,
        CudaBf16RotaryFactors, DEVICE_ONLY_STAGING, Executors, GEMM_WORKSPACE, Gemma4CudaWeights,
        Gemma4Error, Gemma4Source, HostLookups, WEIGHT_STAGING, backend,
    },
    packed::{LayerInputs, MemberBuffers, Packed},
};
use engine_nvidia::CudaBf16Stream;
use std::mem::ManuallyDrop;

// Installed before build(), including unwinding out of a fallible constructor.
// A failed drain quarantines the whole actual preparation, not its charges alone.
pub(super) struct Construction(Option<ManuallyDrop<Box<Preparation>>>);
impl Construction {
    pub fn new(preparation: Preparation) -> Self {
        Self(Some(ManuallyDrop::new(Box::new(preparation))))
    }
    pub fn build(&mut self, source: &Gemma4Source) -> Result<(), Gemma4Error> {
        self.0.as_mut().expect("live construction").build(source)
    }
    pub fn finish(mut self) -> Prepared {
        ManuallyDrop::into_inner(self.0.take().expect("completed construction")).finish()
    }
    pub fn close(&mut self) -> Result<(), CudaBf16Error> {
        if let Some(owner) = &mut self.0 {
            owner.close()?;
        }
        if let Some(owner) = self.0.take() {
            drop(ManuallyDrop::into_inner(owner));
        }
        Ok(())
    }
}
impl Drop for Construction {
    fn drop(&mut self) {
        // On failure ManuallyDrop retains the complete object and every resource.
        let _ = self.close();
    }
}

pub(super) struct Prepared {
    pub host: HostLookups,
    pub cuda: Gemma4CudaWeights,
    pub executors: Executors,
    pub scalar: Buffers,
    pub prefill: Vec<Buffers>,
    pub packed: Packed,
    pub factors: Vec<CudaBf16RotaryFactors>,
    pub stream: CudaBf16Stream,
    pub pool: Arc<BytePool>,
    pub capacity: usize,
}

pub(super) struct Preparation {
    host: HostLookups,
    stream: CudaBf16Stream,
    pool: Arc<BytePool>,
    capacity: usize,
    rows: usize,
    scratch: usize,
    cuda: Option<Gemma4CudaWeights>,
    weight_error: Option<super::super::super::Gemma4CudaLoadError>,
    compute_error: Option<CudaBf16ComputeError>,
    gemm: Option<CudaBf16Gemm>,
    rms: Option<CudaBf16RmsNorm>,
    rotary: Option<CudaBf16RotaryApply>,
    elementwise: Option<CudaBf16Elementwise>,
    attention: Option<CudaBf16Attention>,
    buffers: Vec<Buffers>,
    unfinished: BufferConstruction,
    factors: Vec<CudaBf16RotaryFactors>,
    extra: Vec<CudaBf16Matrix>,
    layers: Vec<LayerInputs>,
}

impl Preparation {
    pub fn new(
        host: HostLookups,
        stream: CudaBf16Stream,
        pool: &Arc<BytePool>,
        capacity: usize,
        rows: usize,
        scratch: usize,
    ) -> Result<Self, Gemma4Error> {
        let count = host.config.layers().len();
        let extra_count = count
            .checked_mul(3)
            .and_then(|n| n.checked_add(5))
            .ok_or_else(|| Gemma4Error::new("CUDA preparation", "matrix count overflow"))?;
        let mut buffers = Vec::new();
        let mut factors = Vec::new();
        let mut extra = Vec::new();
        let mut layers = Vec::new();
        let metadata = |error: std::collections::TryReserveError| {
            Gemma4Error::new("CUDA preparation metadata", error.to_string())
        };
        buffers
            .try_reserve_exact(working_set_rows(rows).count())
            .map_err(metadata)?;
        factors.try_reserve_exact(count).map_err(metadata)?;
        extra.try_reserve_exact(extra_count).map_err(metadata)?;
        layers.try_reserve_exact(count).map_err(metadata)?;
        Ok(Self {
            host,
            stream,
            pool: Arc::clone(pool),
            capacity,
            rows,
            scratch,
            cuda: None,
            weight_error: None,
            compute_error: None,
            gemm: None,
            rms: None,
            rotary: None,
            elementwise: None,
            attention: None,
            buffers,
            unfinished: BufferConstruction::new(),
            factors,
            extra,
            layers,
        })
    }
    fn capture<T>(&mut self, result: Result<T, CudaBf16ComputeError>) -> Result<T, Gemma4Error> {
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                self.compute_error = Some(error);
                Err(backend(
                    self.compute_error
                        .as_ref()
                        .expect("stored preparation failure"),
                ))
            }
        }
    }
    pub fn build(&mut self, source: &Gemma4Source) -> Result<(), Gemma4Error> {
        let checked = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| Gemma4Error::new("CUDA preparation", "matrix extent overflow"))
        };
        let ple_staging = checked(4, self.host.config.ple_dim())?;
        let logit_staging = checked(4, self.host.config.vocab_size())?;
        for layer in self.host.config.layers() {
            checked(self.host.config.num_attention_heads(), layer.head_dim)?;
        }
        match Gemma4CudaWeights::load_on(source, &self.stream, &self.pool, WEIGHT_STAGING) {
            Ok(weights) => self.cuda = Some(weights),
            Err(error) => {
                self.weight_error = Some(error);
                return Err(Gemma4Error::new(
                    "CUDA model-wide load",
                    self.weight_error
                        .as_ref()
                        .expect("stored load failure")
                        .to_string(),
                ));
            }
        }
        self.gemm = Some(self.capture(self.stream.gemm(&self.pool, GEMM_WORKSPACE))?);
        self.rms = Some(self.capture(self.stream.rms_norm())?);
        self.rotary = Some(self.capture(self.stream.rotary())?);
        self.elementwise = Some(self.capture(self.stream.elementwise())?);
        self.attention = Some(self.capture(self.stream.attention(&self.pool, self.scratch))?);
        for rows in working_set_rows(self.rows) {
            let buffers = self
                .unfinished
                .build(&self.host.config, rows, |rows, cols, staging| {
                    self.stream.matrix(rows, cols, &self.pool, staging)
                })
                .map_err(|error| storage(&error))?;
            self.buffers.push(buffers);
        }
        for layer in 0..self.host.config.layers().len() {
            let frequencies = (0..self.host.config.layers()[layer].head_dim / 2)
                .map(|pair| self.host.config.rotary_inverse_frequency(layer, pair))
                .collect::<Result<Vec<_>, _>>()?;
            let factors = self.capture(self.stream.rotary_factors(
                &self.pool,
                &frequencies,
                self.rows.max(2),
            ))?;
            self.factors.push(factors);
            self.matrix(2, self.host.config.ple_dim(), ple_staging)?;
            self.matrix(
                2,
                self.host.config.layers()[layer].head_dim,
                DEVICE_ONLY_STAGING,
            )?;
            self.matrix(
                2,
                self.host.config.layers()[layer].head_dim,
                DEVICE_ONLY_STAGING,
            )?;
        }
        for width in [self.buffers[0].local.width, self.buffers[0].global.width] {
            for _ in 0..2 {
                self.matrix(
                    1,
                    checked(self.host.config.num_attention_heads(), width)?,
                    DEVICE_ONLY_STAGING,
                )?;
            }
        }
        self.matrix(2, self.host.config.vocab_size(), logit_staging)?;
        Ok(())
    }
    fn matrix(&mut self, rows: usize, cols: usize, staging: usize) -> Result<(), Gemma4Error> {
        // `extra` is reserved before physical construction and remains part of
        // this load owner across every subsequent fallible allocation.
        self.extra.push(
            self.stream
                .matrix(rows, cols, &self.pool, staging)
                .map_err(|error| storage(&error))?,
        );
        Ok(())
    }
    pub fn finish(mut self: Box<Self>) -> Prepared {
        let buffers = self
            .buffers
            .pop()
            .expect("packed buffers are constructed last");
        let scalar = self.buffers.remove(0);
        let mut matrices = std::mem::take(&mut self.extra).into_iter();
        let mut next = || {
            matrices
                .next()
                .expect("complete prevalidated packed layout")
        };
        for _ in 0..self.host.config.layers().len() {
            self.layers.push(LayerInputs {
                context: next(),
                cos: next(),
                sin: next(),
            });
        }
        let local = MemberBuffers {
            query: next(),
            attended: next(),
        };
        let global = MemberBuffers {
            query: next(),
            attended: next(),
        };
        let logits = next();
        assert!(
            matrices.next().is_none(),
            "packed layout consumes every allocation"
        );
        Prepared {
            host: self.host,
            cuda: self.cuda.take().expect("prepared weight inventory"),
            executors: Executors {
                gemm: self.gemm.take().expect("prepared GEMM"),
                rms: self.rms.take().expect("prepared RMS"),
                rotary: self.rotary.take().expect("prepared rotary"),
                elementwise: self.elementwise.take().expect("prepared elementwise"),
                attention: self.attention.take().expect("prepared attention"),
            },
            scalar,
            prefill: self.buffers,
            packed: Packed {
                buffers,
                layers: self.layers,
                local,
                global,
                logits,
            },
            factors: self.factors,
            stream: self.stream,
            pool: self.pool,
            capacity: self.capacity,
        }
    }
    pub fn close(&mut self) -> Result<(), CudaBf16Error> {
        self.stream.synchronize()?;
        if let Some(error) = &mut self.compute_error {
            error.retire()?;
        }
        if let Some(error) = &mut self.weight_error {
            error.retire()?;
        }
        self.unfinished.close()?;
        for matrix in &mut self.extra {
            matrix.close()?;
        }
        for buffers in &mut self.buffers {
            buffers.close()?;
        }
        for factors in &mut self.factors {
            factors.close()?;
        }
        if let Some(gemm) = &mut self.gemm {
            gemm.close()?;
        }
        if let Some(rms) = &mut self.rms {
            rms.close()?;
        }
        if let Some(rotary) = &mut self.rotary {
            rotary.close()?;
        }
        if let Some(elementwise) = &mut self.elementwise {
            elementwise.close()?;
        }
        if let Some(attention) = &mut self.attention {
            attention.close()?;
        }
        if let Some(cuda) = &mut self.cuda {
            cuda.close()?;
        }
        Ok(())
    }
}

impl Prepared {
    pub fn close(&mut self) -> Result<(), CudaBf16Error> {
        self.stream.synchronize()?;
        self.scalar.close()?;
        for buffers in &mut self.prefill {
            buffers.close()?;
        }
        self.packed.close()?;
        for factors in &mut self.factors {
            factors.close()?;
        }
        self.executors.gemm.close()?;
        self.executors.rms.close()?;
        self.executors.rotary.close()?;
        self.executors.elementwise.close()?;
        self.executors.attention.close()?;
        self.cuda.close()
    }
}

// Scalar first and packed decode last are consumed separately by finish().
// Every intervening exact-width prefill workspace has its own charged roots.
fn working_set_rows(prefill_rows: usize) -> impl Iterator<Item = usize> {
    std::iter::once(1)
        .chain(
            MODEL_WIDE_PREFILL_LANES
                .into_iter()
                .filter(move |&rows| rows <= prefill_rows),
        )
        .chain(std::iter::once(2))
}

fn storage(error: &CudaBf16Error) -> Gemma4Error {
    Gemma4Error::new("CUDA physical preparation", error.to_string())
}

#[cfg(test)]
mod tests {
    use super::working_set_rows;

    #[test]
    fn construction_reserves_scalar_all_prefill_widths_and_separate_packed_decode() {
        for maximum in 1..=16 {
            let rows: Vec<_> = working_set_rows(maximum).collect();
            let mut expected = vec![1];
            expected.extend(2..=maximum);
            expected.push(2);
            assert_eq!(rows, expected);
            assert_eq!(rows.len(), maximum + 1);
        }
    }
}
