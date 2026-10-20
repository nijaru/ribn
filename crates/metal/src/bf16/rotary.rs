//! Split-half rotary validation and encoding shared by standalone and chained calls.

use super::{MetalBf16Buffer, MetalBf16Error, MetalBf16Executor, THREADS};
use metal::{BufferRef, ComputeCommandEncoderRef, MTLSize};

/// BF16 rotary factors for a normalized head projection.
///
/// Projection columns must equal `heads × normalization width`; both factors
/// contain `projection rows × normalization width` elements, including both halves.
#[derive(Clone, Copy)]
pub struct MetalRotaryFactors<'a> {
    pub heads: usize,
    pub cosine: &'a MetalBf16Buffer,
    pub sine: &'a MetalBf16Buffer,
}

pub(super) struct RotaryGeometry {
    columns: u32,
    width: u32,
    pairs: u32,
}

impl RotaryGeometry {
    pub(super) fn new(
        elements: usize,
        rows: usize,
        width: usize,
        factors: MetalRotaryFactors<'_>,
    ) -> Result<Self, MetalBf16Error> {
        if rows == 0 || factors.heads == 0 || width == 0 || !width.is_multiple_of(2) {
            return Err(MetalBf16Error::Shape(
                "rotary needs positive rows/heads and an even width".into(),
            ));
        }
        let columns = factors
            .heads
            .checked_mul(width)
            .ok_or_else(|| MetalBf16Error::Shape("rotary head stride overflow".into()))?;
        let input_length = rows
            .checked_mul(columns)
            .ok_or_else(|| MetalBf16Error::Shape("rotary input overflow".into()))?;
        let factor_length = rows
            .checked_mul(width)
            .ok_or_else(|| MetalBf16Error::Shape("rotary factor overflow".into()))?;
        let pairs = rows
            .checked_mul(factors.heads)
            .and_then(|n| n.checked_mul(width / 2))
            .ok_or_else(|| MetalBf16Error::Shape("rotary pair count overflow".into()))?;
        if elements != input_length
            || factors.cosine.elements != factor_length
            || factors.sine.elements != factor_length
        {
            return Err(MetalBf16Error::Shape(
                "rotary buffers disagree with the requested shape".into(),
            ));
        }
        Ok(Self {
            columns: u32::try_from(columns)
                .map_err(|_| MetalBf16Error::Shape("columns exceed u32".into()))?,
            width: u32::try_from(width)
                .map_err(|_| MetalBf16Error::Shape("width exceeds u32".into()))?,
            pairs: u32::try_from(pairs)
                .map_err(|_| MetalBf16Error::Shape("rotary pairs exceed u32".into()))?,
        })
    }
}

impl MetalBf16Executor {
    /// Split-half BF16 rotary application with separately prepared factors.
    ///
    /// Each product rounds to BF16 before FP32 addition; both factor halves are read.
    /// # Errors
    /// Rejects invalid shapes, allocation failure or unsuccessful GPU execution.
    pub fn rotary_apply(
        &self,
        input: &MetalBf16Buffer,
        cosine: &MetalBf16Buffer,
        sine: &MetalBf16Buffer,
        rows: usize,
        heads: usize,
        width: usize,
    ) -> Result<MetalBf16Buffer, MetalBf16Error> {
        let factors = MetalRotaryFactors {
            heads,
            cosine,
            sine,
        };
        let geometry = RotaryGeometry::new(input.elements, rows, width, factors)?;
        let output = self.allocate_shared(input.elements * 2)?;
        let command = self.command()?;
        let encoder = Self::compute_encoder(&command)?;
        self.encode_rotary(&encoder, &input.buffer, &output, factors, &geometry);
        encoder.end_encoding();
        Self::submit(&command, "rotary")?;
        Ok(MetalBf16Buffer {
            buffer: output,
            elements: input.elements,
        })
    }

    pub(super) fn encode_rotary(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        output: &BufferRef,
        factors: MetalRotaryFactors<'_>,
        geometry: &RotaryGeometry,
    ) {
        encoder.set_compute_pipeline_state(&self.rotary);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(&factors.cosine.buffer), 0);
        encoder.set_buffer(2, Some(&factors.sine.buffer), 0);
        encoder.set_buffer(3, Some(output), 0);
        encoder.set_bytes(4, 4, (&raw const geometry.columns).cast());
        encoder.set_bytes(5, 4, (&raw const geometry.width).cast());
        encoder.set_bytes(6, 4, (&raw const geometry.pairs).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(geometry.pairs), 1, 1),
            MTLSize::new(u64::from(geometry.pairs).min(THREADS), 1, 1),
        );
    }
}
