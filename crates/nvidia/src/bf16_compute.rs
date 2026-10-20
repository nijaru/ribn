//! CUDA-specific BF16 compute; model roles remain above this boundary.

mod pending;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::{
    CudaBf16ComputeError, CudaBf16Elementwise, CudaBf16Gemm, CudaBf16Matrix, CudaBf16RmsNorm,
    CudaBf16RotaryApply, CudaBf16RotaryFactors,
};

use crate::cuda_allocation::CudaBf16Error;
use engine_core::{DataType, WeightTensorSpec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MatrixShape {
    rows: usize,
    cols: usize,
    bytes: u64,
    allocation_bytes: u64,
}
impl MatrixShape {
    fn new(rows: usize, cols: usize) -> Result<Self, CudaBf16Error> {
        i32::try_from(rows).map_err(|_| CudaBf16Error::Invalid("matrix rows exceed i32"))?;
        i32::try_from(cols).map_err(|_| CudaBf16Error::Invalid("matrix columns exceed i32"))?;
        if rows == 0 || cols == 0 {
            return Err(CudaBf16Error::Invalid("matrix dimensions must be positive"));
        }
        let bytes = (rows as u64)
            .checked_mul(cols as u64)
            .and_then(|n| n.checked_mul(2))
            .ok_or(CudaBf16Error::Invalid("matrix byte overflow"))?;
        let allocation_bytes = bytes
            .checked_add(15)
            .ok_or(CudaBf16Error::Invalid("matrix alignment overflow"))?
            & !15;
        usize::try_from(allocation_bytes)
            .map_err(|_| CudaBf16Error::Invalid("matrix exceeds host address space"))?;
        Ok(Self {
            rows,
            cols,
            bytes,
            allocation_bytes,
        })
    }

    #[cfg(feature = "cuda")]
    fn blas_rows(self) -> i32 {
        i32::try_from(self.rows).expect("validated matrix rows")
    }
    #[cfg(feature = "cuda")]
    fn blas_cols(self) -> i32 {
        i32::try_from(self.cols).expect("validated matrix columns")
    }

    fn validate_elementwise(self, output: Self) -> Result<u32, CudaBf16Error> {
        if self != output {
            return Err(CudaBf16Error::Invalid(
                "elementwise requires equal operand and output shapes",
            ));
        }
        let elements = self
            .rows
            .checked_mul(self.cols)
            .ok_or(CudaBf16Error::Invalid("elementwise element overflow"))?;
        u32::try_from(elements).map_err(|_| CudaBf16Error::Invalid("elementwise count exceeds u32"))
    }

    fn validate_rms(self, output: Self, width: usize, epsilon: f32) -> Result<u32, CudaBf16Error> {
        if self != output || width == 0 || !self.cols.is_multiple_of(width) {
            return Err(CudaBf16Error::Invalid(
                "RMS requires equal shapes and whole groups",
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(CudaBf16Error::Invalid(
                "RMS epsilon must be finite and positive",
            ));
        }
        let groups = self
            .rows
            .checked_mul(self.cols / width)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or(CudaBf16Error::Invalid("RMS grid exceeds CUDA limit"))?;
        Ok(groups.unsigned_abs())
    }

    fn validate_rotary(self, cos: Self, sin: Self, output: Self) -> Result<u32, CudaBf16Error> {
        if self != output
            || cos != sin
            || cos.rows != self.rows
            || !cos.cols.is_multiple_of(2)
            || !self.cols.is_multiple_of(cos.cols)
        {
            return Err(CudaBf16Error::Invalid(
                "rotary expects [M,H*D], factors [M,D], even D",
            ));
        }
        self.rows
            .checked_mul(self.cols / 2)
            .and_then(|n| n.checked_add(255))
            .and_then(|n| i32::try_from(n / 256).ok())
            .map(i32::unsigned_abs)
            .ok_or(CudaBf16Error::Invalid("rotary grid exceeds CUDA limit"))
    }

    fn validate_factors(self, sine: Self, width: usize) -> Result<u32, CudaBf16Error> {
        if self != sine || width == 0 || width > i32::MAX as usize || self.cols != width {
            return Err(CudaBf16Error::Invalid(
                "rotary factors expect matching [M,D] cosine/sine",
            ));
        }
        self.rows
            .checked_mul(width / 2)
            .and_then(|n| n.checked_add(255))
            .and_then(|n| i32::try_from(n / 256).ok())
            .map(i32::unsigned_abs)
            .ok_or(CudaBf16Error::Invalid(
                "rotary factor grid exceeds CUDA limit",
            ))
    }

    fn validate_linear(self, weight: &WeightTensorSpec, output: Self) -> Result<(), CudaBf16Error> {
        if weight.dtype() != DataType::BF16
            || weight.dimensions() != [output.cols as u64, self.cols as u64]
            || output.rows != self.rows
        {
            return Err(CudaBf16Error::Invalid(
                "linear expects [M,K] × [N,K]ᵀ → [M,N]",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotary_checks_complete_row_head_and_factor_shapes() {
        let input = MatrixShape::new(3, 24).unwrap();
        let factors = MatrixShape::new(3, 8).unwrap();
        assert_eq!(input.validate_rotary(factors, factors, input).unwrap(), 1);
        for (rows, cols) in [(1, 8), (3, 7), (3, 10), (3, 32)] {
            let bad = MatrixShape::new(rows, cols).unwrap();
            assert!(input.validate_rotary(bad, bad, input).is_err());
        }
        assert!(
            input
                .validate_rotary(factors, MatrixShape::new(1, 24).unwrap(), input)
                .is_err()
        );
        assert!(
            input
                .validate_rotary(factors, factors, MatrixShape::new(1, 72).unwrap())
                .is_err()
        );
        let huge = MatrixShape::new(i32::MAX as usize, 1024).unwrap();
        let factors = MatrixShape::new(i32::MAX as usize, 2).unwrap();
        assert!(huge.validate_rotary(factors, factors, huge).is_err());
    }

    #[test]
    fn factor_shapes_match_duplicated_half_heads_and_bound_the_grid() {
        let factors = MatrixShape::new(3, 24).unwrap();
        assert_eq!(factors.validate_factors(factors, 24).unwrap(), 1);
        for width in [0, 8, 26] {
            assert!(factors.validate_factors(factors, width).is_err());
        }
        assert!(
            factors
                .validate_factors(MatrixShape::new(1, 24).unwrap(), 24)
                .is_err()
        );
        let huge = MatrixShape::new(i32::MAX as usize, 1024).unwrap();
        assert!(huge.validate_factors(huge, 1024).is_err());
    }

    #[test]
    fn rms_validates_groups_epsilon_and_grid_before_launch() {
        let shape = MatrixShape::new(3, 512).unwrap();
        assert_eq!(shape.validate_rms(shape, 256, 1e-6).unwrap(), 6);
        for width in [0, 3, 1024] {
            assert!(shape.validate_rms(shape, width, 1e-6).is_err());
        }
        for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(shape.validate_rms(shape, 256, epsilon).is_err());
        }
        assert!(
            shape
                .validate_rms(MatrixShape::new(1, 1536).unwrap(), 256, 1e-6)
                .is_err()
        );
        let huge = MatrixShape::new(i32::MAX as usize, 2).unwrap();
        assert!(huge.validate_rms(huge, 1, 1e-6).is_err());
    }

    #[test]
    fn linear_shape_checks_transposition_and_checked_dimensions() {
        let input = MatrixShape::new(3, 7).unwrap();
        let output = MatrixShape::new(3, 11).unwrap();
        assert_eq!((input.bytes, input.allocation_bytes), (42, 48));
        let weight = WeightTensorSpec::new("w", vec![11, 7], DataType::BF16).unwrap();
        input.validate_linear(&weight, output).unwrap();
        for shape in [vec![7, 11], vec![77], vec![1, 11, 7]] {
            let bad = WeightTensorSpec::new("w", shape, DataType::BF16).unwrap();
            assert!(input.validate_linear(&bad, output).is_err());
        }
        let wrong_dtype = WeightTensorSpec::new("w", vec![11, 7], DataType::F16).unwrap();
        assert!(input.validate_linear(&wrong_dtype, output).is_err());
        assert!(
            input
                .validate_linear(&weight, MatrixShape::new(2, 11).unwrap())
                .is_err()
        );
        assert!(MatrixShape::new(0, 7).is_err());
        assert!(MatrixShape::new(3, 0).is_err());
        assert!(MatrixShape::new(i32::MAX as usize + 1, 1).is_err());
        assert!(MatrixShape::new(1, usize::MAX).is_err());
    }
}
