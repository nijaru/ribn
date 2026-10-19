//! CUDA-specific BF16 linear algebra; model roles remain above this boundary.

mod pending;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::{CudaBf16Gemm, CudaBf16GemmError, CudaBf16Matrix};

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
