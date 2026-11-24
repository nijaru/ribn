//! Compact quantized weight storage and row-major dot products (`F32`, `F16`, `Q8_0`, `IQ4_NL`).

use std::io::Read;

use crate::source::{Inkling2Source, TensorEncoding};
use crate::{Inkling2Error, ParameterId};

const IQ4_NL_KVALUES: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];

/// Convert IEEE-754 binary16 bits to `f32`.
#[must_use]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    match exponent {
        0 => sign * f32::from(fraction) * 2_f32.powi(-24),
        0x1f if fraction == 0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        exp => sign * (1.0 + f32::from(fraction) / 1024.0) * 2_f32.powi(i32::from(exp) - 15),
    }
}

/// Round-to-nearest-even conversion from `f32` to IEEE-754 binary16 bits.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff).cast_signed();
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        if mant == 0 {
            return sign | 0x7c00;
        }
        let payload = (mant >> 13) as u16;
        return sign | 0x7c00 | payload.max(1);
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased < -24 {
        return sign;
    }
    if unbiased < -14 {
        // Subnormal binary16
        let full_mant = mant | 0x0080_0000;
        let shift = (-14 - unbiased).cast_unsigned() + 13;
        let truncated = full_mant >> shift;
        let half = 1_u32 << (shift - 1);
        let remainder = full_mant & ((1_u32 << shift) - 1);
        let rounded = if remainder > half || (remainder == half && (truncated & 1) != 0) {
            truncated + 1
        } else {
            truncated
        };
        return sign | (rounded as u16);
    }

    let h_exp = (unbiased + 15).cast_unsigned();
    let truncated = mant >> 13;
    let remainder = mant & 0x1fff;
    let rounded = if remainder > 0x1000 || (remainder == 0x1000 && (truncated & 1) != 0) {
        truncated + 1
    } else {
        truncated
    };
    if rounded == 0x0400 {
        let next_exp = h_exp + 1;
        if next_exp >= 31 {
            return sign | 0x7c00;
        }
        return sign | ((next_exp as u16) << 10);
    }
    sign | ((h_exp as u16) << 10) | (rounded as u16)
}

/// Compact encoded row-major matrix `[rows, cols]` matching GGUF `[ne0=cols, ne1=rows]`.
#[derive(Clone, Debug)]
pub struct QuantizedMatrix {
    rows: usize,
    cols: usize,
    storage: MatrixStorage,
}

#[derive(Clone, Debug)]
enum MatrixStorage {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Q8_0(Vec<u8>),
    Iq4Nl(Vec<u8>),
}

impl QuantizedMatrix {
    /// Construct an unquantized `F32` matrix `[rows, cols]`.
    ///
    /// # Errors
    /// Rejects length mismatch.
    pub fn from_f32(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, Inkling2Error> {
        if data.len() != rows * cols {
            return Err(Inkling2Error::new(
                "f32 matrix",
                format!("expected {} elements, got {}", rows * cols, data.len()),
            ));
        }
        Ok(Self {
            rows,
            cols,
            storage: MatrixStorage::F32(data),
        })
    }

    /// Construct a `Q8_0` matrix from raw GGML block bytes.
    ///
    /// # Errors
    /// Rejects non-multiple-of-32 columns or byte length mismatch.
    pub fn from_q8_0(rows: usize, cols: usize, bytes: Vec<u8>) -> Result<Self, Inkling2Error> {
        if !cols.is_multiple_of(32) {
            return Err(Inkling2Error::new(
                "q8_0 matrix",
                "cols must be divisible by 32",
            ));
        }
        let expected = rows * (cols / 32) * 34;
        if bytes.len() != expected {
            return Err(Inkling2Error::new(
                "q8_0 matrix",
                format!("expected {expected} bytes, got {}", bytes.len()),
            ));
        }
        Ok(Self {
            rows,
            cols,
            storage: MatrixStorage::Q8_0(bytes),
        })
    }

    /// Construct an `IQ4_NL` matrix from raw GGML block bytes.
    ///
    /// # Errors
    /// Rejects non-multiple-of-32 columns or byte length mismatch.
    pub fn from_iq4_nl(rows: usize, cols: usize, bytes: Vec<u8>) -> Result<Self, Inkling2Error> {
        if !cols.is_multiple_of(32) {
            return Err(Inkling2Error::new(
                "iq4_nl matrix",
                "cols must be divisible by 32",
            ));
        }
        let expected = rows * (cols / 32) * 18;
        if bytes.len() != expected {
            return Err(Inkling2Error::new(
                "iq4_nl matrix",
                format!("expected {expected} bytes, got {}", bytes.len()),
            ));
        }
        Ok(Self {
            rows,
            cols,
            storage: MatrixStorage::Iq4Nl(bytes),
        })
    }

    /// Read a validated 2D parameter from `source` in its native encoded representation.
    ///
    /// # Errors
    /// Returns [`Inkling2Error`] on missing parameter, shape mismatch, or I/O error.
    pub fn load_from_source(
        source: &Inkling2Source,
        id: ParameterId,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<Self, Inkling2Error> {
        let encoding = source
            .parameter_encoding(id)
            .ok_or_else(|| Inkling2Error::new(format!("{id:?}"), "missing parameter"))?;
        let mut reader = source.open_parameter(id)?;
        let byte_len = usize::try_from(reader.remaining())
            .map_err(|_| Inkling2Error::new(format!("{id:?}"), "tensor exceeds host usize"))?;
        let mut raw = vec![0_u8; byte_len];
        reader
            .read_exact(&mut raw)
            .map_err(|error| Inkling2Error::new(format!("{id:?}"), error.to_string()))?;
        Self::from_raw(encoding, expected_rows, expected_cols, raw)
    }

    fn from_raw(
        encoding: TensorEncoding,
        rows: usize,
        cols: usize,
        raw: Vec<u8>,
    ) -> Result<Self, Inkling2Error> {
        match encoding {
            TensorEncoding::F32 => {
                if raw.len() != rows * cols * 4 {
                    return Err(Inkling2Error::new("f32 matrix", "unexpected byte length"));
                }
                let data = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&c| f32::from_le_bytes(c))
                    .collect();
                Self::from_f32(rows, cols, data)
            }
            TensorEncoding::F16 => {
                if raw.len() != rows * cols * 2 {
                    return Err(Inkling2Error::new("f16 matrix", "unexpected byte length"));
                }
                let data = raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&c| u16::from_le_bytes(c))
                    .collect();
                Ok(Self {
                    rows,
                    cols,
                    storage: MatrixStorage::F16(data),
                })
            }
            TensorEncoding::Q8_0 => Self::from_q8_0(rows, cols, raw),
            TensorEncoding::Iq4Nl => Self::from_iq4_nl(rows, cols, raw),
        }
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Decode a single row `row` into `out` (length `self.cols()`).
    pub fn decode_row_into(&self, row: usize, out: &mut [f32]) {
        debug_assert!(row < self.rows);
        debug_assert_eq!(out.len(), self.cols);
        match &self.storage {
            MatrixStorage::F32(data) => {
                let start = row * self.cols;
                out.copy_from_slice(&data[start..start + self.cols]);
            }
            MatrixStorage::F16(data) => {
                let start = row * self.cols;
                for (dst, &bits) in out.iter_mut().zip(&data[start..start + self.cols]) {
                    *dst = f16_to_f32(bits);
                }
            }
            MatrixStorage::Q8_0(bytes) => {
                let row_bytes = (self.cols / 32) * 34;
                decode_q8_0_row(&bytes[row * row_bytes..(row + 1) * row_bytes], out);
            }
            MatrixStorage::Iq4Nl(bytes) => {
                let row_bytes = (self.cols / 32) * 18;
                decode_iq4_nl_row(&bytes[row * row_bytes..(row + 1) * row_bytes], out);
            }
        }
    }

    /// Compute `y[0..row_limit] = self[0..row_limit, :] * x`, parallelizing large projections.
    pub fn matvec_prefix(&self, x: &[f32], row_limit: usize, out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert!(row_limit <= self.rows);
        debug_assert_eq!(out.len(), row_limit);
        matvec_storage_slice(&self.storage, self.cols, 0, row_limit, x, out);
    }

    /// Compute `y = self * x`.
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        self.matvec_prefix(x, self.rows, out);
    }

    /// Compute `y = self * x` into a newly allocated vector.
    #[must_use]
    pub fn matvec_vec(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.rows];
        self.matvec(x, &mut out);
        out
    }
}

/// Compact 3D expert tensor `[slices=n_expert, rows, cols]` matching GGUF `[ne0=cols, ne1=rows, ne2=n_expert]`.
#[derive(Clone, Debug)]
pub struct QuantizedTensor3d {
    slices: usize,
    rows: usize,
    cols: usize,
    storage: MatrixStorage,
}

impl QuantizedTensor3d {
    /// Construct an unquantized `F32` 3D expert tensor `[slices, rows, cols]`.
    ///
    /// # Errors
    /// Rejects length mismatch.
    pub fn from_f32(
        slices: usize,
        rows: usize,
        cols: usize,
        data: Vec<f32>,
    ) -> Result<Self, Inkling2Error> {
        if data.len() != slices * rows * cols {
            return Err(Inkling2Error::new(
                "f32 tensor3d",
                format!(
                    "expected {} elements, got {}",
                    slices * rows * cols,
                    data.len()
                ),
            ));
        }
        Ok(Self {
            slices,
            rows,
            cols,
            storage: MatrixStorage::F32(data),
        })
    }

    /// Read a validated 3D expert parameter from `source` in its native encoded format.
    ///
    /// # Errors
    /// Returns [`Inkling2Error`] on missing parameter, shape mismatch, or I/O error.
    pub fn load_from_source(
        source: &Inkling2Source,
        id: ParameterId,
        slices: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, Inkling2Error> {
        let encoding = source
            .parameter_encoding(id)
            .ok_or_else(|| Inkling2Error::new(format!("{id:?}"), "missing parameter"))?;
        let mut reader = source.open_parameter(id)?;
        let byte_len = usize::try_from(reader.remaining())
            .map_err(|_| Inkling2Error::new(format!("{id:?}"), "tensor exceeds host usize"))?;
        let mut raw = vec![0_u8; byte_len];
        reader
            .read_exact(&mut raw)
            .map_err(|error| Inkling2Error::new(format!("{id:?}"), error.to_string()))?;
        let flat = QuantizedMatrix::from_raw(encoding, slices * rows, cols, raw)?;
        Ok(Self {
            slices,
            rows,
            cols,
            storage: flat.storage,
        })
    }

    #[must_use]
    pub const fn slices(&self) -> usize {
        self.slices
    }

    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Compute `out = self[slice, :, :] * x`.
    pub fn matvec_slice(&self, slice: usize, x: &[f32], out: &mut [f32]) {
        debug_assert!(slice < self.slices);
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(out.len(), self.rows);
        matvec_storage_slice(
            &self.storage,
            self.cols,
            slice * self.rows,
            self.rows,
            x,
            out,
        );
    }
}

/// Read a small `F32` (or `F16`) parameter tensor into a flat `Vec<f32>`.
///
/// # Errors
/// Returns [`Inkling2Error`] if the tensor is missing, has an unexpected element count,
/// or uses a quantized block encoding where an unquantized vector is expected.
pub fn load_f32_vec(
    source: &Inkling2Source,
    id: ParameterId,
    expected_len: usize,
) -> Result<Vec<f32>, Inkling2Error> {
    let encoding = source
        .parameter_encoding(id)
        .ok_or_else(|| Inkling2Error::new(format!("{id:?}"), "missing parameter"))?;
    let mut reader = source.open_parameter(id)?;
    let byte_len = usize::try_from(reader.remaining())
        .map_err(|_| Inkling2Error::new(format!("{id:?}"), "tensor exceeds host usize"))?;
    let mut raw = vec![0_u8; byte_len];
    reader
        .read_exact(&mut raw)
        .map_err(|error| Inkling2Error::new(format!("{id:?}"), error.to_string()))?;
    match encoding {
        TensorEncoding::F32 => {
            if raw.len() != expected_len * 4 {
                return Err(Inkling2Error::new(
                    format!("{id:?}"),
                    format!(
                        "expected {} f32 elements, got {} bytes",
                        expected_len,
                        raw.len()
                    ),
                ));
            }
            Ok(raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_le_bytes(c))
                .collect())
        }
        TensorEncoding::F16 => {
            if raw.len() != expected_len * 2 {
                return Err(Inkling2Error::new(
                    format!("{id:?}"),
                    format!(
                        "expected {} f16 elements, got {} bytes",
                        expected_len,
                        raw.len()
                    ),
                ));
            }
            Ok(raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| f16_to_f32(u16::from_le_bytes(c)))
                .collect())
        }
        other => Err(Inkling2Error::new(
            format!("{id:?}"),
            format!("expected F32/F16 vector parameter, got {other:?}"),
        )),
    }
}

fn matvec_storage_slice(
    storage: &MatrixStorage,
    cols: usize,
    row_offset: usize,
    row_count: usize,
    x: &[f32],
    out: &mut [f32],
) {
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let min_parallel_work = 256 * 256;
    if threads <= 1 || row_count < 64 || row_count * cols < min_parallel_work {
        matvec_chunk(storage, cols, row_offset, x, out);
        return;
    }

    let chunk_rows = row_count.div_ceil(threads).max(32);
    std::thread::scope(|scope| {
        for (chunk_idx, out_chunk) in out.chunks_mut(chunk_rows).enumerate() {
            let start_row = row_offset + chunk_idx * chunk_rows;
            scope.spawn(move || {
                matvec_chunk(storage, cols, start_row, x, out_chunk);
            });
        }
    });
}

fn matvec_chunk(
    storage: &MatrixStorage,
    cols: usize,
    start_row: usize,
    x: &[f32],
    out: &mut [f32],
) {
    match storage {
        MatrixStorage::F32(data) => {
            for (i, dst) in out.iter_mut().enumerate() {
                let row = &data[(start_row + i) * cols..(start_row + i + 1) * cols];
                *dst = dot_f32(row, x);
            }
        }
        MatrixStorage::F16(data) => {
            for (i, dst) in out.iter_mut().enumerate() {
                let row = &data[(start_row + i) * cols..(start_row + i + 1) * cols];
                let mut sum = 0.0_f32;
                for (&w, &xv) in row.iter().zip(x) {
                    sum += f16_to_f32(w) * xv;
                }
                *dst = sum;
            }
        }
        MatrixStorage::Q8_0(bytes) => {
            let row_bytes = (cols / 32) * 34;
            for (i, dst) in out.iter_mut().enumerate() {
                let row = &bytes[(start_row + i) * row_bytes..(start_row + i + 1) * row_bytes];
                *dst = dot_q8_0(row, x);
            }
        }
        MatrixStorage::Iq4Nl(bytes) => {
            let row_bytes = (cols / 32) * 18;
            for (i, dst) in out.iter_mut().enumerate() {
                let row = &bytes[(start_row + i) * row_bytes..(start_row + i + 1) * row_bytes];
                *dst = dot_iq4_nl(row, x);
            }
        }
    }
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let (chunks_a, rem_a) = a.as_chunks::<4>();
    let (chunks_b, rem_b) = b.as_chunks::<4>();
    for (ca, cb) in chunks_a.iter().zip(chunks_b) {
        s0 += ca[0] * cb[0];
        s1 += ca[1] * cb[1];
        s2 += ca[2] * cb[2];
        s3 += ca[3] * cb[3];
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    for (&va, &vb) in rem_a.iter().zip(rem_b) {
        sum += va * vb;
    }
    sum
}

#[inline]
fn dot_q8_0(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0.0_f32;
    for (block, xb) in row.as_chunks::<34>().0.iter().zip(x.as_chunks::<32>().0) {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..34];
        let mut block_sum = 0.0_f32;
        for j in 0..32 {
            block_sum += f32::from(qs[j].cast_signed()) * xb[j];
        }
        total += scale * block_sum;
    }
    total
}

#[inline]
fn dot_iq4_nl(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0.0_f32;
    for (block, xb) in row.as_chunks::<18>().0.iter().zip(x.as_chunks::<32>().0) {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..18];
        let mut block_sum = 0.0_f32;
        for j in 0..16 {
            let byte = qs[j];
            let lo = IQ4_NL_KVALUES[usize::from(byte & 0x0f)];
            let hi = IQ4_NL_KVALUES[usize::from(byte >> 4)];
            block_sum += lo * xb[j] + hi * xb[16 + j];
        }
        total += scale * block_sum;
    }
    total
}

fn decode_q8_0_row(row: &[u8], out: &mut [f32]) {
    for (block, dst) in row
        .as_chunks::<34>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<32>().0)
    {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for j in 0..32 {
            dst[j] = scale * f32::from(block[2 + j].cast_signed());
        }
    }
}

fn decode_iq4_nl_row(row: &[u8], out: &mut [f32]) {
    for (block, dst) in row
        .as_chunks::<18>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<32>().0)
    {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for j in 0..16 {
            let byte = block[2 + j];
            dst[j] = scale * IQ4_NL_KVALUES[usize::from(byte & 0x0f)];
            dst[16 + j] = scale * IQ4_NL_KVALUES[usize::from(byte >> 4)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_gguf::dequantize_block;

    #[test]
    fn f16_roundtrip_matches_ieee754_for_normal_subnormal_and_specials() {
        for value in [
            0.0_f32, -0.0, 1.0, -1.0, 0.5, 3.140_625, 65504.0, 1e-5, -1e-6,
        ] {
            let bits = f32_to_f16(value);
            let back = f16_to_f32(bits);
            assert!((back - value).abs() <= value.abs() * 1e-3 + 1e-6);
        }
        assert!(f16_to_f32(f32_to_f16(f32::INFINITY)).is_infinite());
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn q8_0_and_iq4_nl_matvec_match_gguf_block_dequantization() {
        let x: Vec<f32> = (0_u16..64).map(|i| (f32::from(i) - 31.5) * 0.05).collect();

        // Build 2 blocks (64 cols) of Q8_0
        let mut q8_bytes = Vec::new();
        let mut q8_deq = Vec::new();
        for b in 0..2_u8 {
            let mut blk = vec![0_u8; 34];
            let scale = f32_to_f16(if b == 0 { 0.25 } else { 0.125 });
            blk[0..2].copy_from_slice(&scale.to_le_bytes());
            for j in 0_i8..32 {
                blk[2 + usize::try_from(j).unwrap()] = (j * 3 - 40).cast_unsigned();
            }
            q8_deq.extend(dequantize_block(8, &blk).unwrap());
            q8_bytes.extend(blk);
        }
        let m_q8 = QuantizedMatrix::from_q8_0(1, 64, q8_bytes).unwrap();
        let mut row_buf = vec![0.0; 64];
        m_q8.decode_row_into(0, &mut row_buf);
        assert_eq!(row_buf, q8_deq);
        let expected: f32 = q8_deq.iter().zip(&x).map(|(a, b)| a * b).sum();
        let got = m_q8.matvec_vec(&x)[0];
        assert!((got - expected).abs() < 1e-4, "q8_0 {got} vs {expected}");

        // Build 2 blocks (64 cols) of IQ4_NL
        let mut iq4_bytes = Vec::new();
        let mut iq4_deq = Vec::new();
        for b in 0..2_u8 {
            let mut blk = vec![0_u8; 18];
            let scale = f32_to_f16(if b == 0 { 0.015_625 } else { 0.031_25 });
            blk[0..2].copy_from_slice(&scale.to_le_bytes());
            for j in 0_u8..16 {
                blk[2 + usize::from(j)] = ((j + b) & 0x0f) | (((15 - j + b) & 0x0f) << 4);
            }
            iq4_deq.extend(dequantize_block(20, &blk).unwrap());
            iq4_bytes.extend(blk);
        }
        let m_iq4 = QuantizedMatrix::from_iq4_nl(1, 64, iq4_bytes).unwrap();
        m_iq4.decode_row_into(0, &mut row_buf);
        assert_eq!(row_buf, iq4_deq);
        let expected: f32 = iq4_deq.iter().zip(&x).map(|(a, b)| a * b).sum();
        let got = m_iq4.matvec_vec(&x)[0];
        assert!((got - expected).abs() < 1e-4, "iq4_nl {got} vs {expected}");
    }
}
