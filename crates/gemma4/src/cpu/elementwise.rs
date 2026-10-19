//! Elementwise BF16 operations with the pinned eager rounding boundaries.

use crate::Gemma4Error;
use crate::cpu::linear::bf16;
use crate::cpu::try_vector;

fn load(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// BF16 addition: FP32 operands, one BF16 round-to-nearest-even.
/// # Errors
/// Rejects disagreeing lengths.
pub fn add(left: &[u16], right: &[u16]) -> Result<Vec<u16>, Gemma4Error> {
    elementwise(left, right, |a, b| a + b)
}

/// BF16 elementwise multiplication, one rounding.
/// # Errors
/// Rejects disagreeing lengths.
pub fn multiply(left: &[u16], right: &[u16]) -> Result<Vec<u16>, Gemma4Error> {
    elementwise(left, right, |a, b| a * b)
}

fn elementwise(
    left: &[u16],
    right: &[u16],
    operation: impl Fn(f32, f32) -> f32,
) -> Result<Vec<u16>, Gemma4Error> {
    if left.len() != right.len() {
        return Err(Gemma4Error::new(
            "CPU elementwise",
            "operands have different lengths",
        ));
    }
    let mut output = try_vector(left.len(), 0_u16)?;
    for (slot, (&a, &b)) in output.iter_mut().zip(left.iter().zip(right)) {
        *slot = bf16(operation(load(a), load(b)));
    }
    Ok(output)
}

/// Multiply every element by one FP32 scalar with a single BF16 rounding.
/// # Errors
/// Rejects a non-finite scalar or a value that cannot be represented.
pub fn scale(input: &[u16], factor: f32) -> Result<Vec<u16>, Gemma4Error> {
    if !factor.is_finite() {
        return Err(Gemma4Error::new("CPU scale", "factor must be finite"));
    }
    let mut output = try_vector(input.len(), 0_u16)?;
    for (slot, &value) in output.iter_mut().zip(input) {
        *slot = bf16(load(value) * factor);
    }
    Ok(output)
}

/// Gemma's tanh-approximated GELU: FP32 evaluation, one BF16 rounding.
///
/// `u = sqrt(2/π) · x · (1 + 0.044715·x²)`, `0.5·x·(1 + tanh(u))`, matching the
/// pinned activation's algebraic form rather than a fused polynomial.
/// # Errors
/// Rejects a value that is not finite.
pub fn gelu_tanh(input: &[u16]) -> Result<Vec<u16>, Gemma4Error> {
    const ALPHA: f32 = 0.797_884_6;
    const CUBIC: f32 = 0.044_715;
    let mut output = try_vector(input.len(), 0_u16)?;
    for (slot, &value) in output.iter_mut().zip(input) {
        let x = load(value);
        if !x.is_finite() {
            return Err(Gemma4Error::new("CPU GELU", "input must be finite"));
        }
        let inner = ALPHA * x * (1.0 + CUBIC * x * x);
        *slot = bf16(0.5 * x * (1.0 + inner.tanh()));
    }
    Ok(output)
}

/// Final logit soft cap: divide, `tanh`, multiply, with BF16 rounding each step.
/// # Errors
/// Rejects a non-positive/non-finite cap or a non-finite logit.
pub fn softcap(input: &[u16], cap: f32) -> Result<Vec<u16>, Gemma4Error> {
    if !cap.is_finite() || cap <= 0.0 {
        return Err(Gemma4Error::new(
            "CPU soft cap",
            "cap must be finite and positive",
        ));
    }
    let mut output = try_vector(input.len(), 0_u16)?;
    for (slot, &value) in output.iter_mut().zip(input) {
        let scaled = load(value);
        if !scaled.is_finite() {
            return Err(Gemma4Error::new("CPU soft cap", "logit must be finite"));
        }
        let divided = bf16(scaled / cap);
        let hyperbolic = bf16(load(divided).tanh());
        *slot = bf16(load(hyperbolic) * cap);
    }
    Ok(output)
}

/// Per-layer embedding combine: `(context + token) × 2^-1/2`, each step rounded.
/// # Errors
/// Rejects disagreeing lengths.
pub fn ple_combine(context: &[u16], token: &[u16]) -> Result<Vec<u16>, Gemma4Error> {
    let sum = add(context, token)?;
    scale(&sum, std::f32::consts::FRAC_1_SQRT_2)
}
