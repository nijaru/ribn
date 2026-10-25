//! Real-equation enclosures, not an alternate FP32 execution/acceptance policy.
use super::super::super::load;

fn rounded_shift(value: u64, shift: u32) -> u64 {
    assert!((1..=53).contains(&shift));
    let retained = value >> shift;
    let remainder = value & ((1_u64 << shift) - 1);
    let half = 1_u64 << (shift - 1);
    retained + u64::from(remainder > half || (remainder == half && retained & 1 != 0))
}

/// Direct finite F64 -> BF16 RNE. Going through F32 can double-round at a midpoint.
pub(super) fn round(value: f64) -> u16 {
    assert!(value.is_finite());
    let bits = value.to_bits();
    let sign = u16::try_from((bits >> 48) & 0x8000).unwrap();
    let exponent = i32::try_from((bits >> 52) & 0x7ff).unwrap() - 1023;
    if exponent < -134 {
        return sign;
    }
    if exponent > 127 {
        return sign | 0x7f80;
    }
    let mantissa = (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52);
    if exponent < -126 {
        let shift = u32::try_from(-exponent - 81).unwrap();
        return sign | u16::try_from(rounded_shift(mantissa, shift)).unwrap();
    }
    let significand = u16::try_from(rounded_shift(mantissa, 45)).unwrap();
    // A rounded significand of256 carries into the exponent (including infinity).
    sign | (u16::try_from(exponent + 126).unwrap() * 128 + significand)
}

/// Exact BF16 squares with an outward enclosure after *every* FP64 operation.
/// `sqrt` is correctly rounded in Rust; `powf` has no such guarantee and is unused.
pub(super) fn denominator(input: &[u16], epsilon: f32) -> (f64, f64) {
    assert!(!input.is_empty() && input.len() <= 1 << 24);
    assert!(epsilon.is_finite() && epsilon > 0.0);
    let (mut low, mut high) = (0.0_f64, 0.0_f64);
    for &bits in input {
        let x = f64::from(load(bits));
        assert!(x.is_finite());
        let square = x * x; // <=16 significand bits; BF16 extrema stay normal in F64.
        low = (low + square).next_down().max(0.0);
        high = (high + square).next_up();
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "bounded width is exactly representable in F64"
    )]
    let width = input.len() as f64;
    low = ((low / width).next_down() + f64::from(epsilon)).next_down();
    high = ((high / width).next_up() + f64::from(epsilon)).next_up();
    assert!(low > 0.0 && low <= high && high.is_finite());
    let result = (low.sqrt().next_down(), high.sqrt().next_up());
    assert!(result.0 > 0.0 && result.0 <= result.1 && result.1.is_finite());
    result
}

/// Return a word only if the entire enclosed real equation rounds to that word.
/// A zero factor preserves the product's sign by explicit IEEE convention; real
/// numbers themselves do not specify a zero sign.
pub(super) fn equation(x: u16, scale: u16, root: (f64, f64)) -> Option<u16> {
    assert!(root.0 > 0.0 && root.0 <= root.1 && root.1.is_finite());
    let factor = f64::from(load(x)) * f64::from(load(scale));
    assert!(factor.is_finite()); // BF16 * BF16 is exact in F64.
    let (low, high) = if factor == 0.0 {
        (factor, factor)
    } else if factor > 0.0 {
        ((factor / root.1).next_down(), (factor / root.0).next_up())
    } else {
        ((factor / root.0).next_down(), (factor / root.1).next_up())
    };
    assert!(low.is_finite() && low <= high && high.is_finite());
    let (a, b) = (round(low), round(high));
    (a == b).then_some(a)
}

#[test]
fn direct_rounding_covers_midpoints_boundaries_and_double_rounding() {
    for (tie, lower, upper) in [
        (1.003_906_25_f64, 0x3f80, 0x3f81),
        (1.011_718_75, 0x3f81, 0x3f82),
        (1.996_093_75, 0x3fff, 0x4000),
        (2_f64.powi(-134), 0, 1),
        (127.5 * 2_f64.powi(-133), 0x007f, 0x0080),
        (2_f64.powi(128) - 2_f64.powi(119), 0x7f7f, 0x7f80),
    ] {
        let even = if lower & 1 == 0 { lower } else { upper };
        for (value, word) in [
            (tie.next_down(), lower),
            (tie, even),
            (tie.next_up(), upper),
        ] {
            assert_eq!(round(value), word);
            assert_eq!(round(-value), word | 0x8000);
        }
    }
    let value = 1.0 + 2_f64.powi(-8) + 2_f64.powi(-25);
    assert_eq!(round(value), 0x3f81);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "intentionally demonstrate incorrect double-rounding through F32"
    )]
    let double_rounded = super::super::super::bf16(value as f32);
    assert_eq!(double_rounded, 0x3f80);
    assert_eq!(round(0.0), 0);
    assert_eq!(round(-0.0), 0x8000);
    assert_eq!(round(f64::MIN_POSITIVE), 0);
}

#[test]
fn equation_witnesses_keep_signs_and_midpoint_uncertainty() {
    let root = denominator(&[0x3f80, 0xbf80], 3.0);
    assert!(root.0 <= 2.0 && root.1 >= 2.0);
    assert_eq!(equation(0x3f80, 0x4000, root), Some(0x3f80));
    assert_eq!(equation(0xbf80, 0x4000, root), Some(0xbf80));
    assert_eq!(equation(0x3f80, 0x8000, root), Some(0x8000));
    assert_eq!(equation(0x8000, 0xbf80, root), Some(0));
    for scale in [0x3f80, 0xbf80] {
        for factor in [0x3f80, 0x3f81] {
            // Division endpoints straddle successive BF16 midpoints of each parity.
            let tie = if factor == 0x3f80 {
                1.003_906_25
            } else {
                1.011_718_75
            };
            let center = f64::from(load(factor)) / tie;
            assert_eq!(
                equation(factor, scale, (center.next_down(), center.next_up())),
                None
            );
        }
    }
    let tiny = denominator(&[1, 0x3f80], f32::MIN_POSITIVE);
    assert!(tiny.0 <= 0.5_f64.sqrt() && tiny.1 >= 0.5_f64.sqrt());
    assert!(equation(1, 0x3f80, tiny).is_some());
}
