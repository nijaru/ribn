//! Request identity and semantic parameters.

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RequestId(u64);

impl RequestId {
    #[must_use]
    pub const fn new(value: u64) -> Option<Self> {
        if value == 0 { None } else { Some(Self(value)) }
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Sampling settings are semantic: changing them can change model output and
/// therefore they do not belong in a performance-policy snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    seed: Option<u64>,
    temperature: f32,
    top_p: f32,
    top_k: u32,
}

impl SamplingParams {
    /// # Errors
    ///
    /// Returns [`SamplingError`] when temperature or top-p is outside its
    /// valid range.
    pub fn new(
        seed: Option<u64>,
        temperature: f32,
        top_p: f32,
        top_k: u32,
    ) -> Result<Self, SamplingError> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(SamplingError::InvalidTemperature);
        }
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(SamplingError::InvalidTopP);
        }
        Ok(Self {
            seed,
            temperature,
            top_p,
            top_k,
        })
    }

    #[must_use]
    pub fn greedy(seed: Option<u64>) -> Self {
        Self {
            seed,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
        }
    }

    #[must_use]
    pub const fn seed(self) -> Option<u64> {
        self.seed
    }

    #[must_use]
    pub const fn temperature(self) -> f32 {
        self.temperature
    }

    #[must_use]
    pub const fn top_p(self) -> f32 {
        self.top_p
    }

    #[must_use]
    pub const fn top_k(self) -> u32 {
        self.top_k
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SamplingError {
    InvalidTemperature,
    InvalidTopP,
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTemperature => f.write_str("temperature must be finite and non-negative"),
            Self::InvalidTopP => f.write_str("top_p must be finite and in the range (0, 1]"),
        }
    }
}

impl std::error::Error for SamplingError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_parameters_reject_invalid_ranges() {
        assert_eq!(
            SamplingParams::new(None, -1.0, 1.0, 0),
            Err(SamplingError::InvalidTemperature)
        );
        assert_eq!(
            SamplingParams::new(None, f32::NAN, 1.0, 0),
            Err(SamplingError::InvalidTemperature)
        );
        for top_p in [0.0, -0.5, 1.5, f32::NAN] {
            assert_eq!(
                SamplingParams::new(None, 1.0, top_p, 0),
                Err(SamplingError::InvalidTopP)
            );
        }
        let greedy = SamplingParams::greedy(Some(7));
        assert!(greedy.temperature().abs() < f32::EPSILON);
        assert_eq!(greedy.seed(), Some(7));
    }
}
