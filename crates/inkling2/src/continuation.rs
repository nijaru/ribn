//! Pure logical continuation state ranges and byte accounting for Inkling 2.
//!
//! Each logical state slot (48 for Inkling 2 Mini) owns independent KV cache
//! history and four-tap short-convolution history. Depth-residual block banks
//! and cross-pass loop message banks are forward scratch only and are not part
//! of persistent continuation state.

use std::ops::Range;

use crate::Inkling2Error;
use crate::config::{AttentionKind, Inkling2Config};

/// Element precision used for persistent K/V caches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvCachePrecision {
    /// IEEE-754 half precision (2 bytes/element), matching the pinned llama.cpp CUDA oracle.
    F16,
    /// Single precision (4 bytes/element), useful for high-precision diagnostics.
    F32,
}

impl KvCachePrecision {
    #[must_use]
    pub const fn element_bytes(self) -> u64 {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Logical persistent-state step for one state slot (`0..config.state_slot_count()`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SlotContinuationStep {
    slot: usize,
    physical_layer: usize,
    kind: AttentionKind,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
    rel_extent: usize,
    input: Range<usize>,
    before_kv: Range<usize>,
    attention_kv: Range<usize>,
    after_kv: Range<usize>,
    kv_bytes_per_token: u64,
    conv_state_bytes: u64,
}

impl SlotContinuationStep {
    #[must_use]
    pub const fn slot(&self) -> usize {
        self.slot
    }

    #[must_use]
    pub const fn physical_layer(&self) -> usize {
        self.physical_layer
    }

    #[must_use]
    pub const fn kind(&self) -> AttentionKind {
        self.kind
    }

    #[must_use]
    pub fn before_kv(&self) -> Range<usize> {
        self.before_kv.clone()
    }

    #[must_use]
    pub fn attention_kv(&self) -> Range<usize> {
        self.attention_kv.clone()
    }

    #[must_use]
    pub fn after_kv(&self) -> Range<usize> {
        self.after_kv.clone()
    }

    #[must_use]
    pub const fn kv_bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token
    }

    /// Fixed F32 convolution history bytes `(l_cache - 1) * (2*kv_dim + 2*hidden) * 4`.
    #[must_use]
    pub const fn conv_state_bytes(&self) -> u64 {
        self.conv_state_bytes
    }

    /// Exact causal/windowed key range visible to a query at `query_pos`.
    ///
    /// # Errors
    /// Rejects `query_pos` outside this step's accepted input range.
    pub fn visible(&self, query_pos: usize) -> Result<Range<usize>, Inkling2Error> {
        if !self.input.contains(&query_pos) {
            return Err(Inkling2Error::new(
                "query position",
                "outside accepted input range",
            ));
        }
        let start = match self.kind {
            AttentionKind::Full => 0,
            AttentionKind::Sliding => query_pos.saturating_sub(self.window - 1),
        };
        Ok(start..query_pos + 1)
    }

    /// Relative-position bias lookup distance `query_pos - key_pos` when within
    /// `0 <= distance < rel_extent`; `None` indicates zero bias.
    #[must_use]
    pub fn relative_distance(&self, query_pos: usize, key_pos: usize) -> Option<usize> {
        let distance = query_pos.checked_sub(key_pos)?;
        (distance < self.rel_extent).then_some(distance)
    }
}

/// Aggregate logical continuation ranges and payload bytes across all state slots.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContinuationStep {
    input: Range<usize>,
    slots: Vec<SlotContinuationStep>,
    retained_kv_bytes: u64,
    conv_state_bytes: u64,
}

impl ContinuationStep {
    #[must_use]
    pub fn input(&self) -> Range<usize> {
        self.input.clone()
    }

    #[must_use]
    pub fn slots(&self) -> &[SlotContinuationStep] {
        &self.slots
    }

    #[must_use]
    pub const fn retained_kv_bytes(&self) -> u64 {
        self.retained_kv_bytes
    }

    #[must_use]
    pub const fn conv_state_bytes(&self) -> u64 {
        self.conv_state_bytes
    }

    #[must_use]
    pub const fn total_retained_bytes(&self) -> Option<u64> {
        self.retained_kv_bytes.checked_add(self.conv_state_bytes)
    }
}

impl Inkling2Config {
    /// Compute exact per-slot persistent KV and short-convolution state ranges
    /// for advancing one sequence from `prefix` by `tokens` within `capacity`.
    ///
    /// # Errors
    /// Rejects zero progress, overflow, or capacity outside `context_length`.
    pub fn continuation_step(
        &self,
        prefix: usize,
        tokens: usize,
        capacity: usize,
        kv_precision: KvCachePrecision,
    ) -> Result<ContinuationStep, Inkling2Error> {
        let end = prefix
            .checked_add(tokens)
            .filter(|_| tokens > 0)
            .ok_or_else(|| {
                Inkling2Error::new(
                    "continuation input",
                    "positive, non-overflowing progress is required",
                )
            })?;
        if capacity > self.context_length() || end > capacity {
            return Err(Inkling2Error::new(
                "continuation capacity",
                "input exceeds sequence capacity or model context length",
            ));
        }

        let elem_bytes = kv_precision.element_bytes();
        let conv_history = (self.conv_kernel() - 1) as u64;
        let mut slots = Vec::with_capacity(self.state_slot_count());
        let mut retained_kv_bytes = 0_u64;
        let mut conv_state_bytes = 0_u64;

        for slot in 0..self.state_slot_count() {
            let ip = self
                .slot_physical_layer(slot)
                .ok_or_else(|| Inkling2Error::new("state slot", "missing physical mapping"))?;
            let geom = &self.physical_layers()[ip];
            let window = self.sliding_window();

            let before_start = match geom.kind {
                AttentionKind::Full => 0,
                AttentionKind::Sliding => prefix.saturating_sub(window - 1),
            };
            let after_start = match geom.kind {
                AttentionKind::Full => 0,
                AttentionKind::Sliding => end.saturating_sub(window - 1),
            };
            let before_kv = before_start..prefix;
            let attention_kv = before_start..end;
            let after_kv = after_start..end;

            let kv_bytes_per_token = 2_u64
                .checked_mul(geom.kv_dim() as u64)
                .and_then(|v| v.checked_mul(elem_bytes))
                .ok_or_else(|| Inkling2Error::new("kv accounting", "byte size overflow"))?;
            let slot_retained_kv = (after_kv.len() as u64)
                .checked_mul(kv_bytes_per_token)
                .ok_or_else(|| Inkling2Error::new("kv accounting", "byte size overflow"))?;

            let channels = self
                .conv_channels_for_layer(ip)
                .ok_or_else(|| Inkling2Error::new("conv accounting", "missing layer"))?
                as u64;
            let slot_conv_bytes = conv_history
                .checked_mul(channels)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| Inkling2Error::new("conv accounting", "byte size overflow"))?;

            retained_kv_bytes = retained_kv_bytes
                .checked_add(slot_retained_kv)
                .ok_or_else(|| Inkling2Error::new("kv accounting", "total byte overflow"))?;
            conv_state_bytes = conv_state_bytes
                .checked_add(slot_conv_bytes)
                .ok_or_else(|| Inkling2Error::new("conv accounting", "total byte overflow"))?;

            slots.push(SlotContinuationStep {
                slot,
                physical_layer: ip,
                kind: geom.kind,
                kv_heads: geom.kv_heads,
                head_dim: geom.head_dim,
                window,
                rel_extent: geom.rel_extent,
                input: prefix..end,
                before_kv,
                attention_kv,
                after_kv,
                kv_bytes_per_token,
                conv_state_bytes: slot_conv_bytes,
            });
        }

        Ok(ContinuationStep {
            input: prefix..end,
            slots,
            retained_kv_bytes,
            conv_state_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tests::mini_loop_metadata;

    #[test]
    fn continuation_tracks_48_slots_and_sliding_window_boundary() {
        let cfg = Inkling2Config::from_metadata(&mini_loop_metadata()).expect("mini config");
        let step = cfg
            .continuation_step(1023, 2, 2048, KvCachePrecision::F16)
            .expect("boundary step");
        assert_eq!(step.slots().len(), 48);

        // Slot 0 is SWA (window=1024); prefix=1023..1025 spans the boundary!
        let swa = &step.slots()[0];
        assert_eq!(swa.before_kv(), 0..1023);
        assert_eq!(swa.attention_kv(), 0..1025);
        assert_eq!(swa.after_kv(), 2..1025);
        assert_eq!(swa.visible(1023).unwrap(), 0..1024);
        assert_eq!(swa.visible(1024).unwrap(), 1..1025);
        assert_eq!(swa.relative_distance(1024, 1), Some(1023));
        assert_eq!(swa.relative_distance(1024, 0), None);

        // Slot 5 is Full attention
        let full = &step.slots()[5];
        assert_eq!(full.before_kv(), 0..1023);
        assert_eq!(full.attention_kv(), 0..1025);
        assert_eq!(full.after_kv(), 0..1025);
        assert_eq!(full.visible(1024).unwrap(), 0..1025);
        assert_eq!(full.relative_distance(1024, 1), Some(1023));
        assert_eq!(full.relative_distance(1024, 0), None);
    }
}
