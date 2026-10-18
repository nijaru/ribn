//! Pure logical KV ranges. No device storage, reservations, commit tokens or device
//! lifetime proof are represented here; the executor must bind these to live state.

use crate::{AttentionKind, Gemma4Config, Gemma4Error};
use std::ops::Range;

/// Persistent coverage, distinct from the span a current attention operation reads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetainedSpan(Range<usize>);
impl RetainedSpan {
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        self.0.clone()
    }
}

/// Untrimmed old-plus-current KV for this step, including a shared donor's reads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AttentionSpan(Range<usize>);
impl AttentionSpan {
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        self.0.clone()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvOwnerStep {
    layer: usize,
    head_dim: usize,
    kind: AttentionKind,
    last_consumer: usize,
    input: Range<usize>,
    window: usize,
    before: RetainedSpan,
    attention: AttentionSpan,
    after: RetainedSpan,
    bytes_per_token: u64,
}
impl KvOwnerStep {
    #[must_use]
    pub const fn layer(&self) -> usize {
        self.layer
    }
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }
    #[must_use]
    pub const fn kind(&self) -> AttentionKind {
        self.kind
    }
    #[must_use]
    pub const fn last_consumer(&self) -> usize {
        self.last_consumer
    }
    #[must_use]
    pub const fn before(&self) -> &RetainedSpan {
        &self.before
    }
    #[must_use]
    pub const fn attention(&self) -> &AttentionSpan {
        &self.attention
    }
    #[must_use]
    pub const fn after(&self) -> &RetainedSpan {
        &self.after
    }
    #[must_use]
    pub const fn bytes_per_token(&self) -> u64 {
        self.bytes_per_token
    }

    /// Causal visibility for a query in the accepted chunk. A long chunk's entire
    /// attention span is not permission for every query to see all its positions.
    /// # Errors
    /// Rejects queries outside this step's accepted input range.
    pub fn visible(&self, query: usize) -> Result<Range<usize>, Gemma4Error> {
        if !self.input.contains(&query) {
            return Err(Gemma4Error::new(
                "query position",
                "outside accepted input range",
            ));
        }
        let start = match self.kind {
            AttentionKind::Full => 0,
            AttentionKind::Sliding => query.saturating_sub(self.window - 1),
        };
        Ok(start..query + 1)
    }
}

/// Semantic range calculation only. Prefix/capacity are caller assertions, not a
/// restored/committed state, admission result or physical memory grant. The future
/// executor must reject stale sequence/incarnation/coverage before consuming it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvStep {
    input: Range<usize>,
    owners: Vec<KvOwnerStep>,
    retained_payload_bytes: u64,
    donor_payload_bytes: u64,
}
impl KvStep {
    #[must_use]
    pub fn input(&self) -> Range<usize> {
        self.input.clone()
    }
    #[must_use]
    pub fn owners(&self) -> &[KvOwnerStep] {
        &self.owners
    }
    /// Logical post-step persistent BF16 K+V payload, not physical backing.
    #[must_use]
    pub const fn retained_payload_bytes(&self) -> u64 {
        self.retained_payload_bytes
    }
    /// Logical current spans reused by later layers. Do not blindly add this to
    /// persistent payload: they may share backing, whose lifetime is backend-owned.
    #[must_use]
    pub const fn donor_payload_bytes(&self) -> u64 {
        self.donor_payload_bytes
    }
}

impl Gemma4Config {
    /// Calculate the exact old/current/retained ranges for a positive chunk.
    /// No authoritative state is mutated, even on failure. This is not preparation.
    ///
    /// # Errors
    /// Rejects zero progress, overflow, invalid reach or byte-accounting overflow.
    pub fn kv_step(
        &self,
        prefix: usize,
        tokens: usize,
        capacity: usize,
    ) -> Result<KvStep, Gemma4Error> {
        let end = prefix
            .checked_add(tokens)
            .filter(|_| tokens > 0)
            .ok_or_else(|| {
                Gemma4Error::new("KV input", "positive, non-overflowing progress is required")
            })?;
        if capacity > self.context || end > capacity {
            return Err(Gemma4Error::new(
                "KV capacity",
                "input exceeds request reach or reach exceeds model context",
            ));
        }
        let mut owners = Vec::new();
        let mut retained_payload_bytes = 0_u64;
        let mut donor_payload_bytes = 0_u64;
        for (layer, geometry) in self
            .layers
            .iter()
            .enumerate()
            .filter(|(i, g)| *i == g.kv_owner)
        {
            let start = if geometry.kind == AttentionKind::Sliding {
                prefix.saturating_sub(self.window - 1)
            } else {
                0
            };
            let retained = if geometry.kind == AttentionKind::Sliding {
                end.saturating_sub(self.window - 1)
            } else {
                0
            };
            // BF16 K and V for stored KV heads, not repeated query heads.
            let bytes_per_token = u64::try_from(self.kv_heads)
                .ok()
                .and_then(|n| n.checked_mul(u64::try_from(geometry.head_dim).ok()?))
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| Gemma4Error::new("KV geometry", "BF16 byte size overflows"))?;
            add_payload(&mut retained_payload_bytes, end - retained, bytes_per_token)?;
            if geometry.kv_last_consumer > layer {
                add_payload(&mut donor_payload_bytes, end - start, bytes_per_token)?;
            }
            owners.push(KvOwnerStep {
                layer,
                head_dim: geometry.head_dim,
                kind: geometry.kind,
                last_consumer: geometry.kv_last_consumer,
                input: prefix..end,
                window: self.window,
                before: RetainedSpan(start..prefix),
                attention: AttentionSpan(start..end),
                after: RetainedSpan(retained..end),
                bytes_per_token,
            });
        }
        Ok(KvStep {
            input: prefix..end,
            owners,
            retained_payload_bytes,
            donor_payload_bytes,
        })
    }
}

fn add_payload(total: &mut u64, tokens: usize, per_token: u64) -> Result<(), Gemma4Error> {
    *total = u64::try_from(tokens)
        .ok()
        .and_then(|n| n.checked_mul(per_token))
        .and_then(|n| total.checked_add(n))
        .ok_or_else(|| Gemma4Error::new("KV payload", "aggregate logical byte size overflows"))?;
    Ok(())
}
