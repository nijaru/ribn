//! Gemma 4 Metal paged KV pool specialization (`KV_PAGE_TOKENS = 16`) over
//! [`ribn_foundation::PagedBlockPool`].

pub use ribn_foundation::{PagedBlockPool, PrefixCacheStats};

/// Logical tokens stored in one physical Gemma 4 Metal KV page.
pub const KV_PAGE_TOKENS: usize = 16;

pub(crate) type PagedKvPool = PagedBlockPool<KV_PAGE_TOKENS>;
