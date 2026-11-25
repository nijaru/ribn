//! Gemma 4 Metal paged KV pool specialization (`KV_PAGE_TOKENS = 16`) over
//! [`ribn_foundation::PagedBlockPool`].

pub use crate::paged_executor::KV_PAGE_TOKENS;
pub use ribn_foundation::{PagedBlockPool, PrefixCacheStats};

pub(crate) type PagedKvPool = PagedBlockPool<KV_PAGE_TOKENS>;
