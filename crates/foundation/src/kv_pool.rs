//! Backend-agnostic paged KV block allocator with exact content-addressed radix prefix caching.
//!
//! Each physical block stores `PAGE_TOKENS` contiguous token positions across a
//! model's KV-owner layers. Full `PAGE_TOKENS`-token blocks become immutable once
//! published into the prefix trie and are shared zero-copy across concurrent and
//! subsequent requests via reference counting. Partial tail blocks stay private
//! (`ref_count == 1`) until they reach `PAGE_TOKENS`. When uncached free blocks
//! are exhausted, leaf cached blocks (`ref_count == 0 && child_count == 0`) are
//! evicted in LRU order.

use std::collections::HashMap;
use std::fmt;

/// Typed error returned by [`PagedBlockPool`] construction or allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockPoolError {
    /// Pool capacity (`num_blocks`) must be positive.
    ZeroBlocks,
    /// Compile-time `PAGE_TOKENS` must be positive.
    ZeroPageTokens,
    /// All physical blocks are currently pinned by active sequences.
    Exhausted,
}

impl fmt::Display for BlockPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBlocks => f.write_str("KV pool num_blocks must be positive"),
            Self::ZeroPageTokens => f.write_str("KV pool PAGE_TOKENS must be positive"),
            Self::Exhausted => {
                f.write_str("physical KV block capacity exhausted by active sequences")
            }
        }
    }
}

impl std::error::Error for BlockPoolError {}

/// Observable paged KV-pool and prefix-cache telemetry snapshot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefixCacheStats {
    pub num_blocks: usize,
    pub free_blocks: usize,
    pub cached_blocks: usize,
    pub pinned_blocks: usize,
    pub prefix_hit_tokens: u64,
    pub prefix_miss_tokens: u64,
    pub evictions: u64,
}

struct PhysicalBlock<const PAGE_TOKENS: usize> {
    ref_count: u32,
    node_id: Option<u64>,
    parent_node: u64,
    tokens: [u32; PAGE_TOKENS],
    child_count: u32,
    last_used: u64,
}

impl<const PAGE_TOKENS: usize> PhysicalBlock<PAGE_TOKENS> {
    const fn empty() -> Self {
        Self {
            ref_count: 0,
            node_id: None,
            parent_node: 0,
            tokens: [0; PAGE_TOKENS],
            child_count: 0,
            last_used: 0,
        }
    }
}

/// Pure-host physical block allocator and exact content-addressed radix prefix trie.
pub struct PagedBlockPool<const PAGE_TOKENS: usize = 16> {
    blocks: Vec<PhysicalBlock<PAGE_TOKENS>>,
    free_blocks: Vec<usize>,
    prefix_map: HashMap<(u64, [u32; PAGE_TOKENS]), usize>,
    node_to_block: HashMap<u64, usize>,
    tick: u64,
    next_node_id: u64,
    pinned_blocks: usize,
    cached_blocks: usize,
    prefix_hit_tokens: u64,
    prefix_miss_tokens: u64,
    evictions: u64,
}

impl<const PAGE_TOKENS: usize> PagedBlockPool<PAGE_TOKENS> {
    /// Construct a pool managing `num_blocks` physical blocks of `PAGE_TOKENS` tokens each.
    ///
    /// # Errors
    /// Returns [`BlockPoolError::ZeroPageTokens`] if `PAGE_TOKENS == 0`, or
    /// [`BlockPoolError::ZeroBlocks`] if `num_blocks == 0`.
    pub fn new(num_blocks: usize) -> Result<Self, BlockPoolError> {
        if PAGE_TOKENS == 0 {
            return Err(BlockPoolError::ZeroPageTokens);
        }
        if num_blocks == 0 {
            return Err(BlockPoolError::ZeroBlocks);
        }
        let mut blocks = Vec::with_capacity(num_blocks);
        blocks.resize_with(num_blocks, PhysicalBlock::empty);
        // Pop from back so initial allocation starts at block 0 cleanly.
        let free_blocks = (0..num_blocks).rev().collect();
        Ok(Self {
            blocks,
            free_blocks,
            prefix_map: HashMap::new(),
            node_to_block: HashMap::new(),
            tick: 0,
            next_node_id: 1,
            pinned_blocks: 0,
            cached_blocks: 0,
            prefix_hit_tokens: 0,
            prefix_miss_tokens: 0,
            evictions: 0,
        })
    }

    #[must_use]
    pub const fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    #[must_use]
    pub const fn pinned_blocks(&self) -> usize {
        self.pinned_blocks
    }

    /// Physical blocks currently available for new allocation (free blocks plus
    /// unpinned cached blocks evictable via LRU).
    #[must_use]
    pub const fn available_blocks(&self) -> usize {
        self.blocks.len().saturating_sub(self.pinned_blocks)
    }

    #[must_use]
    pub const fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            num_blocks: self.blocks.len(),
            free_blocks: self.free_blocks.len(),
            cached_blocks: self.cached_blocks,
            pinned_blocks: self.pinned_blocks,
            prefix_hit_tokens: self.prefix_hit_tokens,
            prefix_miss_tokens: self.prefix_miss_tokens,
            evictions: self.evictions,
        }
    }

    /// Inspect how many `PAGE_TOKENS`-aligned cached prefix pages match `prompt` without
    /// mutating reference counts, returning `(matched_pages, newly_pinned_blocks)` where
    /// `newly_pinned_blocks` is the number of matched pages currently unpinned (`ref_count == 0`).
    #[must_use]
    pub fn peek_prefix_match(&self, prompt: &[u32]) -> (usize, usize) {
        let max_pages = prompt.len().saturating_sub(1) / PAGE_TOKENS;
        let mut matched_pages = 0_usize;
        let mut newly_pinned = 0_usize;
        let mut parent = 0_u64;
        for page in 0..max_pages {
            let start = page * PAGE_TOKENS;
            let mut key = [0_u32; PAGE_TOKENS];
            key.copy_from_slice(&prompt[start..start + PAGE_TOKENS]);
            let Some(&block_id) = self.prefix_map.get(&(parent, key)) else {
                break;
            };
            let block = &self.blocks[block_id];
            if block.ref_count == 0 {
                newly_pinned += 1;
            }
            let Some(nid) = block.node_id else {
                break;
            };
            parent = nid;
            matched_pages += 1;
        }
        (matched_pages, newly_pinned)
    }

    /// Match the longest exact `PAGE_TOKENS`-aligned cached prefix strictly shorter
    /// than `prompt.len()` (leaving at least one token for logit computation),
    /// pinning every matched physical block (`ref_count += 1`).
    pub fn match_and_pin_prefix(&mut self, prompt: &[u32]) -> (Vec<usize>, u64) {
        let max_pages = prompt.len().saturating_sub(1) / PAGE_TOKENS;
        let mut pinned = Vec::with_capacity(max_pages);
        let mut parent = 0_u64;
        for page in 0..max_pages {
            let start = page * PAGE_TOKENS;
            let mut key = [0_u32; PAGE_TOKENS];
            key.copy_from_slice(&prompt[start..start + PAGE_TOKENS]);
            let Some(&block_id) = self.prefix_map.get(&(parent, key)) else {
                break;
            };
            let Some(nid) = self.blocks[block_id].node_id else {
                break;
            };
            self.tick += 1;
            let block = &mut self.blocks[block_id];
            if block.ref_count == 0 {
                self.pinned_blocks += 1;
            }
            block.ref_count += 1;
            block.last_used = self.tick;
            parent = nid;
            pinned.push(block_id);
        }
        let hit = (pinned.len() * PAGE_TOKENS) as u64;
        let miss = (prompt.len() as u64).saturating_sub(hit);
        self.prefix_hit_tokens += hit;
        self.prefix_miss_tokens += miss;
        (pinned, parent)
    }

    /// Ensure `slot_blocks` has enough physical blocks to cover `0..end_pos`.
    /// On exhaustion, rolls back any newly allocated blocks in this call.
    ///
    /// # Errors
    /// Returns [`BlockPoolError::Exhausted`] when all physical blocks are pinned by active sequences.
    pub fn ensure_blocks_for_range(
        &mut self,
        slot_blocks: &mut Vec<usize>,
        end_pos: usize,
    ) -> Result<(), BlockPoolError> {
        let needed = end_pos.div_ceil(PAGE_TOKENS);
        if needed <= slot_blocks.len() {
            return Ok(());
        }
        let start_len = slot_blocks.len();
        while slot_blocks.len() < needed {
            match self.alloc_private_block() {
                Ok(block_id) => slot_blocks.push(block_id),
                Err(error) => {
                    for block_id in slot_blocks.drain(start_len..) {
                        self.release_private_block(block_id);
                    }
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// Freeze and publish every newly completed `PAGE_TOKENS`-token block into the prefix
    /// trie. Deduplicates against concurrent sequences that completed the exact
    /// same prefix block, freeing the duplicate private block immediately.
    pub fn publish_completed_blocks(
        &mut self,
        slot_blocks: &mut [usize],
        slot_tokens: &[u32],
        cached_pages: &mut usize,
        last_node: &mut u64,
    ) {
        let completed_pages = slot_tokens.len() / PAGE_TOKENS;
        while *cached_pages < completed_pages {
            let page_idx = *cached_pages;
            let start = page_idx * PAGE_TOKENS;
            let mut key = [0_u32; PAGE_TOKENS];
            key.copy_from_slice(&slot_tokens[start..start + PAGE_TOKENS]);
            let priv_block = slot_blocks[page_idx];
            self.tick += 1;

            if let Some(&existing) = self.prefix_map.get(&(*last_node, key)) {
                if existing == priv_block {
                    let block = &mut self.blocks[existing];
                    block.last_used = self.tick;
                    if let Some(nid) = block.node_id {
                        *last_node = nid;
                    }
                } else {
                    let existing_nid = {
                        let block = &mut self.blocks[existing];
                        if block.ref_count == 0 {
                            self.pinned_blocks += 1;
                        }
                        block.ref_count += 1;
                        block.last_used = self.tick;
                        block.node_id
                    };
                    slot_blocks[page_idx] = existing;
                    self.release_private_block(priv_block);
                    if let Some(nid) = existing_nid {
                        *last_node = nid;
                    }
                }
            } else {
                let nid = self.next_node_id;
                self.next_node_id += 1;
                if *last_node != 0 {
                    let parent_block = self.node_to_block[last_node];
                    self.blocks[parent_block].child_count += 1;
                }
                let block = &mut self.blocks[priv_block];
                debug_assert!(block.node_id.is_none(), "private block already cached");
                block.node_id = Some(nid);
                block.parent_node = *last_node;
                block.tokens = key;
                block.child_count = 0;
                block.last_used = self.tick;
                self.prefix_map.insert((*last_node, key), priv_block);
                self.node_to_block.insert(nid, priv_block);
                self.cached_blocks += 1;
                *last_node = nid;
            }
            *cached_pages += 1;
        }
    }

    /// Unpin every block owned by a completed, cancelled, or reset sequence.
    /// Published prefix blocks remain cached for reuse until evicted via LRU.
    pub fn release_slot_blocks(&mut self, slot_blocks: Vec<usize>) {
        for block_id in slot_blocks.into_iter().rev() {
            let block = &mut self.blocks[block_id];
            debug_assert!(block.ref_count > 0, "block ref_count underflow");
            block.ref_count -= 1;
            if block.ref_count == 0 {
                self.pinned_blocks -= 1;
                if block.node_id.is_none() {
                    self.free_blocks.push(block_id);
                }
            }
        }
    }

    fn release_private_block(&mut self, block_id: usize) {
        let block = &mut self.blocks[block_id];
        debug_assert!(block.ref_count > 0);
        block.ref_count -= 1;
        if block.ref_count == 0 {
            self.pinned_blocks -= 1;
            if block.node_id.is_none() {
                self.free_blocks.push(block_id);
            }
        }
    }

    fn alloc_private_block(&mut self) -> Result<usize, BlockPoolError> {
        if let Some(block_id) = self.free_blocks.pop() {
            self.tick += 1;
            let block = &mut self.blocks[block_id];
            debug_assert_eq!(block.ref_count, 0);
            block.ref_count = 1;
            self.pinned_blocks += 1;
            block.node_id = None;
            block.parent_node = 0;
            block.child_count = 0;
            block.last_used = self.tick;
            return Ok(block_id);
        }

        // Evict the LRU unreferenced leaf cached block. Because every live
        // sequence pins its full ancestor chain, ref_count(parent) >= ref_count(child),
        // so whenever any unreferenced cached block exists, at least one is a leaf.
        let victim = self
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.ref_count == 0 && b.node_id.is_some() && b.child_count == 0)
            .min_by_key(|(_, b)| b.last_used)
            .map(|(idx, _)| idx);

        let Some(victim_id) = victim else {
            return Err(BlockPoolError::Exhausted);
        };

        let (nid, parent, tokens) = {
            let block = &self.blocks[victim_id];
            (
                block.node_id.expect("victim is cached"),
                block.parent_node,
                block.tokens,
            )
        };
        self.prefix_map.remove(&(parent, tokens));
        self.node_to_block.remove(&nid);
        if parent != 0 {
            let parent_block = self.node_to_block[&parent];
            self.blocks[parent_block].child_count -= 1;
        }
        self.cached_blocks -= 1;
        self.evictions += 1;
        self.tick += 1;

        let block = &mut self.blocks[victim_id];
        debug_assert_eq!(block.ref_count, 0);
        block.ref_count = 1;
        self.pinned_blocks += 1;
        block.node_id = None;
        block.parent_node = 0;
        block.child_count = 0;
        block.last_used = self.tick;
        Ok(victim_id)
    }
}

#[cfg(test)]
mod tests {
    use super::{BlockPoolError, PagedBlockPool};

    #[test]
    fn rejects_zero_blocks_or_zero_page_tokens() {
        assert_eq!(
            PagedBlockPool::<16>::new(0).err(),
            Some(BlockPoolError::ZeroBlocks)
        );
        assert_eq!(
            PagedBlockPool::<0>::new(4).err(),
            Some(BlockPoolError::ZeroPageTokens)
        );
    }

    #[test]
    fn prefix_cache_matches_shares_peeks_and_evicts_lru_leaves() {
        let mut pool = PagedBlockPool::<16>::new(4).unwrap();
        let prompt: Vec<u32> = (10..50).collect(); // 40 tokens = 2 full pages + 8 tail

        assert_eq!(pool.peek_prefix_match(&prompt), (0, 0));
        let (mut blocks_a, mut node_a) = pool.match_and_pin_prefix(&prompt);
        assert!(blocks_a.is_empty());
        pool.ensure_blocks_for_range(&mut blocks_a, prompt.len())
            .unwrap();
        assert_eq!(blocks_a.len(), 3);
        assert_eq!(pool.pinned_blocks(), 3);
        let mut cached_a = 0;
        pool.publish_completed_blocks(&mut blocks_a, &prompt, &mut cached_a, &mut node_a);
        assert_eq!(cached_a, 2);
        assert_eq!(pool.stats().cached_blocks, 2);

        // Concurrent request with same prefix + different suffix hits both full pages (already pinned -> 0 newly pinned)
        let mut prompt_b = prompt[..32].to_vec();
        prompt_b.extend(200..205);
        assert_eq!(pool.peek_prefix_match(&prompt_b), (2, 0));
        let (mut blocks_b, _) = pool.match_and_pin_prefix(&prompt_b);
        assert_eq!(blocks_b.len(), 2);
        assert_eq!(blocks_b, blocks_a[..2]);
        assert_eq!(pool.stats().prefix_hit_tokens, 32);

        pool.ensure_blocks_for_range(&mut blocks_b, prompt_b.len())
            .unwrap();
        assert_eq!(blocks_b.len(), 3);
        assert_eq!(pool.stats().free_blocks, 0);
        assert_eq!(pool.stats().pinned_blocks, 4);
        assert_eq!(pool.available_blocks(), 0);

        // Exhaustion rolls back cleanly without leaking
        let mut blocks_c = Vec::new();
        assert_eq!(
            pool.ensure_blocks_for_range(&mut blocks_c, 1),
            Err(BlockPoolError::Exhausted)
        );
        assert!(blocks_c.is_empty());

        // Release both sequences: private tails go free, 2 prefix blocks remain cached & unpinned
        pool.release_slot_blocks(blocks_a);
        pool.release_slot_blocks(blocks_b);
        assert_eq!(pool.stats().free_blocks, 2);
        assert_eq!(pool.stats().cached_blocks, 2);
        assert_eq!(pool.stats().pinned_blocks, 0);
        assert_eq!(pool.available_blocks(), 4);
        // Peeking now shows 2 matched pages and 2 newly pinned
        assert_eq!(pool.peek_prefix_match(&prompt_b), (2, 2));

        // Allocating 4 new blocks evicts leaf page 1 then root page 0
        let mut blocks_d = Vec::new();
        pool.ensure_blocks_for_range(&mut blocks_d, 64).unwrap();
        assert_eq!(blocks_d.len(), 4);
        assert_eq!(pool.stats().cached_blocks, 0);
        assert_eq!(pool.stats().evictions, 2);
    }

    #[test]
    fn concurrent_identical_prompts_deduplicate_blocks_on_publish_with_custom_page_size() {
        let mut pool = PagedBlockPool::<4>::new(4).unwrap();
        let prompt: Vec<u32> = (1..=6).collect(); // 1 full 4-token page + 2 tail

        let (mut b1, mut n1) = pool.match_and_pin_prefix(&prompt);
        let (mut b2, mut n2) = pool.match_and_pin_prefix(&prompt);
        pool.ensure_blocks_for_range(&mut b1, prompt.len()).unwrap();
        pool.ensure_blocks_for_range(&mut b2, prompt.len()).unwrap();
        assert_eq!(pool.stats().free_blocks, 0);

        let (mut c1, mut c2) = (0, 0);
        pool.publish_completed_blocks(&mut b1, &prompt, &mut c1, &mut n1);
        pool.publish_completed_blocks(&mut b2, &prompt, &mut c2, &mut n2);
        assert_eq!(b1[0], b2[0]);
        assert_eq!(pool.stats().cached_blocks, 1);
        assert_eq!(pool.stats().free_blocks, 1);
        assert_eq!(pool.stats().pinned_blocks, 3);
    }
}
