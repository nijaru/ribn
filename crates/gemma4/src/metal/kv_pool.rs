//! Paged KV block allocator with exact content-addressed radix prefix caching.
//!
//! Each physical block stores `KV_PAGE_TOKENS` (16) contiguous token positions
//! across all KV-owner layers. Full 16-token blocks become immutable once
//! published into the prefix trie and are shared zero-copy across concurrent and
//! subsequent requests via reference counting. Partial tail blocks stay private
//! (`ref_count == 1`) until they reach 16 tokens. When uncached free blocks are
//! exhausted, leaf cached blocks (`ref_count == 0 && child_count == 0`) are
//! evicted in LRU order.

use crate::Gemma4Error;
use std::collections::HashMap;

/// Logical tokens stored in one physical KV page.
pub const KV_PAGE_TOKENS: usize = 16;

/// Observable prefix-cache and pool counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PrefixCacheStats {
    pub num_blocks: usize,
    pub free_blocks: usize,
    pub cached_blocks: usize,
    pub prefix_hit_tokens: u64,
    pub prefix_miss_tokens: u64,
    pub evictions: u64,
}

struct PhysicalBlock {
    ref_count: u32,
    node_id: Option<u64>,
    parent_node: u64,
    tokens: [u32; KV_PAGE_TOKENS],
    child_count: u32,
    last_used: u64,
}

impl PhysicalBlock {
    const fn empty() -> Self {
        Self {
            ref_count: 0,
            node_id: None,
            parent_node: 0,
            tokens: [0; KV_PAGE_TOKENS],
            child_count: 0,
            last_used: 0,
        }
    }
}

pub(crate) struct PagedKvPool {
    blocks: Vec<PhysicalBlock>,
    free_blocks: Vec<usize>,
    prefix_map: HashMap<(u64, [u32; KV_PAGE_TOKENS]), usize>,
    node_to_block: HashMap<u64, usize>,
    tick: u64,
    next_node_id: u64,
    pinned_blocks: usize,
    cached_blocks: usize,
    prefix_hit_tokens: u64,
    prefix_miss_tokens: u64,
    evictions: u64,
}

impl PagedKvPool {
    pub(crate) fn new(num_blocks: usize) -> Result<Self, Gemma4Error> {
        if num_blocks == 0 {
            return Err(Gemma4Error::new("KV pool", "num_blocks must be positive"));
        }
        let mut blocks = Vec::with_capacity(num_blocks);
        blocks.resize_with(num_blocks, PhysicalBlock::empty);
        // Pop from back so allocation starts at block 0 cleanly.
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

    pub(crate) const fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub(crate) const fn pinned_blocks(&self) -> usize {
        self.pinned_blocks
    }

    /// Inspect how many 16-token cached prefix pages match `prompt` without mutating
    /// reference counts, returning `(matched_pages, newly_pinned_blocks)` where
    /// `newly_pinned_blocks` is the number of matched pages currently unpinned (`ref_count == 0`).
    pub(crate) fn peek_prefix_match(&self, prompt: &[u32]) -> (usize, usize) {
        let max_pages = prompt.len().saturating_sub(1) / KV_PAGE_TOKENS;
        let mut matched_pages = 0_usize;
        let mut newly_pinned = 0_usize;
        let mut parent = 0_u64;
        for page in 0..max_pages {
            let start = page * KV_PAGE_TOKENS;
            let mut key = [0_u32; KV_PAGE_TOKENS];
            key.copy_from_slice(&prompt[start..start + KV_PAGE_TOKENS]);
            let Some(&block_id) = self.prefix_map.get(&(parent, key)) else {
                break;
            };
            let block = &self.blocks[block_id];
            if block.ref_count == 0 {
                newly_pinned += 1;
            }
            parent = block.node_id.expect("cached block has node_id");
            matched_pages += 1;
        }
        (matched_pages, newly_pinned)
    }

    pub(crate) fn stats(&self) -> PrefixCacheStats {
        PrefixCacheStats {
            num_blocks: self.blocks.len(),
            free_blocks: self.free_blocks.len(),
            cached_blocks: self.cached_blocks,
            prefix_hit_tokens: self.prefix_hit_tokens,
            prefix_miss_tokens: self.prefix_miss_tokens,
            evictions: self.evictions,
        }
    }

    /// Match the longest exact 16-token-aligned cached prefix strictly shorter
    /// than `prompt.len()` (leaving at least one token for logit computation),
    /// pinning every matched physical block (`ref_count += 1`).
    pub(crate) fn match_and_pin_prefix(&mut self, prompt: &[u32]) -> (Vec<usize>, u64) {
        let max_pages = prompt.len().saturating_sub(1) / KV_PAGE_TOKENS;
        let mut pinned = Vec::with_capacity(max_pages);
        let mut parent = 0_u64;
        for page in 0..max_pages {
            let start = page * KV_PAGE_TOKENS;
            let mut key = [0_u32; KV_PAGE_TOKENS];
            key.copy_from_slice(&prompt[start..start + KV_PAGE_TOKENS]);
            let Some(&block_id) = self.prefix_map.get(&(parent, key)) else {
                break;
            };
            self.tick += 1;
            let block = &mut self.blocks[block_id];
            if block.ref_count == 0 {
                self.pinned_blocks += 1;
            }
            block.ref_count += 1;
            block.last_used = self.tick;
            parent = block.node_id.expect("cached block has node_id");
            pinned.push(block_id);
        }
        let hit = (pinned.len() * KV_PAGE_TOKENS) as u64;
        let miss = (prompt.len() as u64).saturating_sub(hit);
        self.prefix_hit_tokens += hit;
        self.prefix_miss_tokens += miss;
        (pinned, parent)
    }

    /// Ensure `slot_blocks` has enough physical blocks to cover `0..end_pos`.
    /// On exhaustion, rolls back any newly allocated blocks in this call.
    pub(crate) fn ensure_blocks_for_range(
        &mut self,
        slot_blocks: &mut Vec<usize>,
        end_pos: usize,
    ) -> Result<(), Gemma4Error> {
        let needed = end_pos.div_ceil(KV_PAGE_TOKENS);
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

    /// Freeze and publish every newly completed 16-token block into the prefix
    /// trie. Deduplicates against concurrent sequences that completed the exact
    /// same prefix block, freeing the duplicate private block immediately.
    pub(crate) fn publish_completed_blocks(
        &mut self,
        slot_blocks: &mut [usize],
        slot_tokens: &[u32],
        cached_pages: &mut usize,
        last_node: &mut u64,
    ) {
        let completed_pages = slot_tokens.len() / KV_PAGE_TOKENS;
        while *cached_pages < completed_pages {
            let page_idx = *cached_pages;
            let start = page_idx * KV_PAGE_TOKENS;
            let mut key = [0_u32; KV_PAGE_TOKENS];
            key.copy_from_slice(&slot_tokens[start..start + KV_PAGE_TOKENS]);
            let priv_block = slot_blocks[page_idx];
            self.tick += 1;

            if let Some(&existing) = self.prefix_map.get(&(*last_node, key)) {
                if existing == priv_block {
                    let block = &mut self.blocks[existing];
                    block.last_used = self.tick;
                    *last_node = block.node_id.expect("cached block has node_id");
                } else {
                    let node_id = {
                        let block = &mut self.blocks[existing];
                        if block.ref_count == 0 {
                            self.pinned_blocks += 1;
                        }
                        block.ref_count += 1;
                        block.last_used = self.tick;
                        block.node_id.expect("cached block has node_id")
                    };
                    slot_blocks[page_idx] = existing;
                    self.release_private_block(priv_block);
                    *last_node = node_id;
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

    /// Unpin every block owned by a completed/cancelled/reset sequence.
    /// Published prefix blocks remain cached for reuse until evicted via LRU.
    pub(crate) fn release_slot_blocks(&mut self, slot_blocks: Vec<usize>) {
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

    fn alloc_private_block(&mut self) -> Result<usize, Gemma4Error> {
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
            return Err(Gemma4Error::new(
                "KV pool",
                "physical KV block capacity exhausted by active sequences",
            ));
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
    use super::{KV_PAGE_TOKENS, PagedKvPool};

    #[test]
    fn prefix_cache_matches_shares_deduplicates_and_evicts_lru_leaves() {
        let mut pool = PagedKvPool::new(4).unwrap();
        let prompt: Vec<u32> = (10..50).collect(); // 40 tokens = 2 full pages + 8 tail

        // First request: 0 hit, allocates 3 blocks (pages 0, 1, 2)
        let (mut blocks_a, mut node_a) = pool.match_and_pin_prefix(&prompt);
        assert!(blocks_a.is_empty());
        pool.ensure_blocks_for_range(&mut blocks_a, prompt.len())
            .unwrap();
        assert_eq!(blocks_a.len(), 3);
        let mut cached_a = 0;
        pool.publish_completed_blocks(&mut blocks_a, &prompt, &mut cached_a, &mut node_a);
        assert_eq!(cached_a, 2);
        assert_eq!(pool.stats().cached_blocks, 2);

        // Concurrent request with same prefix + different suffix hits both full pages (32 tokens)!
        let mut prompt_b = prompt[..32].to_vec();
        prompt_b.extend(200..205);
        let (mut blocks_b, _) = pool.match_and_pin_prefix(&prompt_b);
        assert_eq!(blocks_b.len(), 2);
        assert_eq!(blocks_b, blocks_a[..2]);
        assert_eq!(pool.stats().prefix_hit_tokens, 32);

        // Needs 1 private tail block -> uses the 4th physical block
        pool.ensure_blocks_for_range(&mut blocks_b, prompt_b.len())
            .unwrap();
        assert_eq!(blocks_b.len(), 3);
        assert_eq!(pool.stats().free_blocks, 0);

        // Exhaustion while all 4 blocks are pinned must fail cleanly without leaking
        let mut blocks_c = Vec::new();
        assert!(pool.ensure_blocks_for_range(&mut blocks_c, 1).is_err());
        assert!(blocks_c.is_empty());

        // Release both sequences: private tail blocks go free, 2 prefix blocks stay cached
        pool.release_slot_blocks(blocks_a);
        pool.release_slot_blocks(blocks_b);
        assert_eq!(pool.stats().free_blocks, 2);
        assert_eq!(pool.stats().cached_blocks, 2);

        // Allocate 4 blocks for a brand-new unrelated prompt -> evicts leaf page 1 then root page 0!
        let mut blocks_d = Vec::new();
        pool.ensure_blocks_for_range(&mut blocks_d, 4 * KV_PAGE_TOKENS)
            .unwrap();
        assert_eq!(blocks_d.len(), 4);
        assert_eq!(pool.stats().cached_blocks, 0);
        assert_eq!(pool.stats().evictions, 2);
    }

    #[test]
    fn concurrent_identical_prompts_deduplicate_blocks_on_publish() {
        let mut pool = PagedKvPool::new(4).unwrap();
        let prompt: Vec<u32> = (1..=20).collect(); // 1 full page + 4 tail

        let (mut b1, mut n1) = pool.match_and_pin_prefix(&prompt);
        let (mut b2, mut n2) = pool.match_and_pin_prefix(&prompt);
        pool.ensure_blocks_for_range(&mut b1, prompt.len()).unwrap();
        pool.ensure_blocks_for_range(&mut b2, prompt.len()).unwrap();
        assert_eq!(pool.stats().free_blocks, 0);

        let (mut c1, mut c2) = (0, 0);
        pool.publish_completed_blocks(&mut b1, &prompt, &mut c1, &mut n1);
        pool.publish_completed_blocks(&mut b2, &prompt, &mut c2, &mut n2);
        // Page 0 was deduplicated, freeing 1 physical block immediately!
        assert_eq!(b1[0], b2[0]);
        assert_eq!(pool.stats().cached_blocks, 1);
        assert_eq!(pool.stats().free_blocks, 1);
    }
}
