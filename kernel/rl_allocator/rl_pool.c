#include <linux/errno.h>
#include <linux/bitops.h>
#include <linux/kernel.h>
#include <linux/ktime.h>
#include <linux/list.h>
#include <linux/math64.h>
#include <linux/minmax.h>
#include <linux/rcupdate.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/vmalloc.h>

#include "rl_pool.h"

static struct rl_block *rl_pool_desc_get(struct rl_pool *pool)
{
	struct rl_block *block;

	if (list_empty(&pool->desc_free_list))
		return NULL;

	block = list_first_entry(&pool->desc_free_list, struct rl_block, list);
	list_del_init(&block->list);
	return block;
}

static void rl_pool_desc_put(struct rl_block *block, struct rl_pool *pool)
{
	block->offset = 0;
	block->size = 0;
	block->tags = 0;
	list_add_tail(&block->list, &pool->desc_free_list);
}

static void rl_pool_insert_free_sorted(struct rl_pool *pool, struct rl_block *block)
{
	struct rl_block *cursor;

	list_for_each_entry(cursor, &pool->free_list, list) {
		if (block->offset < cursor->offset) {
			list_add_tail(&block->list, &cursor->list);
			return;
		}
	}

	list_add_tail(&block->list, &pool->free_list);
}

static void rl_pool_coalesce_locked(struct rl_pool *pool)
{
	struct rl_block *block, *tmp;

	list_for_each_entry_safe(block, tmp, &pool->free_list, list) {
		struct rl_block *next;

		if (list_is_last(&block->list, &pool->free_list))
			break;

		next = list_next_entry(block, list);
		if (block->offset + block->size != next->offset)
			continue;

		block->size += next->size;
		block->tags |= next->tags;
		list_del_init(&next->list);
		rl_pool_desc_put(next, pool);
	}
}

static struct rl_block *rl_pool_find_used_locked(struct rl_pool *pool, u32 offset)
{
	struct rl_block *block;

	list_for_each_entry(block, &pool->used_list, list) {
		if (block->offset == offset)
			return block;
	}

	return NULL;
}

static u8 rl_pool_resolve_semantic_action(u8 action, u32 req_flags)
{
	if (action != RL_ACTION_SEMANTIC_DEFAULT)
		return action;
	if (rl_req_has(req_flags, RL_REQ_HIGH_ORDER))
		return RL_ACTION_HIGH_ORDER_GUARD;
	if (rl_req_has(req_flags, RL_REQ_ASYNC))
		return RL_ACTION_ASYNC_DEFER;
	if (rl_req_has(req_flags, RL_REQ_SYNC))
		return RL_ACTION_SYNC_COMPACT;
	if (rl_req_has(req_flags, RL_REQ_FILE))
		return RL_ACTION_FILE_AFFINITY;
	if (rl_req_has(req_flags, RL_REQ_ANON))
		return RL_ACTION_ANON_AFFINITY;
	if (rl_req_has(req_flags, RL_REQ_RECLAIMABLE))
		return RL_ACTION_RECLAIM_REUSE;
	if (rl_req_has(req_flags, RL_REQ_MOVABLE))
		return RL_ACTION_MOVABLE_SPREAD;

	return RL_ACTION_BEST_FIT;
}

static int rl_pool_tag_score(u32 tags, u32 req_flags)
{
	return hweight32(tags & req_flags);
}

static struct rl_block *rl_pool_pick_tagged_locked(struct rl_pool *pool, size_t size,
						   u32 wanted_flags)
{
	struct rl_block *block;
	struct rl_block *best = NULL;
	int best_score = -1;

	list_for_each_entry(block, &pool->free_list, list) {
		int score;

		if (block->size < size)
			continue;
		score = rl_pool_tag_score(block->tags, wanted_flags);
		if (!best || score > best_score ||
		    (score == best_score && block->size < best->size)) {
			best = block;
			best_score = score;
		}
	}

	return best;
}

static struct rl_block *rl_pool_pick_first_locked(struct rl_pool *pool, size_t size)
{
	struct rl_block *block;

	list_for_each_entry(block, &pool->free_list, list) {
		if (block->size >= size)
			return block;
	}

	return NULL;
}

static struct rl_block *rl_pool_pick_best_locked(struct rl_pool *pool, size_t size)
{
	struct rl_block *block;
	struct rl_block *best = NULL;

	list_for_each_entry(block, &pool->free_list, list) {
		if (block->size < size)
			continue;
		if (!best || block->size < best->size)
			best = block;
	}

	return best;
}

static struct rl_block *rl_pool_pick_largest_locked(struct rl_pool *pool, size_t size)
{
	struct rl_block *block;
	struct rl_block *largest = NULL;
	u32 largest_size = 0;

	list_for_each_entry(block, &pool->free_list, list) {
		if (block->size < size)
			continue;
		if (!largest || block->size > largest_size) {
			largest = block;
			largest_size = block->size;
		}
	}

	return largest;
}

static struct rl_block *rl_pool_pick_highest_offset_locked(struct rl_pool *pool, size_t size)
{
	struct rl_block *block;
	struct rl_block *selected = NULL;

	list_for_each_entry(block, &pool->free_list, list) {
		if (block->size < size)
			continue;
		if (!selected || block->offset > selected->offset)
			selected = block;
	}

	return selected;
}

static struct rl_block *rl_pool_candidate_locked(struct rl_pool *pool, size_t size,
						 u8 action, u32 req_flags)
{
	action = rl_pool_resolve_semantic_action(action, req_flags);

	switch (action) {
	case RL_ACTION_FIRST_FIT:
	case RL_ACTION_FIRST_FIT_EAGER:
	case RL_ACTION_ASYNC_DEFER:
		return rl_pool_pick_first_locked(pool, size);
	case RL_ACTION_BEST_FIT:
	case RL_ACTION_BEST_FIT_EAGER:
	case RL_ACTION_SYNC_COMPACT:
		return rl_pool_pick_best_locked(pool, size);
	case RL_ACTION_FLAG_AFFINITY:
	case RL_ACTION_FLAG_AFFINITY_EAGER:
		return rl_pool_pick_tagged_locked(pool, size, req_flags);
	case RL_ACTION_LARGEST_FIT:
	case RL_ACTION_LARGEST_FIT_EAGER:
		return rl_pool_pick_largest_locked(pool, size);
	case RL_ACTION_ANON_AFFINITY:
		return rl_pool_pick_tagged_locked(pool, size, RL_REQ_ANON);
	case RL_ACTION_FILE_AFFINITY:
		return rl_pool_pick_tagged_locked(pool, size, RL_REQ_FILE);
	case RL_ACTION_RECLAIM_REUSE:
		return rl_pool_pick_tagged_locked(pool, size, RL_REQ_RECLAIMABLE);
	case RL_ACTION_MOVABLE_SPREAD:
		return rl_pool_pick_highest_offset_locked(pool, size);
	case RL_ACTION_HIGH_ORDER_GUARD:
		if (rl_req_has(req_flags, RL_REQ_HIGH_ORDER))
			return rl_pool_pick_largest_locked(pool, size);
		return rl_pool_pick_best_locked(pool, size);
	default:
		return rl_pool_pick_best_locked(pool, size);
	}
}

static u32 rl_pool_bucket_request(size_t size)
{
	static const u32 edges[RL_REQ_BUCKETS - 1] = {
		16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
	};
	u32 i;

	for (i = 0; i < ARRAY_SIZE(edges); i++) {
		if (size <= edges[i])
			return i;
	}

	return RL_REQ_BUCKETS - 1;
}

static u32 rl_pool_bucket_fragmentation(const struct rl_pool *pool)
{
	u32 largest = rl_pool_largest_free_block(pool);
	u64 frag_pct;

	if (!pool->free_bytes)
		return RL_FRAG_BUCKETS - 1;

	frag_pct = 100 - div_u64((u64)largest * 100, pool->free_bytes);
	if (!frag_pct)
		return 0;
	if (frag_pct <= 10)
		return 1;
	if (frag_pct <= 25)
		return 2;
	if (frag_pct <= 50)
		return 3;
	if (frag_pct <= 75)
		return 4;
	return 5;
}

static u32 rl_pool_bucket_holes(const struct rl_pool *pool)
{
	u32 holes = rl_pool_free_hole_count(pool);

	if (holes <= 1)
		return 0;
	if (holes <= 3)
		return 1;
	if (holes <= 7)
		return 2;
	if (holes <= 15)
		return 3;
	return 4;
}

static u32 rl_pool_bucket_pressure(const struct rl_pool *pool)
{
	u64 used_pct;

	if (!pool->total_bytes)
		return RL_PRESSURE_BUCKETS - 1;

	used_pct = div_u64((u64)(pool->total_bytes - pool->free_bytes) * 100,
			   pool->total_bytes);
	if (used_pct <= 25)
		return 0;
	if (used_pct <= 50)
		return 1;
	if (used_pct <= 75)
		return 2;
	if (used_pct <= 90)
		return 3;
	return 4;
}

static u32 rl_pool_bucket_mix(const struct rl_pool *pool)
{
	if (pool->recent_allocs > pool->recent_frees + 2)
		return 0;
	if (pool->recent_frees > pool->recent_allocs + 2)
		return 2;
	return 1;
}

static u32 rl_pool_bucket_req_flags(u32 req_flags)
{
	return req_flags & RL_REQ_FLAG_MASK;
}

int rl_pool_init(struct rl_pool *pool, size_t total_bytes, u32 max_blocks,
		 u8 baseline_action, gfp_t gfp_mask)
{
	struct rl_block *block;
	u32 i;

	if (!pool || !total_bytes || max_blocks < 2)
		return -EINVAL;

	memset(pool, 0, sizeof(*pool));
	spin_lock_init(&pool->lock);
	INIT_LIST_HEAD(&pool->free_list);
	INIT_LIST_HEAD(&pool->used_list);
	INIT_LIST_HEAD(&pool->desc_free_list);

	pool->base = vzalloc(total_bytes);
	if (!pool->base)
		return -ENOMEM;

	pool->descriptors = kcalloc(max_blocks, sizeof(*pool->descriptors), gfp_mask);
	if (!pool->descriptors) {
		vfree(pool->base);
		pool->base = NULL;
		return -ENOMEM;
	}

	pool->total_bytes = total_bytes;
	pool->free_bytes = total_bytes;
	pool->max_blocks = max_blocks;
	pool->baseline_action = baseline_action;

	for (i = 0; i < max_blocks; i++)
		INIT_LIST_HEAD(&pool->descriptors[i].list);
	for (i = 0; i < max_blocks; i++)
		list_add_tail(&pool->descriptors[i].list, &pool->desc_free_list);

	block = rl_pool_desc_get(pool);
	if (!block) {
		kfree(pool->descriptors);
		vfree(pool->base);
		return -ENOMEM;
	}

	block->offset = 0;
	block->size = total_bytes;
	block->tags = 0;
	list_add_tail(&block->list, &pool->free_list);

	return 0;
}

void rl_pool_destroy(struct rl_pool *pool)
{
	if (!pool)
		return;

	RCU_INIT_POINTER(pool->policy, NULL);
	kfree(pool->descriptors);
	pool->descriptors = NULL;
	if (pool->base)
		vfree(pool->base);
	pool->base = NULL;
}

void rl_pool_set_policy(struct rl_pool *pool, struct rl_policy *policy)
{
	unsigned long flags;

	spin_lock_irqsave(&pool->lock, flags);
	rcu_assign_pointer(pool->policy, policy);
	spin_unlock_irqrestore(&pool->lock, flags);
}

u32 rl_pool_largest_free_block(const struct rl_pool *pool)
{
	const struct rl_block *block;
	u32 largest = 0;

	list_for_each_entry(block, &pool->free_list, list)
		largest = max(largest, block->size);

	return largest;
}

u32 rl_pool_free_hole_count(const struct rl_pool *pool)
{
	const struct rl_block *block;
	u32 count = 0;

	list_for_each_entry(block, &pool->free_list, list)
		count++;

	return count;
}

u32 rl_pool_build_state_key(const struct rl_pool *pool, size_t size, bool is_free,
			    u32 req_flags)
{
	u32 key = is_free ? 1 : 0;

	key = key * RL_REQ_BUCKETS + rl_pool_bucket_request(size);
	key = key * RL_FRAG_BUCKETS + rl_pool_bucket_fragmentation(pool);
	key = key * RL_HOLE_BUCKETS + rl_pool_bucket_holes(pool);
	key = key * RL_PRESSURE_BUCKETS + rl_pool_bucket_pressure(pool);
	key = key * RL_MIX_BUCKETS + rl_pool_bucket_mix(pool);
	key = key * RL_REQ_FLAG_BUCKETS + rl_pool_bucket_req_flags(req_flags);

	return key;
}

u8 rl_pool_select_action(const struct rl_pool *pool, size_t size, bool is_free,
			 u32 req_flags)
{
	const struct rl_policy *policy;
	u8 action = READ_ONCE(pool->baseline_action);
	u32 state_key;

	rcu_read_lock();
	policy = rcu_dereference(pool->policy);
	if (!policy)
		goto out;

	state_key = rl_pool_build_state_key(pool, size, is_free, req_flags);
	if (rl_policy_lookup(policy, state_key, &action))
		action = READ_ONCE(pool->baseline_action);
out:
	rcu_read_unlock();
	return action;
}

void *rl_pool_alloc(struct rl_pool *pool, size_t size, u8 action, u32 req_flags,
		    u64 *latency_ns)
{
	struct rl_block *free_block;
	struct rl_block *used_block;
	unsigned long flags;
	u64 started_ns;
	u32 offset;
	char *base;

	if (!pool || !size || size > pool->total_bytes)
		return NULL;

	started_ns = ktime_get_ns();
	spin_lock_irqsave(&pool->lock, flags);

	free_block = rl_pool_candidate_locked(pool, size, action, req_flags);
	if (!free_block) {
		spin_unlock_irqrestore(&pool->lock, flags);
		if (latency_ns)
			*latency_ns = ktime_get_ns() - started_ns;
		return NULL;
	}

	offset = free_block->offset;
	if (free_block->size == size) {
		free_block->tags = req_flags;
		list_del_init(&free_block->list);
		list_add_tail(&free_block->list, &pool->used_list);
	} else {
		used_block = rl_pool_desc_get(pool);
		if (!used_block) {
			spin_unlock_irqrestore(&pool->lock, flags);
			if (latency_ns)
				*latency_ns = ktime_get_ns() - started_ns;
			return NULL;
		}
		used_block->offset = offset;
		used_block->size = size;
		used_block->tags = req_flags;
		list_add_tail(&used_block->list, &pool->used_list);
		free_block->offset += size;
		free_block->size -= size;
	}

	pool->free_bytes -= size;
	pool->recent_allocs++;
	spin_unlock_irqrestore(&pool->lock, flags);

	if (latency_ns)
		*latency_ns = ktime_get_ns() - started_ns;
	base = pool->base;
	return base + offset;
}

int rl_pool_free(struct rl_pool *pool, void *ptr, bool eager_coalesce, u64 *latency_ns)
{
	struct rl_block *block;
	unsigned long flags;
	u64 started_ns;
	u32 offset;
	char *base;

	if (!pool || !ptr)
		return -EINVAL;

	base = pool->base;
	if ((char *)ptr < base || (char *)ptr >= base + pool->total_bytes)
		return -EINVAL;

	offset = (u32)((char *)ptr - base);
	started_ns = ktime_get_ns();
	spin_lock_irqsave(&pool->lock, flags);

	block = rl_pool_find_used_locked(pool, offset);
	if (!block) {
		spin_unlock_irqrestore(&pool->lock, flags);
		if (latency_ns)
			*latency_ns = ktime_get_ns() - started_ns;
		return -ENOENT;
	}

	list_del_init(&block->list);
	rl_pool_insert_free_sorted(pool, block);
	pool->free_bytes += block->size;
	pool->recent_frees++;
	if (eager_coalesce)
		rl_pool_coalesce_locked(pool);

	spin_unlock_irqrestore(&pool->lock, flags);
	if (latency_ns)
		*latency_ns = ktime_get_ns() - started_ns;

	return 0;
}

u32 rl_pool_request_flags_for_ptr(struct rl_pool *pool, void *ptr)
{
	struct rl_block *block;
	unsigned long flags;
	u32 req_flags = 0;
	u32 offset;
	char *base;

	if (!pool || !ptr)
		return 0;

	base = pool->base;
	if ((char *)ptr < base || (char *)ptr >= base + pool->total_bytes)
		return 0;

	offset = (u32)((char *)ptr - base);
	spin_lock_irqsave(&pool->lock, flags);
	block = rl_pool_find_used_locked(pool, offset);
	if (block)
		req_flags = block->tags;
	spin_unlock_irqrestore(&pool->lock, flags);

	return req_flags;
}
