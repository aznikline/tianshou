#ifndef RL_POOL_H
#define RL_POOL_H

#include <linux/gfp.h>
#include <linux/list.h>
#include <linux/rcupdate.h>
#include <linux/spinlock.h>
#include <linux/types.h>

#include "rl_alloc.h"
#include "rl_policy.h"

#define RL_REQ_BUCKETS 10
#define RL_FRAG_BUCKETS 6
#define RL_HOLE_BUCKETS 5
#define RL_PRESSURE_BUCKETS 5
#define RL_MIX_BUCKETS 3
#define RL_STATE_KEY_SPACE \
	(2 * RL_REQ_BUCKETS * RL_FRAG_BUCKETS * RL_HOLE_BUCKETS * \
	 RL_PRESSURE_BUCKETS * RL_MIX_BUCKETS)

struct rl_block {
	u32 offset;
	u32 size;
	struct list_head list;
};

struct rl_pool {
	spinlock_t lock;
	void *base;
	u32 total_bytes;
	u32 free_bytes;
	u32 max_blocks;
	u32 recent_allocs;
	u32 recent_frees;
	u8 baseline_action;
	struct rl_policy __rcu *policy;
	struct list_head free_list;
	struct list_head used_list;
	struct list_head desc_free_list;
	struct rl_block *descriptors;
};

int rl_pool_init(struct rl_pool *pool, size_t total_bytes, u32 max_blocks,
		 u8 baseline_action, gfp_t gfp_mask);
void rl_pool_destroy(struct rl_pool *pool);
void rl_pool_set_policy(struct rl_pool *pool, struct rl_policy *policy);
u32 rl_pool_build_state_key(const struct rl_pool *pool, size_t size, bool is_free);
u8 rl_pool_select_action(const struct rl_pool *pool, size_t size, bool is_free);
void *rl_pool_alloc(struct rl_pool *pool, size_t size, u8 action, u64 *latency_ns);
int rl_pool_free(struct rl_pool *pool, void *ptr, bool eager_coalesce, u64 *latency_ns);
u32 rl_pool_largest_free_block(const struct rl_pool *pool);
u32 rl_pool_free_hole_count(const struct rl_pool *pool);

#endif /* RL_POOL_H */
