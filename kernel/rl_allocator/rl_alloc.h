#ifndef RL_ALLOC_H
#define RL_ALLOC_H

#include <linux/bits.h>
#include <linux/gfp_types.h>
#include <linux/mm.h>
#include <linux/types.h>

#define RL_POLICY_MAGIC "RLP1"
#define RL_POLICY_MAGIC_SIZE 4
#define RL_REQ_FLAG_MASK GENMASK(6, 0)

enum rl_req_flag {
	RL_REQ_SYNC = BIT(0),
	RL_REQ_ASYNC = BIT(1),
	RL_REQ_ANON = BIT(2),
	RL_REQ_FILE = BIT(3),
	RL_REQ_RECLAIMABLE = BIT(4),
	RL_REQ_MOVABLE = BIT(5),
	RL_REQ_HIGH_ORDER = BIT(6),
};

enum rl_mode {
	RL_MODE_FIRST_FIT = 0,
	RL_MODE_BEST_FIT = 1,
	RL_MODE_TABLE = 2,
};

enum rl_action {
	RL_ACTION_FIRST_FIT = 0,
	RL_ACTION_BEST_FIT = 1,
	RL_ACTION_FLAG_AFFINITY = 2,
	RL_ACTION_LARGEST_FIT = 3,
	RL_ACTION_FIRST_FIT_EAGER = 4,
	RL_ACTION_BEST_FIT_EAGER = 5,
	RL_ACTION_FLAG_AFFINITY_EAGER = 6,
	RL_ACTION_LARGEST_FIT_EAGER = 7,
	RL_ACTION_SYNC_COMPACT = 8,
	RL_ACTION_ASYNC_DEFER = 9,
	RL_ACTION_ANON_AFFINITY = 10,
	RL_ACTION_FILE_AFFINITY = 11,
	RL_ACTION_RECLAIM_REUSE = 12,
	RL_ACTION_MOVABLE_SPREAD = 13,
	RL_ACTION_HIGH_ORDER_GUARD = 14,
	RL_ACTION_SEMANTIC_DEFAULT = 15,
	RL_ACTION_MAX = 16,
};

static inline bool rl_action_is_eager(u8 action)
{
	return action == RL_ACTION_FIRST_FIT_EAGER ||
	       action == RL_ACTION_BEST_FIT_EAGER ||
	       action == RL_ACTION_FLAG_AFFINITY_EAGER ||
	       action == RL_ACTION_LARGEST_FIT_EAGER ||
	       action == RL_ACTION_SYNC_COMPACT;
}

static inline u8 rl_action_base(u8 action)
{
	switch (action) {
	case RL_ACTION_FIRST_FIT_EAGER:
		return RL_ACTION_FIRST_FIT;
	case RL_ACTION_BEST_FIT_EAGER:
		return RL_ACTION_BEST_FIT;
	case RL_ACTION_FLAG_AFFINITY_EAGER:
		return RL_ACTION_FLAG_AFFINITY;
	case RL_ACTION_LARGEST_FIT_EAGER:
		return RL_ACTION_LARGEST_FIT;
	default:
		return action;
	}
}

static inline u8 rl_mode_to_baseline_action(int mode)
{
	switch (mode) {
	case RL_MODE_FIRST_FIT:
		return RL_ACTION_FIRST_FIT;
	case RL_MODE_BEST_FIT:
	case RL_MODE_TABLE:
	default:
		return RL_ACTION_BEST_FIT;
	}
}

static inline bool rl_req_has(u32 req_flags, u32 flag)
{
	return (req_flags & flag) == flag;
}

static inline u32 rl_req_flags_from_gfp(gfp_t flags, size_t size)
{
	u32 req_flags = 0;

	if (flags & __GFP_DIRECT_RECLAIM)
		req_flags |= RL_REQ_SYNC;
	if ((flags & __GFP_KSWAPD_RECLAIM) && !(flags & __GFP_DIRECT_RECLAIM))
		req_flags |= RL_REQ_ASYNC;
	if (flags & __GFP_RECLAIMABLE)
		req_flags |= RL_REQ_RECLAIMABLE;
	if (flags & __GFP_MOVABLE)
		req_flags |= RL_REQ_MOVABLE;
	if (size > PAGE_SIZE)
		req_flags |= RL_REQ_HIGH_ORDER;

	return req_flags;
}

void *rl_alloc(size_t size, gfp_t flags);
void *rl_alloc_ex(size_t size, gfp_t flags, u32 req_flags);
void rl_free(void *ptr);

#endif /* RL_ALLOC_H */
