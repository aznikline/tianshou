#ifndef RL_ALLOC_H
#define RL_ALLOC_H

#include <linux/gfp_types.h>
#include <linux/types.h>

#define RL_POLICY_MAGIC "RLP1"
#define RL_POLICY_MAGIC_SIZE 4

enum rl_mode {
	RL_MODE_FIRST_FIT = 0,
	RL_MODE_BEST_FIT = 1,
	RL_MODE_TABLE = 2,
};

enum rl_action {
	RL_ACTION_FIRST_FIT = 0,
	RL_ACTION_BEST_FIT = 1,
	RL_ACTION_CANDIDATE_2 = 2,
	RL_ACTION_CANDIDATE_3 = 3,
	RL_ACTION_FIRST_FIT_EAGER = 4,
	RL_ACTION_BEST_FIT_EAGER = 5,
	RL_ACTION_CANDIDATE_2_EAGER = 6,
	RL_ACTION_CANDIDATE_3_EAGER = 7,
	RL_ACTION_MAX = 8,
};

static inline bool rl_action_is_eager(u8 action)
{
	return action >= RL_ACTION_FIRST_FIT_EAGER;
}

static inline u8 rl_action_base(u8 action)
{
	return rl_action_is_eager(action) ? action - RL_ACTION_FIRST_FIT_EAGER : action;
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

void *rl_alloc(size_t size, gfp_t flags);
void rl_free(void *ptr);

#endif /* RL_ALLOC_H */
