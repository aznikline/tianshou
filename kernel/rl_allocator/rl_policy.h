#ifndef RL_POLICY_H
#define RL_POLICY_H

#include <linux/gfp.h>
#include <linux/types.h>

#include "rl_alloc.h"

struct rl_policy_blob_header {
	u8 magic[RL_POLICY_MAGIC_SIZE];
	__le32 version;
	__le32 entry_count;
	__le32 checksum;
} __packed;

struct rl_policy {
	u32 version;
	u32 entry_count;
	u32 checksum;
	u8 *table;
};

u32 rl_policy_checksum(const u8 *table, size_t entry_count);
int rl_policy_validate_blob(const void *buf, size_t len);
int rl_policy_from_blob(const void *buf, size_t len, gfp_t gfp_mask,
			struct rl_policy **out_policy);
int rl_policy_lookup(const struct rl_policy *policy, u32 state_key, u8 *action);
void rl_policy_destroy(struct rl_policy *policy);

#endif /* RL_POLICY_H */
