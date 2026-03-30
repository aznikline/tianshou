#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/string.h>

#include "rl_policy.h"

u32 rl_policy_checksum(const u8 *table, size_t entry_count)
{
	u32 checksum = 0;
	size_t i;

	for (i = 0; i < entry_count; i++)
		checksum += table[i];

	return checksum;
}

int rl_policy_validate_blob(const void *buf, size_t len)
{
	const struct rl_policy_blob_header *header = buf;
	u32 entry_count;
	u32 checksum;
	const u8 *table;

	if (!buf || len < sizeof(*header))
		return -EINVAL;
	if (memcmp(header->magic, RL_POLICY_MAGIC, RL_POLICY_MAGIC_SIZE))
		return -EINVAL;

	entry_count = le32_to_cpu(header->entry_count);
	if (!entry_count)
		return -EINVAL;
	if (len != sizeof(*header) + entry_count)
		return -EINVAL;

	table = (const u8 *)(header + 1);
	checksum = rl_policy_checksum(table, entry_count);
	if (checksum != le32_to_cpu(header->checksum))
		return -EINVAL;

	return 0;
}

int rl_policy_from_blob(const void *buf, size_t len, gfp_t gfp_mask,
			struct rl_policy **out_policy)
{
	const struct rl_policy_blob_header *header = buf;
	struct rl_policy *policy;
	u32 entry_count;
	int err;

	if (!out_policy)
		return -EINVAL;

	err = rl_policy_validate_blob(buf, len);
	if (err)
		return err;

	entry_count = le32_to_cpu(header->entry_count);

	policy = kzalloc(sizeof(*policy), gfp_mask);
	if (!policy)
		return -ENOMEM;

	policy->table = kmemdup(header + 1, entry_count, gfp_mask);
	if (!policy->table) {
		kfree(policy);
		return -ENOMEM;
	}

	policy->version = le32_to_cpu(header->version);
	policy->entry_count = entry_count;
	policy->checksum = le32_to_cpu(header->checksum);
	*out_policy = policy;

	return 0;
}

int rl_policy_lookup(const struct rl_policy *policy, u32 state_key, u8 *action)
{
	if (!policy || !action)
		return -EINVAL;
	if (state_key >= policy->entry_count)
		return -ENOENT;

	*action = policy->table[state_key];
	if (*action >= RL_ACTION_MAX)
		return -ERANGE;

	return 0;
}

void rl_policy_destroy(struct rl_policy *policy)
{
	if (!policy)
		return;

	kfree(policy->table);
	kfree(policy);
}
