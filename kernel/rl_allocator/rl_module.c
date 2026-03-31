#include <linux/cpu.h>
#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/kobject.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/rcupdate.h>
#include <linux/slab.h>
#include <linux/smp.h>
#include <linux/string.h>
#include <linux/sysfs.h>

#include "rl_alloc.h"
#include "rl_pool.h"
#include "rl_policy.h"

static int rl_mode = RL_MODE_BEST_FIT;
module_param_named(mode, rl_mode, int, 0644);
MODULE_PARM_DESC(mode, "Allocator mode: 0=first_fit, 1=best_fit, 2=rl_table");

static ulong rl_pool_bytes = 1UL << 20;
module_param_named(pool_bytes, rl_pool_bytes, ulong, 0644);
MODULE_PARM_DESC(pool_bytes, "Bytes reserved per allocator pool");

static uint rl_max_blocks = 4096;
module_param_named(max_blocks, rl_max_blocks, uint, 0644);
MODULE_PARM_DESC(max_blocks, "Maximum block descriptors per pool");

static uint rl_pool_count;
module_param_named(pool_count, rl_pool_count, uint, 0644);
MODULE_PARM_DESC(pool_count, "Number of allocator pools to create");

static DEFINE_MUTEX(rl_control_lock);
static struct rl_pool *rl_pools;
static struct kobject *rl_kobj;
static struct rl_policy *rl_active_policy;

static struct rl_pool *rl_current_pool(void)
{
	u32 cpu = raw_smp_processor_id();

	if (!rl_pools || !rl_pool_count)
		return NULL;

	return &rl_pools[cpu % rl_pool_count];
}

static struct rl_pool *rl_find_owner_pool(void *ptr)
{
	u32 i;

	for (i = 0; i < rl_pool_count; i++) {
		char *base = rl_pools[i].base;

		if (!base)
			continue;
		if ((char *)ptr >= base && (char *)ptr < base + rl_pools[i].total_bytes)
			return &rl_pools[i];
	}

	return NULL;
}

static void rl_publish_policy(struct rl_policy *policy)
{
	u32 i;

	for (i = 0; i < rl_pool_count; i++)
		rl_pool_set_policy(&rl_pools[i], policy);
}

static void rl_set_baseline_action_all(u8 action)
{
	u32 i;

	for (i = 0; i < rl_pool_count; i++)
		WRITE_ONCE(rl_pools[i].baseline_action, action);
}

void *rl_alloc(size_t size, gfp_t flags)
{
	return rl_alloc_ex(size, flags, rl_req_flags_from_gfp(flags, size));
}
EXPORT_SYMBOL_GPL(rl_alloc);

void *rl_alloc_ex(size_t size, gfp_t flags, u32 req_flags)
{
	struct rl_pool *pool;
	u8 action;

	(void)flags;
	pool = rl_current_pool();
	if (!pool)
		return NULL;

	if (READ_ONCE(rl_mode) == RL_MODE_TABLE)
		action = rl_pool_select_action(pool, size, false, req_flags);
	else
		action = rl_mode_to_baseline_action(READ_ONCE(rl_mode));

	return rl_pool_alloc(pool, size, action, req_flags, NULL);
}
EXPORT_SYMBOL_GPL(rl_alloc_ex);

void rl_free(void *ptr)
{
	struct rl_pool *pool;
	bool eager = false;
	u8 action;
	u32 req_flags = 0;

	if (!ptr)
		return;

	pool = rl_find_owner_pool(ptr);
	if (!pool)
		return;

	req_flags = rl_pool_request_flags_for_ptr(pool, ptr);
	if (READ_ONCE(rl_mode) == RL_MODE_TABLE) {
		action = rl_pool_select_action(pool, 0, true, req_flags);
		eager = rl_action_is_eager(action);
	}

	rl_pool_free(pool, ptr, eager, NULL);
}
EXPORT_SYMBOL_GPL(rl_free);

static ssize_t mode_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
	(void)kobj;
	(void)attr;

	switch (READ_ONCE(rl_mode)) {
	case RL_MODE_FIRST_FIT:
		return sysfs_emit(buf, "first_fit\n");
	case RL_MODE_BEST_FIT:
		return sysfs_emit(buf, "best_fit\n");
	case RL_MODE_TABLE:
		return sysfs_emit(buf, "rl_table\n");
	default:
		return sysfs_emit(buf, "unknown\n");
	}
}

static ssize_t mode_store(struct kobject *kobj, struct kobj_attribute *attr,
			  const char *buf, size_t count)
{
	u8 baseline;

	(void)kobj;
	(void)attr;

	mutex_lock(&rl_control_lock);
	if (sysfs_streq(buf, "first_fit") || sysfs_streq(buf, "0"))
		WRITE_ONCE(rl_mode, RL_MODE_FIRST_FIT);
	else if (sysfs_streq(buf, "best_fit") || sysfs_streq(buf, "1"))
		WRITE_ONCE(rl_mode, RL_MODE_BEST_FIT);
	else if (sysfs_streq(buf, "rl_table") || sysfs_streq(buf, "2"))
		WRITE_ONCE(rl_mode, RL_MODE_TABLE);
	else {
		mutex_unlock(&rl_control_lock);
		return -EINVAL;
	}

	baseline = rl_mode_to_baseline_action(READ_ONCE(rl_mode));
	rl_set_baseline_action_all(baseline);
	mutex_unlock(&rl_control_lock);

	return count;
}

static struct kobj_attribute mode_attr = __ATTR(mode, 0644, mode_show, mode_store);

static ssize_t policy_version_show(struct kobject *kobj, struct kobj_attribute *attr,
				   char *buf)
{
	u32 version = 0;

	(void)kobj;
	(void)attr;

	mutex_lock(&rl_control_lock);
	if (rl_active_policy)
		version = rl_active_policy->version;
	mutex_unlock(&rl_control_lock);

	return sysfs_emit(buf, "%u\n", version);
}

static struct kobj_attribute policy_version_attr = __ATTR_RO(policy_version);

static ssize_t policy_blob_write(struct file *file, struct kobject *kobj,
				 struct bin_attribute *attr, char *buf,
				 loff_t off, size_t count)
{
	struct rl_policy *new_policy = NULL;
	struct rl_policy *old_policy;
	int err;

	(void)file;
	(void)kobj;
	(void)attr;

	if (off != 0)
		return -EINVAL;

	mutex_lock(&rl_control_lock);
	err = rl_policy_from_blob(buf, count, GFP_KERNEL, &new_policy);
	if (err) {
		mutex_unlock(&rl_control_lock);
		return err;
	}

	old_policy = rl_active_policy;
	rl_active_policy = new_policy;
	rl_publish_policy(new_policy);
	mutex_unlock(&rl_control_lock);

	synchronize_rcu();
	rl_policy_destroy(old_policy);
	return count;
}

static struct bin_attribute policy_blob_attr = {
	.attr = { .name = "policy_blob", .mode = 0200 },
	.write = policy_blob_write,
};

static struct attribute *rl_attrs[] = {
	&mode_attr.attr,
	&policy_version_attr.attr,
	NULL,
};

static const struct attribute_group rl_attr_group = {
	.attrs = rl_attrs,
};

static int __init rl_allocator_init(void)
{
	u8 baseline = rl_mode_to_baseline_action(rl_mode);
	u32 i;
	int err;

	if (!rl_pool_count)
		rl_pool_count = max_t(uint, num_possible_cpus(), 1U);

	rl_pools = kcalloc(rl_pool_count, sizeof(*rl_pools), GFP_KERNEL);
	if (!rl_pools)
		return -ENOMEM;

	for (i = 0; i < rl_pool_count; i++) {
		err = rl_pool_init(&rl_pools[i], rl_pool_bytes, rl_max_blocks, baseline,
				   GFP_KERNEL);
		if (err)
			goto err_destroy_pools;
	}

	rl_kobj = kobject_create_and_add("rl_allocator", kernel_kobj);
	if (!rl_kobj) {
		err = -ENOMEM;
		goto err_destroy_pools;
	}

	err = sysfs_create_group(rl_kobj, &rl_attr_group);
	if (err)
		goto err_kobj_put;

	err = sysfs_create_bin_file(rl_kobj, &policy_blob_attr);
	if (err)
		goto err_remove_group;

	pr_info("rl_allocator: initialized %u pools of %lu bytes each\n",
		rl_pool_count, rl_pool_bytes);
	return 0;

err_remove_group:
	sysfs_remove_group(rl_kobj, &rl_attr_group);
err_kobj_put:
	kobject_put(rl_kobj);
	rl_kobj = NULL;
err_destroy_pools:
	while (i > 0) {
		i--;
		rl_pool_destroy(&rl_pools[i]);
	}
	kfree(rl_pools);
	rl_pools = NULL;
	return err;
}

static void __exit rl_allocator_exit(void)
{
	u32 i;

	if (rl_kobj) {
		sysfs_remove_bin_file(rl_kobj, &policy_blob_attr);
		sysfs_remove_group(rl_kobj, &rl_attr_group);
		kobject_put(rl_kobj);
		rl_kobj = NULL;
	}

	for (i = 0; i < rl_pool_count; i++)
		rl_pool_destroy(&rl_pools[i]);
	kfree(rl_pools);
	rl_pools = NULL;

	synchronize_rcu();
	rl_policy_destroy(rl_active_policy);
	rl_active_policy = NULL;
}

module_init(rl_allocator_init);
module_exit(rl_allocator_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("OpenAI Codex");
MODULE_DESCRIPTION("RL-guided self-managed kernel allocator prototype");
