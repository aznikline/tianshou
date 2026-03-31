# RL Allocator Kernel Module Prototype

This directory contains a Linux 5.x style kernel module that exposes a self-managed allocator:

- `rl_alloc(size, flags)`
- `rl_free(ptr)`

The module does not replace global `kmalloc`/`kfree`. It manages private pools and can run in:

- `first_fit`
- `best_fit`
- `rl_table`

## Build

On a Linux host with kernel headers installed:

```bash
make -C /lib/modules/$(uname -r)/build M=$PWD/kernel/rl_allocator modules
```

## Load

```bash
sudo insmod kernel/rl_allocator/rl_allocator.ko mode=1 pool_bytes=1048576
```

## sysfs Control

After loading, the module creates:

```text
/sys/kernel/rl_allocator/mode
/sys/kernel/rl_allocator/policy_version
/sys/kernel/rl_allocator/policy_blob
```

Switch modes:

```bash
echo rl_table | sudo tee /sys/kernel/rl_allocator/mode
echo best_fit | sudo tee /sys/kernel/rl_allocator/mode
```

Load a policy blob:

```bash
sudo dd if=policy.bin of=/sys/kernel/rl_allocator/policy_blob bs=4096
```

Unload:

```bash
sudo rmmod rl_allocator
```
