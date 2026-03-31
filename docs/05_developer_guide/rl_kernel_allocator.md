# RL Kernel Allocator Prototype

## Overview

This prototype splits the problem into two cooperating pieces:

- user-space RL infrastructure under `examples/kernel_allocator_rl/`
- kernel module code under `kernel/rl_allocator/`

The user-space side replays traces, trains a policy offline, and exports a compact table. The kernel side hosts a self-managed allocator that computes a bounded integer state key and does constant-time policy lookup with safe fallback.

## Directory Layout

User-space:

- `examples/kernel_allocator_rl/config.py`
- `examples/kernel_allocator_rl/trace.py`
- `examples/kernel_allocator_rl/simulator.py`
- `examples/kernel_allocator_rl/env.py`
- `examples/kernel_allocator_rl/train_grpo.py`
- `examples/kernel_allocator_rl/train_dqn.py`
- `examples/kernel_allocator_rl/policy_export.py`
- `examples/kernel_allocator_rl/generate_trace.py`
- `examples/kernel_allocator_rl/data/sample_trace.csv`

Kernel-space:

- `kernel/rl_allocator/rl_alloc.h`
- `kernel/rl_allocator/rl_policy.h`
- `kernel/rl_allocator/rl_policy.c`
- `kernel/rl_allocator/rl_pool.h`
- `kernel/rl_allocator/rl_pool.c`
- `kernel/rl_allocator/rl_module.c`
- `kernel/rl_allocator/bench/bench.sh`
- `kernel/rl_allocator/bench/stress_alloc.py`
- `kernel/rl_allocator/tools/make_policy_blob.py`

## User-Space Workflow

Generate a synthetic trace:

```bash
python3 examples/kernel_allocator_rl/generate_trace.py \
  --output examples/kernel_allocator_rl/data/generated_trace.csv \
  --events 128
```

Validate the training entrypoint:

```bash
python3 examples/kernel_allocator_rl/train_grpo.py \
  --trace examples/kernel_allocator_rl/data/sample_trace.csv \
  --dry-run
```

The legacy `train_dqn.py` path is kept as a compatibility wrapper and forwards to the GRPO entrypoint.

Run user-space tests:

```bash
.venv/bin/python -m pytest examples/kernel_allocator_rl/tests -q
```

## Policy Format

The deployable policy blob uses this binary layout:

- magic: `RLP1`
- version: 32-bit little-endian
- entry count: 32-bit little-endian
- checksum: 32-bit little-endian sum of payload bytes
- payload: one byte per discrete state key

Inspect or rewrite a blob:

```bash
python3 kernel/rl_allocator/tools/make_policy_blob.py policy.bin --output policy.v2.bin --version 2
```

## Kernel Module Workflow

Build on a Linux host with kernel headers:

```bash
make -C /lib/modules/$(uname -r)/build M=$PWD/kernel/rl_allocator modules
```

Load:

```bash
sudo insmod kernel/rl_allocator/rl_allocator.ko mode=1 pool_bytes=1048576 max_blocks=4096
```

Switch modes:

```bash
echo first_fit | sudo tee /sys/kernel/rl_allocator/mode
echo best_fit | sudo tee /sys/kernel/rl_allocator/mode
echo rl_table | sudo tee /sys/kernel/rl_allocator/mode
```

Load a policy blob:

```bash
sudo dd if=policy.bin of=/sys/kernel/rl_allocator/policy_blob bs=4096
```

Unload:

```bash
sudo rmmod rl_allocator
```

## Benchmarking

For a portable smoke benchmark on the simulator:

```bash
bash kernel/rl_allocator/bench/bench.sh
```

This script:

- attempts a Linux module build when a kernel build tree exists,
- replays the sample trace under `first_fit`, `best_fit`, and `rl_table`,
- prints JSON metrics for average latency, P95, P99, failures, and average fragmentation.

## Real Trace Integration

Expected CSV schema:

```text
ts,cpu,op,ptr_id,size,flags
```

To plug in a real trace later:

1. convert the trace into the CSV schema above,
2. point `train_grpo.py --trace` to the converted file,
3. export the resulting policy table,
4. inspect or rewrite the blob with `make_policy_blob.py`,
5. load it into `/sys/kernel/rl_allocator/policy_blob`.

## Known Limitations

- The current host environment may not provide a Linux kernel build tree, so `.ko` compilation must be verified on Linux.
- The benchmark script currently replays traces through the user-space simulator for portability.
- The module prototype manages private pools; it does not replace global kernel allocators.
- The training entrypoint currently exposes a validated `--dry-run` path; full GRPO execution depends on a Linux/PyTorch-capable runtime with the appropriate dependencies installed.
