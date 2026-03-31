# RL Kernel Allocator User-Space Prototype

This directory contains the user-space side of the RL-guided kernel allocator prototype:

- trace loading,
- allocator simulation,
- Gym-style environment wrapper,
- semantic request flags for `sync`, `async`, `anon`, `file`, `reclaimable`, `movable`, and `high_order`,
- policy table export,
- synthetic trace generation,
- training entrypoint scaffolding.

## Quick Start

Generate a synthetic trace:

```bash
python3 examples/kernel_allocator_rl/generate_trace.py \
  --output examples/kernel_allocator_rl/data/generated_trace.csv \
  --events 64
```

Run the dry-run training pipeline:

```bash
python3 examples/kernel_allocator_rl/train_grpo.py \
  --trace examples/kernel_allocator_rl/data/sample_trace.csv \
  --dry-run
```

`train_dqn.py` remains available as a compatibility wrapper and forwards to the GRPO entrypoint.

Run the user-space tests:

```bash
.venv/bin/python -m pytest examples/kernel_allocator_rl/tests -q
```

## Trace Flags

The `flags` column accepts either an integer bitmask or symbolic tokens joined by `|`, for example:

```text
sync|anon
async|file|reclaimable
sync|movable|high_order
```

The simulator and benchmark pipeline use these flags both in the state representation and in the expanded 16-action allocator policy.
