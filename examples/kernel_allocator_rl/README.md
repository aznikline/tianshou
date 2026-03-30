# RL Kernel Allocator User-Space Prototype

This directory contains the user-space side of the RL-guided kernel allocator prototype:

- trace loading,
- allocator simulation,
- Gym-style environment wrapper,
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
python3 examples/kernel_allocator_rl/train_dqn.py \
  --trace examples/kernel_allocator_rl/data/sample_trace.csv \
  --dry-run
```

Run the user-space tests:

```bash
.venv/bin/python -m pytest examples/kernel_allocator_rl/tests -q
```
