#!/usr/bin/env bash
set -eu

python3 examples/kernel_allocator_rl/train_dqn.py \
  --trace examples/kernel_allocator_rl/data/sample_trace.csv \
  --dry-run
