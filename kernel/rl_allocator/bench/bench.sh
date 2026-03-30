#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TRACE_PATH="${TRACE_PATH:-$ROOT_DIR/examples/kernel_allocator_rl/data/sample_trace.csv}"
ITERATIONS="${ITERATIONS:-25}"
POLICY_PATH="${POLICY_PATH:-}"

if [[ "$(uname -s)" == "Linux" && -d "/lib/modules/$(uname -r)/build" ]]; then
  echo "[bench] kernel headers detected, attempting module build"
  make -C "/lib/modules/$(uname -r)/build" M="$ROOT_DIR/kernel/rl_allocator" modules
else
  echo "[bench] Linux kernel build tree not available on this host; skipping .ko build"
fi

for mode in first_fit best_fit rl_table; do
  echo "[bench] mode=${mode}"
  if [[ "$mode" == "rl_table" && -n "$POLICY_PATH" ]]; then
    python3 "$ROOT_DIR/kernel/rl_allocator/bench/stress_alloc.py" \
      --trace "$TRACE_PATH" \
      --mode "$mode" \
      --policy "$POLICY_PATH" \
      --iterations "$ITERATIONS" \
      --json
  else
    python3 "$ROOT_DIR/kernel/rl_allocator/bench/stress_alloc.py" \
      --trace "$TRACE_PATH" \
      --mode "$mode" \
      --iterations "$ITERATIONS" \
      --json
  fi
done
