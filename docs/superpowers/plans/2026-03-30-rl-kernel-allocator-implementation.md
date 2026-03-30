# RL Kernel Allocator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained RL-guided kernel allocator prototype with a Python simulator/training pipeline, table export path, Linux 5.x kernel module, benchmark scripts, and reproducible documentation.

**Architecture:** The implementation is split into a user-space side and a kernel-space side. The user-space side provides a trace-driven simulator, DQN training pipeline, and policy-table exporter. The kernel-space side implements a bounded free-list allocator over private memory pools, constant-time table lookup, sysfs-based policy management, and safe fallback to baseline heuristics.

**Tech Stack:** Python 3, pytest, Tianshou/PyTorch for training, Linux kernel module C, Kbuild/Makefile, shell scripts for build and benchmark automation.

---

## File Structure

- Create: `examples/kernel_allocator_rl/README.md`
- Create: `examples/kernel_allocator_rl/__init__.py`
- Create: `examples/kernel_allocator_rl/config.py`
- Create: `examples/kernel_allocator_rl/trace.py`
- Create: `examples/kernel_allocator_rl/simulator.py`
- Create: `examples/kernel_allocator_rl/env.py`
- Create: `examples/kernel_allocator_rl/policy_export.py`
- Create: `examples/kernel_allocator_rl/train_dqn.py`
- Create: `examples/kernel_allocator_rl/generate_trace.py`
- Create: `examples/kernel_allocator_rl/data/sample_trace.csv`
- Create: `examples/kernel_allocator_rl/scripts/reproduce_training.sh`
- Create: `examples/kernel_allocator_rl/tests/test_trace.py`
- Create: `examples/kernel_allocator_rl/tests/test_simulator.py`
- Create: `examples/kernel_allocator_rl/tests/test_env.py`
- Create: `examples/kernel_allocator_rl/tests/test_policy_export.py`
- Create: `examples/kernel_allocator_rl/tests/test_train_dqn.py`
- Create: `kernel/rl_allocator/Makefile`
- Create: `kernel/rl_allocator/Kbuild`
- Create: `kernel/rl_allocator/README.md`
- Create: `kernel/rl_allocator/rl_alloc.h`
- Create: `kernel/rl_allocator/rl_policy.h`
- Create: `kernel/rl_allocator/rl_policy.c`
- Create: `kernel/rl_allocator/rl_pool.h`
- Create: `kernel/rl_allocator/rl_pool.c`
- Create: `kernel/rl_allocator/rl_module.c`
- Create: `kernel/rl_allocator/bench/bench.sh`
- Create: `kernel/rl_allocator/bench/stress_alloc.py`
- Create: `kernel/rl_allocator/tools/make_policy_blob.py`
- Create: `docs/05_developer_guide/rl_kernel_allocator.md`

### Task 1: Scaffold the user-space simulator package

**Files:**
- Create: `examples/kernel_allocator_rl/__init__.py`
- Create: `examples/kernel_allocator_rl/config.py`
- Create: `examples/kernel_allocator_rl/trace.py`
- Create: `examples/kernel_allocator_rl/tests/test_trace.py`

- [ ] **Step 1: Write the failing trace parsing and bucket-config tests**

```python
from pathlib import Path

from examples.kernel_allocator_rl.config import BucketConfig
from examples.kernel_allocator_rl.trace import TraceEvent, load_trace_csv


def test_bucket_config_provides_expected_request_edges() -> None:
    cfg = BucketConfig.default()
    assert cfg.request_size_edges[:4] == (16, 32, 64, 128)
    assert cfg.request_size_edges[-1] == 4096


def test_load_trace_csv_parses_alloc_and_free_rows(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.csv"
    trace_path.write_text(
        "ts,cpu,op,ptr_id,size,flags\n"
        "1,0,alloc,a0,64,0\n"
        "2,0,free,a0,0,0\n",
        encoding="utf-8",
    )

    rows = load_trace_csv(trace_path)

    assert rows == [
        TraceEvent(ts=1, cpu=0, op="alloc", ptr_id="a0", size=64, flags=0),
        TraceEvent(ts=2, cpu=0, op="free", ptr_id="a0", size=0, flags=0),
    ]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python3 -m pytest examples/kernel_allocator_rl/tests/test_trace.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors for `examples.kernel_allocator_rl`.

- [ ] **Step 3: Write the minimal package, config, and trace loader implementation**

```python
from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass(frozen=True)
class BucketConfig:
    request_size_edges: tuple[int, ...]

    @classmethod
    def default(cls) -> "BucketConfig":
        return cls(request_size_edges=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096))


@dataclass(frozen=True)
class TraceEvent:
    ts: int
    cpu: int
    op: str
    ptr_id: str
    size: int
    flags: int


def load_trace_csv(path: str | Path) -> list[TraceEvent]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            TraceEvent(
                ts=int(row["ts"]),
                cpu=int(row["cpu"]),
                op=row["op"],
                ptr_id=row["ptr_id"],
                size=int(row["size"]),
                flags=int(row["flags"]),
            )
            for row in reader
        ]
```

- [ ] **Step 4: Run the trace tests to verify they pass**

Run: `python3 -m pytest examples/kernel_allocator_rl/tests/test_trace.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/kernel_allocator_rl/__init__.py \
  examples/kernel_allocator_rl/config.py \
  examples/kernel_allocator_rl/trace.py \
  examples/kernel_allocator_rl/tests/test_trace.py
git commit -m "feat: scaffold RL allocator trace tooling"
```

### Task 2: Implement the allocator simulator with red-green tests

**Files:**
- Create: `examples/kernel_allocator_rl/simulator.py`
- Create: `examples/kernel_allocator_rl/tests/test_simulator.py`
- Modify: `examples/kernel_allocator_rl/config.py`

- [ ] **Step 1: Write the failing simulator tests for alloc/free, split, coalescing, and metrics**

```python
from examples.kernel_allocator_rl.simulator import AllocatorSimulator


def test_alloc_splits_single_free_block() -> None:
    sim = AllocatorSimulator(pool_bytes=256)
    result = sim.allocate(ptr_id="a0", size=64, action=0)

    assert result.success is True
    assert result.offset == 0
    assert sim.free_bytes == 192
    assert sim.largest_free_block == 192


def test_free_with_eager_coalescing_restores_single_hole() -> None:
    sim = AllocatorSimulator(pool_bytes=256)
    sim.allocate(ptr_id="a0", size=64, action=0)
    sim.allocate(ptr_id="a1", size=64, action=0)
    sim.free(ptr_id="a0", eager_coalesce=True)
    sim.free(ptr_id="a1", eager_coalesce=True)

    assert sim.free_hole_count == 1
    assert sim.free_bytes == 256


def test_state_vector_contains_fragmentation_and_pressure_buckets() -> None:
    sim = AllocatorSimulator(pool_bytes=256)
    sim.allocate(ptr_id="a0", size=64, action=0)
    state = sim.build_state(request_size=64, op="alloc", cpu=0)

    assert "req_bucket" in state
    assert "frag_bucket" in state
    assert "pressure_bucket" in state
```

- [ ] **Step 2: Run the simulator tests to verify they fail**

Run: `python3 -m pytest examples/kernel_allocator_rl/tests/test_simulator.py -q`
Expected: FAIL because `AllocatorSimulator` is not implemented yet.

- [ ] **Step 3: Write the minimal simulator implementation**

```python
from dataclasses import dataclass


@dataclass
class AllocationResult:
    success: bool
    offset: int | None
    scanned: int


class AllocatorSimulator:
    def __init__(self, pool_bytes: int) -> None:
        self.pool_bytes = pool_bytes
        self.free_list = [(0, pool_bytes)]
        self.allocations: dict[str, tuple[int, int]] = {}

    def allocate(self, ptr_id: str, size: int, action: int) -> AllocationResult:
        offset, length = self.free_list[0]
        self.allocations[ptr_id] = (offset, size)
        self.free_list[0] = (offset + size, length - size)
        return AllocationResult(success=True, offset=offset, scanned=1)

    def free(self, ptr_id: str, eager_coalesce: bool) -> None:
        offset, size = self.allocations.pop(ptr_id)
        self.free_list.append((offset, size))
        self.free_list.sort()
        if eager_coalesce:
            self._coalesce()

    def _coalesce(self) -> None:
        merged: list[tuple[int, int]] = []
        for offset, size in self.free_list:
            if not merged:
                merged.append((offset, size))
                continue
            prev_offset, prev_size = merged[-1]
            if prev_offset + prev_size == offset:
                merged[-1] = (prev_offset, prev_size + size)
            else:
                merged.append((offset, size))
        self.free_list = merged
```

- [ ] **Step 4: Extend the implementation until all simulator tests pass**

Add the missing properties and state builder required by the tests:

```python
    @property
    def free_bytes(self) -> int:
        return sum(size for _, size in self.free_list)

    @property
    def largest_free_block(self) -> int:
        return max((size for _, size in self.free_list), default=0)

    @property
    def free_hole_count(self) -> int:
        return len(self.free_list)

    def build_state(self, request_size: int, op: str, cpu: int) -> dict[str, int]:
        return {
            "op": 0 if op == "alloc" else 1,
            "req_bucket": self.bucket_request_size(request_size),
            "frag_bucket": self.bucket_fragmentation(),
            "pressure_bucket": self.bucket_pressure(),
        }
```

- [ ] **Step 5: Run the simulator tests to verify they pass**

Run: `python3 -m pytest examples/kernel_allocator_rl/tests/test_simulator.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add examples/kernel_allocator_rl/config.py \
  examples/kernel_allocator_rl/simulator.py \
  examples/kernel_allocator_rl/tests/test_simulator.py
git commit -m "feat: add RL allocator simulator core"
```

### Task 3: Add the Gym-style environment and policy export path

**Files:**
- Create: `examples/kernel_allocator_rl/env.py`
- Create: `examples/kernel_allocator_rl/policy_export.py`
- Create: `examples/kernel_allocator_rl/tests/test_env.py`
- Create: `examples/kernel_allocator_rl/tests/test_policy_export.py`

- [ ] **Step 1: Write the failing environment and exporter tests**

```python
from pathlib import Path

from examples.kernel_allocator_rl.env import KernelAllocatorEnv
from examples.kernel_allocator_rl.policy_export import export_policy_table
from examples.kernel_allocator_rl.simulator import AllocatorSimulator
from examples.kernel_allocator_rl.trace import TraceEvent


def test_env_step_returns_reward_done_and_info() -> None:
    env = KernelAllocatorEnv(
        trace=[
            TraceEvent(ts=1, cpu=0, op="alloc", ptr_id="a0", size=64, flags=0),
            TraceEvent(ts=2, cpu=0, op="free", ptr_id="a0", size=0, flags=0),
        ],
        simulator=AllocatorSimulator(pool_bytes=256),
    )
    obs, info = env.reset()
    next_obs, reward, terminated, truncated, step_info = env.step(0)

    assert obs.shape == next_obs.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert "latency_ns" in step_info


def test_export_policy_table_writes_magic_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "policy.bin"
    export_policy_table(path, [0, 1, 2, 3], version=1)
    raw = path.read_bytes()

    assert raw[:4] == b"RLP1"
    assert len(raw) > 16
```

- [ ] **Step 2: Run the environment/export tests to verify they fail**

Run: `python3 -m pytest examples/kernel_allocator_rl/tests/test_env.py examples/kernel_allocator_rl/tests/test_policy_export.py -q`
Expected: FAIL because the environment and export functions do not exist yet.

- [ ] **Step 3: Implement the minimal environment API**

```python
import numpy as np
import gymnasium as gym


class KernelAllocatorEnv(gym.Env):
    def __init__(self, trace, simulator) -> None:
        self.trace = trace
        self.simulator = simulator
        self.index = 0
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(8,), dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        self.index = 0
        return self._obs(), {}

    def step(self, action: int):
        event = self.trace[self.index]
        reward, info = self._apply_event(event, action)
        self.index += 1
        terminated = self.index >= len(self.trace)
        return self._obs(), float(reward), terminated, False, info
```

- [ ] **Step 4: Implement the minimal binary policy exporter**

```python
from pathlib import Path
import struct


def export_policy_table(path: str | Path, table: list[int], version: int) -> None:
    header = struct.pack("<4sIII", b"RLP1", version, len(table), sum(table) & 0xFFFFFFFF)
    payload = bytes(table)
    Path(path).write_bytes(header + payload)
```

- [ ] **Step 5: Run the environment/export tests to verify they pass**

Run: `python3 -m pytest examples/kernel_allocator_rl/tests/test_env.py examples/kernel_allocator_rl/tests/test_policy_export.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add examples/kernel_allocator_rl/env.py \
  examples/kernel_allocator_rl/policy_export.py \
  examples/kernel_allocator_rl/tests/test_env.py \
  examples/kernel_allocator_rl/tests/test_policy_export.py
git commit -m "feat: add RL allocator environment and policy export"
```

### Task 4: Add training, synthetic traces, and reproducibility docs

**Files:**
- Create: `examples/kernel_allocator_rl/train_dqn.py`
- Create: `examples/kernel_allocator_rl/generate_trace.py`
- Create: `examples/kernel_allocator_rl/data/sample_trace.csv`
- Create: `examples/kernel_allocator_rl/scripts/reproduce_training.sh`
- Create: `examples/kernel_allocator_rl/README.md`
- Create: `examples/kernel_allocator_rl/tests/test_train_dqn.py`

- [ ] **Step 1: Write the failing training smoke test**

```python
from pathlib import Path
import subprocess
import sys


def test_training_script_supports_dry_run(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.csv"
    trace_path.write_text(
        "ts,cpu,op,ptr_id,size,flags\n"
        "1,0,alloc,a0,64,0\n"
        "2,0,free,a0,0,0\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "examples/kernel_allocator_rl/train_dqn.py",
            "--trace",
            str(trace_path),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "dry-run" in result.stdout.lower()
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `python3 -m pytest examples/kernel_allocator_rl/tests/test_train_dqn.py -q`
Expected: FAIL because `train_dqn.py` does not exist yet.

- [ ] **Step 3: Implement the training and synthetic trace scaffolding**

```python
import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"dry-run training using trace: {args.trace}")
        return 0
    raise SystemExit("full training path not implemented in this step")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add the README, sample trace, and reproduce script**

Add a sample trace file like:

```text
ts,cpu,op,ptr_id,size,flags
1,0,alloc,a0,64,0
2,0,alloc,a1,32,0
3,0,free,a0,0,0
4,0,free,a1,0,0
```

Add a reproduce script like:

```bash
#!/usr/bin/env bash
set -eu
python3 examples/kernel_allocator_rl/train_dqn.py \
  --trace examples/kernel_allocator_rl/data/sample_trace.csv \
  --dry-run
```

- [ ] **Step 5: Run the smoke test to verify it passes**

Run: `python3 -m pytest examples/kernel_allocator_rl/tests/test_train_dqn.py -q`
Expected: PASS for the dry-run coverage added in this task.

- [ ] **Step 6: Commit**

```bash
git add examples/kernel_allocator_rl/train_dqn.py \
  examples/kernel_allocator_rl/generate_trace.py \
  examples/kernel_allocator_rl/data/sample_trace.csv \
  examples/kernel_allocator_rl/scripts/reproduce_training.sh \
  examples/kernel_allocator_rl/README.md \
  examples/kernel_allocator_rl/tests/test_train_dqn.py
git commit -m "feat: add RL allocator training scaffolding"
```

### Task 5: Implement the kernel policy loader and allocator core

**Files:**
- Create: `kernel/rl_allocator/rl_alloc.h`
- Create: `kernel/rl_allocator/rl_policy.h`
- Create: `kernel/rl_allocator/rl_policy.c`
- Create: `kernel/rl_allocator/rl_pool.h`
- Create: `kernel/rl_allocator/rl_pool.c`

- [ ] **Step 1: Write the failing kernel build for missing allocator and policy symbols**

Create a minimal `kernel/rl_allocator/Makefile` and `kernel/rl_allocator/Kbuild`:

```make
obj-m += rl_allocator.o
rl_allocator-y := rl_module.o rl_pool.o rl_policy.o
```

Run: `make -C /lib/modules/$(uname -r)/build M=$PWD/kernel/rl_allocator modules`
Expected: FAIL because `rl_module.c`, `rl_pool.c`, and `rl_policy.c` are not all implemented yet.

- [ ] **Step 2: Implement the policy header and lookup path**

Add interfaces similar to:

```c
struct rl_policy_blob_header {
	u8 magic[4];
	__le32 version;
	__le32 action_count;
	__le32 checksum;
};

struct rl_policy {
	u32 version;
	u32 action_count;
	u8 *table;
};

int rl_policy_lookup(const struct rl_policy *policy, u32 state_key, u8 *action);
int rl_policy_validate_blob(const void *buf, size_t len);
```

- [ ] **Step 3: Implement the allocator pool core**

Add structures and helpers similar to:

```c
struct rl_block {
	u32 offset;
	u32 size;
	u32 flags;
	struct list_head list;
};

struct rl_pool {
	spinlock_t lock;
	void *base;
	u32 total_bytes;
	u32 free_bytes;
	struct list_head free_list;
	struct list_head used_list;
};

void *rl_pool_alloc(struct rl_pool *pool, size_t size, u8 action, u64 *latency_ns);
int rl_pool_free(struct rl_pool *pool, void *ptr, bool eager_coalesce, u64 *latency_ns);
```

- [ ] **Step 4: Add state-key computation and fallback action handling**

Implement a bounded decision path:

```c
u32 rl_pool_build_state_key(const struct rl_pool *pool, size_t size, bool is_free);
u8 rl_pool_select_action(const struct rl_pool *pool, size_t size, bool is_free);
```

Rules:
- if no active policy, return baseline action;
- if lookup fails, return baseline action;
- if candidate chosen by the action does not exist, return baseline action.

- [ ] **Step 5: Run the module build to verify it passes**

Run: `make -C /lib/modules/$(uname -r)/build M=$PWD/kernel/rl_allocator modules`
Expected: PASS and produce `kernel/rl_allocator/rl_allocator.ko`

- [ ] **Step 6: Commit**

```bash
git add kernel/rl_allocator/Makefile \
  kernel/rl_allocator/Kbuild \
  kernel/rl_allocator/rl_alloc.h \
  kernel/rl_allocator/rl_policy.h \
  kernel/rl_allocator/rl_policy.c \
  kernel/rl_allocator/rl_pool.h \
  kernel/rl_allocator/rl_pool.c
git commit -m "feat: add RL allocator kernel core"
```

### Task 6: Implement the kernel module entrypoints, sysfs, and tools

**Files:**
- Create: `kernel/rl_allocator/rl_module.c`
- Create: `kernel/rl_allocator/tools/make_policy_blob.py`
- Create: `kernel/rl_allocator/README.md`

- [ ] **Step 1: Write the failing sysfs and module lifecycle build**

Add a stub `rl_module.c` with module init/exit declarations only, then run:

Run: `make -C /lib/modules/$(uname -r)/build M=$PWD/kernel/rl_allocator modules`
Expected: FAIL because module symbols and exported interfaces are incomplete.

- [ ] **Step 2: Implement module lifecycle and allocator mode control**

Add code skeleton like:

```c
static int rl_mode = RL_MODE_BEST_FIT;
module_param_named(mode, rl_mode, int, 0644);

static int __init rl_allocator_init(void)
{
	return rl_pool_init_all();
}

static void __exit rl_allocator_exit(void)
{
	rl_pool_destroy_all();
}
```

- [ ] **Step 3: Implement sysfs attributes for mode, version, and policy loading**

Expose attributes similar to:

```c
static ssize_t mode_show(...);
static ssize_t mode_store(...);
static ssize_t policy_version_show(...);
static ssize_t policy_blob_store(...);
```

The store path must:
- validate header,
- reject malformed blobs,
- allocate a replacement table,
- atomically publish the new policy,
- keep fallback mode available if loading fails.

- [ ] **Step 4: Implement the userspace policy-blob helper and module README**

The helper should:
- read the exported binary table,
- wrap or verify the kernel blob header,
- print the version and entry count,
- write an output blob suitable for sysfs loading.

The README should cover:
- kernel build command,
- `insmod` command,
- switching modes,
- loading a policy blob,
- unloading the module.

- [ ] **Step 5: Run the module build again to verify it passes**

Run: `make -C /lib/modules/$(uname -r)/build M=$PWD/kernel/rl_allocator modules`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add kernel/rl_allocator/rl_module.c \
  kernel/rl_allocator/tools/make_policy_blob.py \
  kernel/rl_allocator/README.md
git commit -m "feat: add RL allocator module control plane"
```

### Task 7: Add benchmark, stress, and final documentation

**Files:**
- Create: `kernel/rl_allocator/bench/bench.sh`
- Create: `kernel/rl_allocator/bench/stress_alloc.py`
- Create: `docs/05_developer_guide/rl_kernel_allocator.md`

- [ ] **Step 1: Write the failing benchmark script smoke check**

Create a smoke expectation:

```bash
bash kernel/rl_allocator/bench/bench.sh --help
```

Expected: FAIL because the benchmark script does not exist yet.

- [ ] **Step 2: Implement the benchmark and stress scripts**

`bench.sh` should:
- build the module,
- optionally create/load a policy blob,
- run mode-specific benchmark loops,
- print throughput, average latency, P95, P99, failures, and fragmentation metrics.

`stress_alloc.py` should:
- spawn concurrent workers,
- mix alloc/free requests,
- emit a CSV or JSON summary for post-processing.

- [ ] **Step 3: Write the developer guide**

The guide must include:
- architecture summary,
- directory layout,
- training commands,
- policy export commands,
- kernel build/load commands,
- benchmark commands,
- known limitations,
- how to plug in a real trace later.

- [ ] **Step 4: Run the new Python tests**

Run: `python3 -m pytest examples/kernel_allocator_rl/tests -q`
Expected: PASS

- [ ] **Step 5: Run the kernel build smoke test**

Run: `make -C /lib/modules/$(uname -r)/build M=$PWD/kernel/rl_allocator modules`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add kernel/rl_allocator/bench/bench.sh \
  kernel/rl_allocator/bench/stress_alloc.py \
  docs/05_developer_guide/rl_kernel_allocator.md
git commit -m "docs: add RL allocator benchmark and usage guide"
```

### Task 8: Final verification and cleanup

**Files:**
- Modify: `examples/kernel_allocator_rl/README.md`
- Modify: `kernel/rl_allocator/README.md`
- Modify: `docs/05_developer_guide/rl_kernel_allocator.md`

- [ ] **Step 1: Run the full user-space verification**

Run: `python3 -m pytest examples/kernel_allocator_rl/tests -q`
Expected: PASS

- [ ] **Step 2: Run the full kernel build verification**

Run: `make -C /lib/modules/$(uname -r)/build M=$PWD/kernel/rl_allocator clean modules`
Expected: PASS

- [ ] **Step 3: Run the reproducibility smoke script**

Run: `bash examples/kernel_allocator_rl/scripts/reproduce_training.sh`
Expected: PASS and print the dry-run or training summary.

- [ ] **Step 4: Review docs for consistency**

Check that:
- directory names match the repository,
- commands use the same file names used in code,
- policy blob references match the implemented binary header,
- kernel mode names are identical in docs and code.

- [ ] **Step 5: Commit**

```bash
git add examples/kernel_allocator_rl/README.md \
  kernel/rl_allocator/README.md \
  docs/05_developer_guide/rl_kernel_allocator.md
git commit -m "chore: finalize RL allocator prototype verification"
```
