# RL Kernel Allocator Design

**Date:** 2026-03-30

**Status:** Approved for prototype implementation

## 1. Goal

Build a Linux 5.x compatible prototype of a reinforcement-learning-guided memory allocator that:

- runs as a loadable kernel module,
- manages its own memory pools instead of replacing the global kernel allocator,
- learns allocation and free/coalescing policies offline in user space,
- deploys a deterministic, constant-time policy table in kernel space,
- falls back safely to traditional allocation heuristics whenever the policy is invalid, unavailable, or too costly to use.

The prototype is intended to demonstrate a safe path from trace-driven RL training to kernel deployment while preserving deterministic inference latency and production-like guardrails.

## 2. Non-Goals

The prototype explicitly does not:

- replace global `kmalloc` or `kfree`,
- patch or hook SLAB, SLUB, or the buddy allocator,
- run floating-point or full neural-network inference in kernel space,
- learn online inside the kernel,
- depend on external kernel-space libraries beyond standard Linux kernel headers and Kbuild.

## 3. Deliverable Layout

The implementation will live in isolated directories so the existing `tianshou` codebase remains untouched:

- `kernel/rl_allocator/`
  - loadable kernel module source,
  - policy table format and sysfs hooks,
  - Kbuild/Makefile support,
  - kernel-space benchmark and stress helpers.
- `examples/kernel_allocator_rl/`
  - user-space allocator simulator,
  - trace loader and synthetic trace generator,
  - RL training script,
  - policy export tool,
  - benchmark and reproducibility scripts.
- `docs/superpowers/specs/2026-03-30-rl-kernel-allocator-design.md`
  - this design document.

## 4. System Overview

The system has four layers:

1. **Trace and simulation layer**
   - Replays allocation and deallocation traces in user space.
   - Mirrors the kernel allocator module's data structures and behaviors.

2. **Training layer**
   - Trains a discrete-action RL policy using a trace-driven simulator.
   - Produces a compact state-to-action table.

3. **Kernel policy layer**
   - Loads a validated policy table into the module.
   - Computes a fixed-size discrete state key and returns an action in O(1).

4. **Allocator execution layer**
   - Applies the action only through validated kernel allocator operations.
   - Falls back to baseline heuristics if the policy output is invalid or unavailable.

## 5. Kernel Allocator Scope

### 5.1 Prototype allocator model

The kernel module owns one or more pre-reserved memory pools and exports:

- `void *rl_alloc(size_t size, gfp_t flags);`
- `void rl_free(void *ptr);`

The allocator uses a classic variable-sized free-list allocator inside those pools:

- block headers,
- free-list links,
- split on allocation,
- coalescing on free or deferred coalescing based on policy/baseline mode,
- allocation bookkeeping to detect double-free and invalid-pointer cases.

### 5.2 Baseline modes

The module must support three execution modes over the same pool implementation:

- `first_fit`
- `best_fit`
- `rl_table`

Using one allocator implementation with switchable policies keeps the benchmark comparison fair.

## 6. State Formulation

The deployed policy must be a table lookup. Therefore the state is discretized into a compact integer key.

### 6.1 State inputs

Each allocation or free decision uses a subset of the following features:

- request size bucket,
- free bytes bucket,
- largest free block bucket,
- external fragmentation bucket,
- free-hole count bucket,
- small-hole ratio bucket,
- recent alloc/free mix bucket over a fixed window,
- recent average request size bucket,
- current CPU-local pool pressure bucket,
- operation type bucket (`alloc` or `free`).

### 6.2 Bucket definitions

The implementation will use small bounded bucket sets such as:

- request size:
  - `<=16`, `<=32`, `<=64`, `<=128`, `<=256`, `<=512`, `<=1024`, `<=2048`, `<=4096`, `>4096`
- fragmentation ratio:
  - derived from `1 - largest_free_block / total_free_bytes`
  - bucketed into a fixed number of ranges such as `0`, `(0,0.1]`, `(0.1,0.25]`, `(0.25,0.5]`, `(0.5,0.75]`, `>0.75`
- free-hole count:
  - `0-1`, `2-3`, `4-7`, `8-15`, `16+`
- small-hole ratio:
  - ratio of free blocks below configurable thresholds.

Exact bucket edges will be centralized in both the simulator and kernel code to keep behavior aligned.

### 6.3 State encoding

The kernel module computes a compact integer key:

```text
state_key =
  (((((op * N1 + req_bucket) * N2 + frag_bucket) * N3 + hole_bucket)
     * N4 + small_hole_bucket) * N5 + pressure_bucket) * N6
  + recent_mix_bucket
```

The exported policy table is indexed by this `state_key`.

## 7. Action Formulation

The action space must stay discrete and fixed-size.

### 7.1 Candidate generation

Before consulting the policy, the allocator computes a bounded candidate set from the free list:

- candidate 0: first fit,
- candidate 1: best fit,
- candidate 2: smallest fitting block within a preferred size window,
- candidate 3: largest fitting block under a bounded scan budget.

This avoids exposing arbitrary free-list indices to the policy while still allowing meaningful choice.

### 7.2 Action set

The initial prototype action set is:

- `ACT_FIRST_FIT`
- `ACT_BEST_FIT`
- `ACT_CANDIDATE_2`
- `ACT_CANDIDATE_3`
- `ACT_FIRST_FIT_EAGER_COALESCE`
- `ACT_BEST_FIT_EAGER_COALESCE`
- `ACT_CANDIDATE_2_EAGER_COALESCE`
- `ACT_CANDIDATE_3_EAGER_COALESCE`

For `free` operations the action influences the post-free coalescing mode:

- deferred coalescing,
- immediate coalescing.

Any chosen candidate is revalidated before use.

## 8. Reward Function

The primary objective is a weighted trade-off between fragmentation, latency, and throughput.

### 8.1 Per-step reward

For each step:

```text
reward =
  - w_frag * fragmentation_delta
  - w_lat  * normalized_latency
  + w_tp   * success_bonus
  - w_fail * allocation_failure_penalty
```

### 8.2 Default weights

The prototype default uses:

- `w_frag = 0.45`
- `w_lat = 0.35`
- `w_tp = 0.20`
- `w_fail = 1.00`

These are defaults, not hard-coded assumptions. The training pipeline will allow overriding them via CLI.

### 8.3 Episode bonuses

At the end of an episode, the simulator adds optional bonuses or penalties for:

- final fragmentation,
- success rate,
- utilization stability,
- tail latency.

## 9. RL Algorithm Choice

### 9.1 Training algorithm

The prototype uses **DQN** in user space because:

- the action space is discrete and moderate,
- the simulator exposes a fixed-width observation vector,
- DQN is straightforward to implement and debug with the existing `tianshou` stack,
- deployment does not require kernel-side neural-network inference.

### 9.2 Deployment representation

The runtime policy deployed in the kernel is **not** a neural network.

Instead:

1. DQN is trained over the simulator state/action space.
2. The learned policy is evaluated over all valid discretized states.
3. The best action for each state is exported into a dense or sparse table.
4. The kernel only loads and uses the table.

This yields deterministic lookup latency and simple validation logic.

## 10. Trace-Driven Training

### 10.1 Trace format

The user-space tooling will support a CSV trace format:

```text
ts,cpu,op,ptr_id,size,flags
```

Where:

- `ts`: logical or real timestamp,
- `cpu`: originating CPU id,
- `op`: `alloc` or `free`,
- `ptr_id`: trace-local allocation identifier,
- `size`: request size in bytes,
- `flags`: optional request metadata.

For real traces that are not yet available, the tooling will also generate synthetic traces matching realistic allocation mixes.

### 10.2 Replay guarantees

The simulator must reproduce:

- split behavior,
- free-list insertion order,
- coalescing semantics,
- pool exhaustion behavior,
- invalid free detection,
- candidate generation logic,
- state bucketization.

The simulator and kernel module will share the same constants where practical, and mirrored tests will check parity.

## 11. Kernel Inference Requirements

### 11.1 Determinism

The inference path must:

- perform no dynamic memory allocation,
- perform no sleeping operations,
- perform no floating-point math,
- complete in constant bounded time,
- remain comfortably below the `<1us` policy decision budget on typical hardware.

### 11.2 Inference pipeline

For each allocator decision:

1. sample current allocator metrics,
2. bucketize into discrete state fields,
3. compute `state_key`,
4. read action from the current policy table,
5. validate action and candidate availability,
6. execute action or fall back.

## 12. Fallback and Safety Rules

The allocator always remains guarded by baseline heuristics.

Fallback is triggered when:

- no policy is loaded,
- policy mode is disabled,
- `state_key` is out of range,
- action id is invalid,
- selected candidate does not exist,
- validation detects corrupted allocator metadata,
- sysfs update leaves no usable active policy,
- a bounded scan budget is exceeded.

Fallback target is configurable but defaults to `best_fit`.

## 13. Concurrency and Multi-CPU Scalability

### 13.1 Per-CPU pool model

The prototype uses per-CPU pools by default:

- each CPU has a local pool instance,
- allocation prefers the local pool,
- pool-local metrics feed the RL state,
- each pool has its own lock to reduce contention.

### 13.2 Cross-CPU free handling

When memory allocated on one CPU is freed on another:

- ownership metadata identifies the source pool,
- the freeing CPU does not directly mutate the foreign free list under nested locks,
- instead it appends to a bounded deferred-return queue or uses a lock-safe transfer path,
- the owner pool later reconciles returned blocks.

This avoids deadlocks and keeps lock ordering simple.

### 13.3 Locking model

The module will use:

- spinlocks for pool metadata,
- IRQ-safe variants where needed by the interface contract,
- strict lock ordering,
- no sleeping while holding allocator locks.

## 14. Hot Policy Update

### 14.1 sysfs interface

The module will expose a sysfs interface for:

- current mode (`first_fit`, `best_fit`, `rl_table`),
- active policy version,
- policy statistics,
- loading a new policy blob,
- forcing fallback mode.

### 14.2 Update protocol

Policy updates proceed in this order:

1. user space writes a policy blob,
2. kernel validates header, size, version, and checksum,
3. kernel allocates and populates a replacement policy object,
4. kernel atomically swaps the active pointer,
5. the old policy object is retired safely.

The swap must not require reboot and must never leave the allocator without a valid baseline path.

## 15. Security Considerations

The policy must never directly control raw addresses or unchecked offsets.

Safety rules include:

- action ids map only to predefined allocator operations,
- every chosen candidate is validated against allocator metadata,
- pointer arithmetic uses checked bounds,
- free only accepts pointers inside managed pools,
- double-free and invalid-free detection are mandatory,
- model blob loading validates length, magic, version, and checksum,
- malformed policy data results in rejection, not partial activation.

## 16. Testing Strategy

### 16.1 User-space tests

- simulator unit tests,
- trace replay determinism tests,
- policy export/import tests,
- bucketization parity tests,
- RL training smoke tests on a short synthetic trace.

### 16.2 Kernel-space tests

- module build verification,
- allocator functional tests,
- policy load/unload tests,
- fallback-path tests,
- cross-CPU stress tests,
- long-run fragmentation and latency measurements.

### 16.3 Benchmarks

The benchmark output must compare:

- `first_fit`
- `best_fit`
- `rl_table`

Metrics:

- allocation throughput,
- average latency,
- P95 latency,
- P99 latency,
- failure rate,
- external fragmentation ratio,
- largest free block ratio,
- hole count growth.

## 17. Reproducibility

The repository will include:

- a sample synthetic trace,
- commands to train on synthetic traces,
- a documented path for plugging in a real trace later,
- a policy export format with versioning,
- a benchmark script for the kernel module prototype.

## 18. Implementation Risks

Key risks and mitigations:

- **State mismatch between simulator and kernel**
  - Mitigation: parity tests and shared constants.
- **Policy overfitting to a trace mix**
  - Mitigation: synthetic and mixed-trace evaluation.
- **Kernel latency regression from candidate search**
  - Mitigation: bounded scan budget and fixed candidate count.
- **Cross-CPU ownership complexity**
  - Mitigation: per-CPU pools with deferred return queues.
- **Unsafe model updates**
  - Mitigation: atomic swap, checksum, and always-available fallback.

## 19. Success Criteria

The prototype is considered successful if it provides:

- a buildable Linux 5.x loadable module,
- a trace-driven user-space training pipeline,
- a table-based deployable policy format,
- a safe fallback path,
- benchmark output comparing RL-guided and baseline heuristics,
- documentation sufficient to rebuild, retrain, and redeploy the policy.
