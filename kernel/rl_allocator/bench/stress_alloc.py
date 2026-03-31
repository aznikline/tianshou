from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter_ns

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.kernel_allocator_rl.policy_export import POLICY_HEADER, POLICY_MAGIC
from examples.kernel_allocator_rl.simulator import AllocatorSimulator
from examples.kernel_allocator_rl.trace import TraceEvent, load_trace_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay allocator traces and emit benchmark metrics.")
    parser.add_argument("--trace", required=True, help="Trace CSV to replay.")
    parser.add_argument(
        "--mode",
        choices=("first_fit", "best_fit", "rl_table"),
        default="best_fit",
        help="Allocator mode to benchmark.",
    )
    parser.add_argument("--policy", help="Optional binary policy table for rl_table mode.")
    parser.add_argument("--pool-bytes", type=int, default=4096, help="Simulator pool size.")
    parser.add_argument("--iterations", type=int, default=10, help="Replay loop count.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of key=value lines.")
    return parser


def load_policy_table(path: str | None) -> bytes | None:
    if not path:
        return None
    raw = Path(path).read_bytes()
    magic, _version, entry_count, _checksum = POLICY_HEADER.unpack_from(raw, 0)
    if magic != POLICY_MAGIC:
        raise ValueError(f"unexpected policy magic: {magic!r}")
    payload = raw[POLICY_HEADER.size :]
    if len(payload) != entry_count:
        raise ValueError("policy payload length mismatch")
    return payload


def bucket_holes(count: int) -> int:
    if count <= 1:
        return 0
    if count <= 3:
        return 1
    if count <= 7:
        return 2
    if count <= 15:
        return 3
    return 4


def bucket_mix(event: TraceEvent) -> int:
    return 0 if event.op == "alloc" else 2


def build_state_key(simulator: AllocatorSimulator, event: TraceEvent) -> int:
    request_flags = event.flags
    if event.op == "free" and not request_flags:
        request_flags = simulator.request_flags_for_ptr(event.ptr_id)
    state = simulator.build_state(
        request_size=event.size,
        op=event.op,
        cpu=event.cpu,
        request_flags=request_flags,
    )
    key = state["op"]
    key = key * 10 + state["req_bucket"]
    key = key * 6 + state["frag_bucket"]
    key = key * 5 + bucket_holes(simulator.free_hole_count)
    key = key * 5 + state["pressure_bucket"]
    key = key * 3 + bucket_mix(event)
    key = key * 128 + state["req_flags"]
    return key


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]


def pick_action(mode: str, simulator: AllocatorSimulator, event: TraceEvent, policy: bytes | None) -> int:
    if mode == "first_fit":
        return 0
    if mode == "best_fit":
        return 1
    if not policy:
        return 1
    key = build_state_key(simulator, event)
    if key >= len(policy):
        return 1
    return policy[key]


def replay(trace: list[TraceEvent], mode: str, policy: bytes | None, pool_bytes: int, iterations: int) -> dict[str, float | int | str]:
    latencies: list[int] = []
    failures = 0
    fragmentation_samples: list[float] = []

    for _ in range(iterations):
        simulator = AllocatorSimulator(pool_bytes=pool_bytes)
        for event in trace:
            action = pick_action(mode, simulator, event, policy)
            started_ns = perf_counter_ns()
            if event.op == "alloc":
                result = simulator.allocate(
                    ptr_id=event.ptr_id,
                    size=event.size,
                    action=action,
                    request_flags=event.flags,
                )
                if not result.success:
                    failures += 1
            else:
                simulator.free(ptr_id=event.ptr_id, eager_coalesce=action >= 4)
            latencies.append(perf_counter_ns() - started_ns)
            fragmentation_samples.append(simulator.fragmentation_ratio())

    return {
        "mode": mode,
        "iterations": iterations,
        "events": len(trace) * iterations,
        "failures": failures,
        "avg_latency_ns": int(mean(latencies)) if latencies else 0,
        "p95_latency_ns": percentile(latencies, 95),
        "p99_latency_ns": percentile(latencies, 99),
        "avg_fragmentation": round(mean(fragmentation_samples), 6) if fragmentation_samples else 0.0,
    }


def main() -> int:
    args = build_parser().parse_args()
    trace = load_trace_csv(args.trace)
    policy = load_policy_table(args.policy)
    metrics = replay(trace, args.mode, policy, args.pool_bytes, args.iterations)
    if args.json:
        print(json.dumps(metrics, sort_keys=True))
    else:
        for key, value in metrics.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
