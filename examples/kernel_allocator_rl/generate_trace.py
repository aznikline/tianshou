from __future__ import annotations

import argparse
import csv
from pathlib import Path

from examples.kernel_allocator_rl.config import (
    RL_REQ_ANON,
    RL_REQ_ASYNC,
    RL_REQ_FILE,
    RL_REQ_FLAG_NAMES,
    RL_REQ_HIGH_ORDER,
    RL_REQ_MOVABLE,
    RL_REQ_RECLAIMABLE,
    RL_REQ_SYNC,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic allocator trace.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--events", type=int, default=32, help="Number of logical operations.")
    return parser


def generate_trace_rows(event_count: int) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    outstanding: list[str] = []
    alloc_index = 0
    for ts in range(1, event_count + 1):
        if outstanding and ts % 3 == 0:
            ptr_id = outstanding.pop(0)
            rows.append(
                {"ts": ts, "cpu": 0, "op": "free", "ptr_id": ptr_id, "size": 0, "flags": "0"},
            )
            continue
        ptr_id = f"a{alloc_index}"
        size = 2 ** (4 + (alloc_index % 5))
        request_flags = _format_flags(
            _pick_request_flags(size=size, alloc_index=alloc_index),
        )
        rows.append(
            {
                "ts": ts,
                "cpu": 0,
                "op": "alloc",
                "ptr_id": ptr_id,
                "size": size,
                "flags": request_flags,
            },
        )
        outstanding.append(ptr_id)
        alloc_index += 1
    return rows


def _pick_request_flags(size: int, alloc_index: int) -> int:
    cycle = alloc_index % 4
    if cycle == 0:
        flags = RL_REQ_SYNC | RL_REQ_ANON
    elif cycle == 1:
        flags = RL_REQ_ASYNC | RL_REQ_FILE | RL_REQ_RECLAIMABLE
    elif cycle == 2:
        flags = RL_REQ_SYNC | RL_REQ_MOVABLE
    else:
        flags = RL_REQ_FILE
    if size > 4096:
        flags |= RL_REQ_HIGH_ORDER
    return flags


def _format_flags(flags: int) -> str:
    names = [name for name, bit in RL_REQ_FLAG_NAMES.items() if flags & bit]
    return "|".join(names) if names else "0"


def main() -> int:
    args = build_parser().parse_args()
    rows = generate_trace_rows(args.events)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ts", "cpu", "op", "ptr_id", "size", "flags"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote synthetic trace with {len(rows)} events to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
