from __future__ import annotations

import argparse
import csv
from pathlib import Path


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
                {"ts": ts, "cpu": 0, "op": "free", "ptr_id": ptr_id, "size": 0, "flags": 0},
            )
            continue
        ptr_id = f"a{alloc_index}"
        size = 2 ** (4 + (alloc_index % 5))
        rows.append(
            {"ts": ts, "cpu": 0, "op": "alloc", "ptr_id": ptr_id, "size": size, "flags": 0},
        )
        outstanding.append(ptr_id)
        alloc_index += 1
    return rows


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
