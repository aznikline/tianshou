from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from examples.kernel_allocator_rl.config import parse_request_flags


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
                flags=parse_request_flags(row["flags"]),
            )
            for row in reader
        ]
