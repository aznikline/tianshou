from __future__ import annotations

from dataclasses import dataclass

RL_REQ_SYNC = 1 << 0
RL_REQ_ASYNC = 1 << 1
RL_REQ_ANON = 1 << 2
RL_REQ_FILE = 1 << 3
RL_REQ_RECLAIMABLE = 1 << 4
RL_REQ_MOVABLE = 1 << 5
RL_REQ_HIGH_ORDER = 1 << 6

RL_REQ_FLAG_NAMES = {
    "sync": RL_REQ_SYNC,
    "async": RL_REQ_ASYNC,
    "anon": RL_REQ_ANON,
    "file": RL_REQ_FILE,
    "reclaimable": RL_REQ_RECLAIMABLE,
    "movable": RL_REQ_MOVABLE,
    "high_order": RL_REQ_HIGH_ORDER,
}

RL_ACTION_FIRST_FIT = 0
RL_ACTION_BEST_FIT = 1
RL_ACTION_FLAG_AFFINITY = 2
RL_ACTION_LARGEST_FIT = 3
RL_ACTION_FIRST_FIT_EAGER = 4
RL_ACTION_BEST_FIT_EAGER = 5
RL_ACTION_FLAG_AFFINITY_EAGER = 6
RL_ACTION_LARGEST_FIT_EAGER = 7
RL_ACTION_SYNC_COMPACT = 8
RL_ACTION_ASYNC_DEFER = 9
RL_ACTION_ANON_AFFINITY = 10
RL_ACTION_FILE_AFFINITY = 11
RL_ACTION_RECLAIM_REUSE = 12
RL_ACTION_MOVABLE_SPREAD = 13
RL_ACTION_HIGH_ORDER_GUARD = 14
RL_ACTION_SEMANTIC_DEFAULT = 15
RL_ACTION_COUNT = 16


@dataclass(frozen=True)
class BucketConfig:
    request_size_edges: tuple[int, ...]
    fragmentation_edges: tuple[float, ...]
    pressure_edges: tuple[float, ...]

    @classmethod
    def default(cls) -> "BucketConfig":
        return cls(
            request_size_edges=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
            fragmentation_edges=(0.0, 0.1, 0.25, 0.5, 0.75),
            pressure_edges=(0.25, 0.5, 0.75, 0.9),
        )

    @staticmethod
    def bucket_for_int(value: int, edges: tuple[int, ...]) -> int:
        for idx, edge in enumerate(edges):
            if value <= edge:
                return idx
        return len(edges)

    @staticmethod
    def bucket_for_float(value: float, edges: tuple[float, ...]) -> int:
        for idx, edge in enumerate(edges):
            if value <= edge:
                return idx
        return len(edges)


def parse_request_flags(value: int | str) -> int:
    if isinstance(value, int):
        return value

    text = value.strip().lower()
    if not text:
        return 0
    if text.isdigit():
        return int(text)

    flags = 0
    for token in text.split("|"):
        name = token.strip()
        if not name:
            continue
        if name not in RL_REQ_FLAG_NAMES:
            raise ValueError(f"unknown request flag token: {name}")
        flags |= RL_REQ_FLAG_NAMES[name]
    return flags


def request_has(flags: int, flag: int) -> bool:
    return (flags & flag) == flag


def action_is_eager(action: int) -> bool:
    return action in {
        RL_ACTION_FIRST_FIT_EAGER,
        RL_ACTION_BEST_FIT_EAGER,
        RL_ACTION_FLAG_AFFINITY_EAGER,
        RL_ACTION_LARGEST_FIT_EAGER,
        RL_ACTION_SYNC_COMPACT,
    }
