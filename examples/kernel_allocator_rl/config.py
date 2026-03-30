from __future__ import annotations

from dataclasses import dataclass


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
