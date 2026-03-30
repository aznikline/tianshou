from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BucketConfig:
    request_size_edges: tuple[int, ...]

    @classmethod
    def default(cls) -> "BucketConfig":
        return cls(
            request_size_edges=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096),
        )
