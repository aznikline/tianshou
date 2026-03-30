from __future__ import annotations

from dataclasses import dataclass

from examples.kernel_allocator_rl.config import BucketConfig


@dataclass(frozen=True)
class AllocationResult:
    success: bool
    offset: int | None
    scanned: int


class AllocatorSimulator:
    def __init__(
        self,
        pool_bytes: int,
        bucket_config: BucketConfig | None = None,
    ) -> None:
        self.pool_bytes = pool_bytes
        self.bucket_config = bucket_config or BucketConfig.default()
        self.free_list: list[tuple[int, int]] = [(0, pool_bytes)]
        self.allocations: dict[str, tuple[int, int]] = {}

    @property
    def free_bytes(self) -> int:
        return sum(size for _, size in self.free_list)

    @property
    def largest_free_block(self) -> int:
        return max((size for _, size in self.free_list), default=0)

    @property
    def free_hole_count(self) -> int:
        return len(self.free_list)

    def allocate(self, ptr_id: str, size: int, action: int) -> AllocationResult:
        del action
        for index, (offset, block_size) in enumerate(self.free_list):
            if block_size < size:
                continue
            self.allocations[ptr_id] = (offset, size)
            remainder = block_size - size
            if remainder == 0:
                self.free_list.pop(index)
            else:
                self.free_list[index] = (offset + size, remainder)
            return AllocationResult(success=True, offset=offset, scanned=index + 1)
        return AllocationResult(success=False, offset=None, scanned=len(self.free_list))

    def free(self, ptr_id: str, eager_coalesce: bool) -> None:
        offset, size = self.allocations.pop(ptr_id)
        self.free_list.append((offset, size))
        self.free_list.sort()
        if eager_coalesce:
            self._coalesce()

    def build_state(self, request_size: int, op: str, cpu: int) -> dict[str, int]:
        del cpu
        return {
            "op": 0 if op == "alloc" else 1,
            "req_bucket": self.bucket_request_size(request_size),
            "frag_bucket": self.bucket_fragmentation(),
            "pressure_bucket": self.bucket_pressure(),
        }

    def bucket_request_size(self, request_size: int) -> int:
        return BucketConfig.bucket_for_int(
            request_size,
            self.bucket_config.request_size_edges,
        )

    def bucket_fragmentation(self) -> int:
        return BucketConfig.bucket_for_float(
            self.fragmentation_ratio(),
            self.bucket_config.fragmentation_edges,
        )

    def bucket_pressure(self) -> int:
        pressure = 1.0 - (self.free_bytes / self.pool_bytes if self.pool_bytes else 0.0)
        return BucketConfig.bucket_for_float(pressure, self.bucket_config.pressure_edges)

    def fragmentation_ratio(self) -> float:
        if self.free_bytes == 0:
            return 1.0
        return 1.0 - (self.largest_free_block / self.free_bytes)

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
