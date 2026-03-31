from __future__ import annotations

from dataclasses import dataclass

from examples.kernel_allocator_rl.config import (
    BucketConfig,
    RL_ACTION_ANON_AFFINITY,
    RL_ACTION_ASYNC_DEFER,
    RL_ACTION_BEST_FIT,
    RL_ACTION_BEST_FIT_EAGER,
    RL_ACTION_FILE_AFFINITY,
    RL_ACTION_FIRST_FIT,
    RL_ACTION_FIRST_FIT_EAGER,
    RL_ACTION_FLAG_AFFINITY,
    RL_ACTION_FLAG_AFFINITY_EAGER,
    RL_ACTION_HIGH_ORDER_GUARD,
    RL_ACTION_LARGEST_FIT,
    RL_ACTION_LARGEST_FIT_EAGER,
    RL_ACTION_MOVABLE_SPREAD,
    RL_ACTION_RECLAIM_REUSE,
    RL_ACTION_SEMANTIC_DEFAULT,
    RL_ACTION_SYNC_COMPACT,
    RL_REQ_ANON,
    RL_REQ_ASYNC,
    RL_REQ_FILE,
    RL_REQ_HIGH_ORDER,
    RL_REQ_MOVABLE,
    RL_REQ_RECLAIMABLE,
    RL_REQ_SYNC,
    action_is_eager,
    request_has,
)


@dataclass(frozen=True)
class AllocationResult:
    success: bool
    offset: int | None
    scanned: int


@dataclass
class MemoryBlock:
    offset: int
    size: int
    tags: int = 0


@dataclass(frozen=True)
class AllocationRecord:
    offset: int
    size: int
    request_flags: int


class AllocatorSimulator:
    def __init__(
        self,
        pool_bytes: int,
        bucket_config: BucketConfig | None = None,
    ) -> None:
        self.pool_bytes = pool_bytes
        self.bucket_config = bucket_config or BucketConfig.default()
        self.free_list: list[MemoryBlock] = [MemoryBlock(offset=0, size=pool_bytes)]
        self.allocations: dict[str, AllocationRecord] = {}

    @property
    def free_bytes(self) -> int:
        return sum(block.size for block in self.free_list)

    @property
    def largest_free_block(self) -> int:
        return max((block.size for block in self.free_list), default=0)

    @property
    def free_hole_count(self) -> int:
        return len(self.free_list)

    def allocate(
        self,
        ptr_id: str,
        size: int,
        action: int,
        request_flags: int = 0,
    ) -> AllocationResult:
        block, scanned = self._select_block(size=size, action=action, request_flags=request_flags)
        if block is None:
            return AllocationResult(success=False, offset=None, scanned=scanned)

        for index, free_block in enumerate(self.free_list):
            if free_block is not block:
                continue
            self.allocations[ptr_id] = AllocationRecord(
                offset=block.offset,
                size=size,
                request_flags=request_flags,
            )
            remainder = block.size - size
            if remainder == 0:
                self.free_list.pop(index)
            else:
                self.free_list[index] = MemoryBlock(
                    offset=block.offset + size,
                    size=remainder,
                    tags=block.tags,
                )
            return AllocationResult(success=True, offset=block.offset, scanned=scanned)
        return AllocationResult(success=False, offset=None, scanned=scanned)

    def free(self, ptr_id: str, eager_coalesce: bool) -> None:
        record = self.allocations.pop(ptr_id)
        self.free_list.append(
            MemoryBlock(
                offset=record.offset,
                size=record.size,
                tags=record.request_flags,
            ),
        )
        self.free_list.sort(key=lambda block: block.offset)
        if eager_coalesce:
            self._coalesce()

    def build_state(self, request_size: int, op: str, cpu: int, request_flags: int = 0) -> dict[str, int]:
        del cpu
        return {
            "op": 0 if op == "alloc" else 1,
            "req_bucket": self.bucket_request_size(request_size),
            "frag_bucket": self.bucket_fragmentation(),
            "pressure_bucket": self.bucket_pressure(),
            "req_flags": request_flags,
        }

    def request_flags_for_ptr(self, ptr_id: str) -> int:
        record = self.allocations.get(ptr_id)
        return 0 if record is None else record.request_flags

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
        merged: list[MemoryBlock] = []
        for block in self.free_list:
            if not merged:
                merged.append(MemoryBlock(offset=block.offset, size=block.size, tags=block.tags))
                continue
            prev_block = merged[-1]
            if prev_block.offset + prev_block.size == block.offset:
                merged[-1] = MemoryBlock(
                    offset=prev_block.offset,
                    size=prev_block.size + block.size,
                    tags=prev_block.tags | block.tags,
                )
            else:
                merged.append(MemoryBlock(offset=block.offset, size=block.size, tags=block.tags))
        self.free_list = merged

    def _select_block(self, size: int, action: int, request_flags: int) -> tuple[MemoryBlock | None, int]:
        fitting = [block for block in self.free_list if block.size >= size]
        if not fitting:
            return None, len(self.free_list)

        semantic_action = self._resolve_semantic_action(action, request_flags)
        if semantic_action in {RL_ACTION_FIRST_FIT, RL_ACTION_FIRST_FIT_EAGER, RL_ACTION_ASYNC_DEFER}:
            return fitting[0], 1
        if semantic_action in {RL_ACTION_BEST_FIT, RL_ACTION_BEST_FIT_EAGER, RL_ACTION_SYNC_COMPACT}:
            block = min(fitting, key=lambda item: (item.size, item.offset))
            return block, self.free_list.index(block) + 1
        if semantic_action in {RL_ACTION_LARGEST_FIT, RL_ACTION_LARGEST_FIT_EAGER}:
            block = max(fitting, key=lambda item: (item.size, -item.offset))
            return block, self.free_list.index(block) + 1
        if semantic_action in {RL_ACTION_FLAG_AFFINITY, RL_ACTION_FLAG_AFFINITY_EAGER}:
            return self._select_by_tag_affinity(fitting, request_flags)
        if semantic_action == RL_ACTION_ANON_AFFINITY:
            return self._select_by_tag_affinity(fitting, RL_REQ_ANON)
        if semantic_action == RL_ACTION_FILE_AFFINITY:
            return self._select_by_tag_affinity(fitting, RL_REQ_FILE)
        if semantic_action == RL_ACTION_RECLAIM_REUSE:
            return self._select_by_tag_affinity(fitting, RL_REQ_RECLAIMABLE)
        if semantic_action == RL_ACTION_MOVABLE_SPREAD:
            block = max(fitting, key=lambda item: (item.offset, item.size))
            return block, self.free_list.index(block) + 1
        if semantic_action == RL_ACTION_HIGH_ORDER_GUARD:
            if request_has(request_flags, RL_REQ_HIGH_ORDER):
                block = max(fitting, key=lambda item: (item.size, -item.offset))
            else:
                block = min(fitting, key=lambda item: (item.size, item.offset))
            return block, self.free_list.index(block) + 1
        block = min(fitting, key=lambda item: (item.size, item.offset))
        return block, self.free_list.index(block) + 1

    def _resolve_semantic_action(self, action: int, request_flags: int) -> int:
        if action != RL_ACTION_SEMANTIC_DEFAULT:
            return action
        if request_has(request_flags, RL_REQ_HIGH_ORDER):
            return RL_ACTION_HIGH_ORDER_GUARD
        if request_has(request_flags, RL_REQ_ASYNC):
            return RL_ACTION_ASYNC_DEFER
        if request_has(request_flags, RL_REQ_SYNC):
            return RL_ACTION_SYNC_COMPACT
        if request_has(request_flags, RL_REQ_FILE):
            return RL_ACTION_FILE_AFFINITY
        if request_has(request_flags, RL_REQ_ANON):
            return RL_ACTION_ANON_AFFINITY
        if request_has(request_flags, RL_REQ_RECLAIMABLE):
            return RL_ACTION_RECLAIM_REUSE
        if request_has(request_flags, RL_REQ_MOVABLE):
            return RL_ACTION_MOVABLE_SPREAD
        return RL_ACTION_BEST_FIT

    def _select_by_tag_affinity(
        self,
        fitting: list[MemoryBlock],
        request_flags: int,
    ) -> tuple[MemoryBlock, int]:
        block = max(
            fitting,
            key=lambda item: (
                self._tag_score(item.tags, request_flags),
                -item.size,
                -item.offset,
            ),
        )
        return block, self.free_list.index(block) + 1

    @staticmethod
    def _tag_score(tags: int, request_flags: int) -> int:
        return (tags & request_flags).bit_count()
