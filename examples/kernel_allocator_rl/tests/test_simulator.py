from examples.kernel_allocator_rl.config import (
    RL_ACTION_FILE_AFFINITY,
    RL_REQ_ANON,
    RL_REQ_FILE,
    RL_REQ_SYNC,
)
from examples.kernel_allocator_rl.simulator import AllocatorSimulator


def test_alloc_splits_single_free_block() -> None:
    sim = AllocatorSimulator(pool_bytes=256)

    result = sim.allocate(ptr_id="a0", size=64, action=0)

    assert result.success is True
    assert result.offset == 0
    assert sim.free_bytes == 192
    assert sim.largest_free_block == 192


def test_free_with_eager_coalescing_restores_single_hole() -> None:
    sim = AllocatorSimulator(pool_bytes=256)

    sim.allocate(ptr_id="a0", size=64, action=0)
    sim.allocate(ptr_id="a1", size=64, action=0)
    sim.free(ptr_id="a0", eager_coalesce=True)
    sim.free(ptr_id="a1", eager_coalesce=True)

    assert sim.free_hole_count == 1
    assert sim.free_bytes == 256


def test_state_vector_contains_fragmentation_and_pressure_buckets() -> None:
    sim = AllocatorSimulator(pool_bytes=256)

    sim.allocate(ptr_id="a0", size=64, action=0, request_flags=RL_REQ_SYNC | RL_REQ_ANON)
    state = sim.build_state(request_size=64, op="alloc", cpu=0, request_flags=RL_REQ_SYNC | RL_REQ_ANON)

    assert "req_bucket" in state
    assert "frag_bucket" in state
    assert "pressure_bucket" in state
    assert state["req_flags"] == (RL_REQ_SYNC | RL_REQ_ANON)


def test_file_affinity_action_prefers_matching_tagged_block() -> None:
    sim = AllocatorSimulator(pool_bytes=256)

    sim.allocate(ptr_id="anon0", size=64, action=0, request_flags=RL_REQ_ANON)
    sim.allocate(ptr_id="file0", size=64, action=0, request_flags=RL_REQ_FILE)
    sim.free(ptr_id="anon0", eager_coalesce=False)
    sim.free(ptr_id="file0", eager_coalesce=False)

    result = sim.allocate(
        ptr_id="file1",
        size=32,
        action=RL_ACTION_FILE_AFFINITY,
        request_flags=RL_REQ_FILE,
    )

    assert result.success is True
    assert result.offset == 64
