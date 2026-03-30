from examples.kernel_allocator_rl.env import KernelAllocatorEnv
from examples.kernel_allocator_rl.simulator import AllocatorSimulator
from examples.kernel_allocator_rl.trace import TraceEvent


def test_env_step_returns_reward_done_and_info() -> None:
    env = KernelAllocatorEnv(
        trace=[
            TraceEvent(ts=1, cpu=0, op="alloc", ptr_id="a0", size=64, flags=0),
            TraceEvent(ts=2, cpu=0, op="free", ptr_id="a0", size=0, flags=0),
        ],
        simulator=AllocatorSimulator(pool_bytes=256),
    )

    obs, _ = env.reset()
    next_obs, reward, terminated, truncated, step_info = env.step(0)

    assert obs.shape == next_obs.shape
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert "latency_ns" in step_info
