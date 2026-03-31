from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

from examples.kernel_allocator_rl.config import RL_ACTION_COUNT, action_is_eager
from examples.kernel_allocator_rl.simulator import AllocatorSimulator
from examples.kernel_allocator_rl.trace import TraceEvent

try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight test envs
    gym = None

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight test envs
    np = None


class Observation(list[int]):
    @property
    def shape(self) -> tuple[int]:
        return (len(self),)


@dataclass(frozen=True)
class DiscreteSpace:
    n: int


@dataclass(frozen=True)
class BoxSpace:
    shape: tuple[int, ...]


@dataclass(frozen=True)
class RewardWeights:
    fragmentation: float = 0.45
    latency: float = 0.35
    throughput: float = 0.20
    failure: float = 1.00


class KernelAllocatorEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        trace: list[TraceEvent],
        simulator: AllocatorSimulator,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        self.trace = trace
        self._simulator_type = type(simulator)
        self._simulator_kwargs = {
            "pool_bytes": simulator.pool_bytes,
            "bucket_config": simulator.bucket_config,
        }
        self.simulator = simulator
        self.reward_weights = reward_weights or RewardWeights()
        self.index = 0
        if gym is not None and np is not None:
            self.action_space = gym.spaces.Discrete(RL_ACTION_COUNT)
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(9,),
                dtype=np.int32,
            )
        else:
            self.action_space = DiscreteSpace(n=RL_ACTION_COUNT)
            self.observation_space = BoxSpace(shape=(9,))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if gym is not None:
            super_reset = getattr(super(), "reset", None)
            if callable(super_reset):
                super_reset(seed=seed)
        del options
        self.index = 0
        self.simulator = self._simulator_type(**self._simulator_kwargs)
        return self._obs(), {}

    def step(self, action: int):
        event = self.trace[self.index]
        reward, info = self._apply_event(event, action)
        self.index += 1
        terminated = self.index >= len(self.trace)
        return self._obs(), float(reward), terminated, False, info

    def _obs(self):
        if not self.trace:
            values = [0] * 9
            return np.array(values, dtype=np.int32) if np is not None else Observation(values)
        event = self.trace[min(self.index, len(self.trace) - 1)]
        request_flags = event.flags
        if event.op == "free" and not request_flags:
            request_flags = self.simulator.request_flags_for_ptr(event.ptr_id)
        state = self.simulator.build_state(
            request_size=event.size,
            op=event.op,
            cpu=event.cpu,
            request_flags=request_flags,
        )
        values = [
            state.get("op", 0),
            state.get("req_bucket", 0),
            state.get("frag_bucket", 0),
            state.get("pressure_bucket", 0),
            state.get("req_flags", 0),
            self.simulator.free_hole_count,
            self.simulator.free_bytes,
            self.simulator.largest_free_block,
            min(self.index, 255),
        ]
        return np.array(values, dtype=np.int32) if np is not None else Observation(values)

    def _apply_event(self, event: TraceEvent, action: int) -> tuple[float, dict[str, int | bool]]:
        before_frag = self.simulator.fragmentation_ratio()
        started_ns = perf_counter_ns()
        request_flags = event.flags
        if event.op == "free" and not request_flags:
            request_flags = self.simulator.request_flags_for_ptr(event.ptr_id)
        if event.op == "alloc":
            result = self.simulator.allocate(
                ptr_id=event.ptr_id,
                size=event.size,
                action=action,
                request_flags=request_flags,
            )
            success = result.success
            eager_coalesce = action_is_eager(action)
        else:
            eager_coalesce = action_is_eager(action)
            self.simulator.free(ptr_id=event.ptr_id, eager_coalesce=eager_coalesce)
            success = True
        latency_ns = perf_counter_ns() - started_ns
        frag_delta = self.simulator.fragmentation_ratio() - before_frag
        reward = (
            -self.reward_weights.fragmentation * frag_delta
            -self.reward_weights.latency * (latency_ns / 1_000_000.0)
            + self.reward_weights.throughput * (1.0 if success else 0.0)
            - self.reward_weights.failure * (0.0 if success else 1.0)
        )
        return reward, {
            "latency_ns": latency_ns,
            "success": success,
            "eager_coalesce": eager_coalesce,
            "request_flags": request_flags,
        }
