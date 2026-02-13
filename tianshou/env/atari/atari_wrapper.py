# Borrow a lot from openai baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import logging
import warnings
from collections import deque
from typing import Any, SupportsFloat

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import Env

from tianshou.env import BaseVectorEnv
from tianshou.highlevel.env import (
    EnvFactoryRegistered,
    EnvMode,
    EnvPoolFactory,
    VectorEnvType,
)
from tianshou.highlevel.trainer import EpochStopCallback, TrainingContext

envpool_is_available = True
try:
    import envpool
except ImportError:
    envpool_is_available = False
    envpool = None
log = logging.getLogger(__name__)
_ale_namespace_registered = False


def is_envpool_available() -> bool:
    """Return whether envpool is importable in the current runtime."""
    return envpool_is_available


def _ensure_ale_namespace_registered() -> None:
    """Ensure Gymnasium Atari ALE namespace is registered."""
    global _ale_namespace_registered
    if _ale_namespace_registered:
        return
    try:
        import ale_py
    except ImportError as exc:
        raise ImportError(
            "Atari ALE environments require `ale-py`. Install with `pip install ale-py` "
            "or `pip install \"gymnasium[atari]\"`."
        ) from exc
    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)
    else:
        from gymnasium.envs.registration import register_envs

        register_envs(ale_py)
    _ale_namespace_registered = True


def _to_gymnasium_atari_task(task: str) -> str:
    """Normalize Atari task identifiers to Gymnasium's ALE namespace."""
    if task.startswith("ALE/"):
        return task
    if task.endswith("NoFrameskip-v4"):
        game = task.removesuffix("NoFrameskip-v4")
        return f"ALE/{game}-v5"
    if task.endswith("-v5") and "/" not in task:
        return f"ALE/{task}"
    return task


def _to_envpool_atari_task(task: str) -> str:
    """Normalize Atari task identifiers to EnvPool's task format."""
    gymnasium_task = _to_gymnasium_atari_task(task)
    if gymnasium_task.startswith("ALE/"):
        return gymnasium_task.split("/", 1)[1]
    return gymnasium_task.replace("NoFrameskip-v4", "-v5")


def _parse_reset_result(reset_result: tuple) -> tuple[tuple, dict, bool]:
    contains_info = (
        isinstance(reset_result, tuple)
        and len(reset_result) == 2
        and isinstance(reset_result[1], dict)
    )
    if contains_info:
        return reset_result[0], reset_result[1], contains_info
    return reset_result, {}, contains_info


def get_space_dtype(obs_space: gym.spaces.Box) -> np.dtype:
    """Return a concrete numpy dtype accepted by gymnasium.spaces.Box."""
    return np.dtype(obs_space.dtype)


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset.

    No-op is assumed to be action 0.

    :param gym.Env env: the environment to wrap.
    :param int noop_max: the maximum value of no-ops to run.
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert hasattr(env.unwrapped, "get_action_meanings")
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        _, info, return_info = _parse_reset_result(self.env.reset(**kwargs))
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            step_result = self.env.step(self.noop_action)
            if len(step_result) == 4:
                obs, rew, done, info = step_result  # type: ignore[unreachable]  # mypy doesn't know that Gym version <0.26 has only 4 items (no truncation)
            else:
                obs, rew, term, trunc, info = step_result
                done = term or trunc
            if done:
                obs, info, _ = _parse_reset_result(self.env.reset())
        if return_info:
            return obs, info
        return obs, {}


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame (frameskipping) using most recent raw observations (for max pooling across time steps).

    :param gym.Env env: the environment to wrap.
    :param int skip: number of `skip`-th frame.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the environment with the given action.

        Repeat action, sum reward, and max over last observations.
        """
        obs_list = []
        total_reward = 0.0
        new_step_api = False
        for _ in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result  # type: ignore[unreachable]  # mypy doesn't know that Gym version <0.26 has only 4 items (no truncation)
            else:
                obs, reward, term, trunc, info = step_result
                done = term or trunc
                new_step_api = True
            obs_list.append(obs)
            total_reward += float(reward)
            if done:
                break
        max_frame = np.max(obs_list[-2:], axis=0)
        if new_step_api:
            return max_frame, total_reward, term, trunc, info

        return (
            max_frame,
            total_reward,
            done,
            info.get("TimeLimit.truncated", False),
            info,
        )


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over.

    It helps the value estimation.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        self._return_info = False

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result  # type: ignore[unreachable]  # mypy doesn't know that Gym version <0.26 has only 4 items (no truncation)
            new_step_api = False
        else:
            obs, reward, term, trunc, info = step_result
            done = term or trunc
            new_step_api = True
        reward = float(reward)
        self.was_real_done = done
        # check current lives, make loss of life terminal, then update lives to
        # handle bonus lives
        assert hasattr(self.env.unwrapped, "ale")
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condition for a few
            # frames, so its important to keep lives > 0, so that we only reset
            # once the environment is actually done.
            done = True
            term = True
        self.lives = lives
        if new_step_api:
            return obs, reward, term, trunc, info
        return obs, reward, done, info.get("TimeLimit.truncated", False), info

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Calls the Gym environment reset, only when lives are exhausted.

        This way all states are still reachable even though lives are episodic, and
        the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info, self._return_info = _parse_reset_result(self.env.reset(**kwargs))
        else:
            # no-op step to advance from terminal/lost life state
            step_result = self.env.step(0)
            obs, info = step_result[0], step_result[-1]
        assert hasattr(self.env.unwrapped, "ale")
        self.lives = self.env.unwrapped.ale.lives()
        if self._return_info:
            return obs, info
        return obs, {}


class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing.

    Related discussion: https://github.com/openai/baselines/issues/240.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert hasattr(env.unwrapped, "get_action_meanings")
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        _, _, return_info = _parse_reset_result(self.env.reset(**kwargs))
        obs = self.env.step(1)[0]
        return obs, {}


class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.size = 84
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        obs_space_dtype = get_space_dtype(obs_space)
        self.observation_space = gym.spaces.Box(
            low=np.min(obs_space.low),
            high=np.max(obs_space.high),
            shape=(self.size, self.size),
            dtype=obs_space_dtype,
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """Returns the current observation from a frame."""
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize observations to 0~1.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        low = np.min(obs_space.low)
        high = np.max(obs_space.high)
        self.bias = low
        self.scale = high - low
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return (observation - self.bias) / self.scale


class ClipRewardEnv(gym.RewardWrapper):
    """clips the reward to {+1, 0, -1} by its sign.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.reward_range = (-1, 1)

    def reward(self, reward: SupportsFloat) -> int:
        """Bin reward to {+1, 0, -1} by its sign. Note: np.sign(0) == 0."""
        return np.sign(float(reward))


class FrameStack(gym.Wrapper):
    """Stack n_frames last frames.

    :param gym.Env env: the environment to wrap.
    :param int n_frames: the number of frames to stack.
    """

    def __init__(self, env: gym.Env, n_frames: int) -> None:
        super().__init__(env)
        self.n_frames: int = n_frames
        self.frames: deque[tuple[Any, ...]] = deque([], maxlen=n_frames)
        obs_space = env.observation_space
        obs_space_shape = env.observation_space.shape
        assert obs_space_shape is not None
        shape = (n_frames, *obs_space_shape)
        assert isinstance(obs_space, gym.spaces.Box)
        obs_space_dtype = get_space_dtype(obs_space)
        self.observation_space = gym.spaces.Box(
            low=np.min(obs_space.low),
            high=np.max(obs_space.high),
            shape=shape,
            dtype=obs_space_dtype,
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        obs, info, return_info = _parse_reset_result(self.env.reset(**kwargs))
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return (self._get_ob(), info) if return_info else (self._get_ob(), {})

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step_result = self.env.step(action)
        done: bool
        if len(step_result) == 4:
            obs, reward, done, info = step_result  # type: ignore[unreachable] # mypy doesn't know that Gym version <0.26 has only 4 items (no truncation)
            new_step_api = False
        else:
            obs, reward, term, trunc, info = step_result
            new_step_api = True
        self.frames.append(obs)
        reward = float(reward)
        if new_step_api:
            return self._get_ob(), reward, term, trunc, info
        return (
            self._get_ob(),
            reward,
            done,
            info.get("TimeLimit.truncated", False),
            info,
        )

    def _get_ob(self) -> np.ndarray:
        # the original wrapper use `LazyFrames` but since we use np buffer,
        # it has no effect
        return np.stack(self.frames, axis=0)


def wrap_deepmind(
    env: gym.Env,
    episode_life: bool = True,
    clip_rewards: bool = True,
    frame_stack: int = 4,
    scale: bool = False,
    warp_frame: bool = True,
) -> (
    MaxAndSkipEnv
    | EpisodicLifeEnv
    | FireResetEnv
    | WarpFrame
    | ScaledFloatFrame
    | ClipRewardEnv
    | FrameStack
):
    """Configure environment for DeepMind-style Atari.

    The observation is channel-first: (c, h, w) instead of (h, w, c).

    :param env: the Atari environment to wrap.
    :param bool episode_life: wrap the episode life wrapper.
    :param bool clip_rewards: wrap the reward clipping wrapper.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool scale: wrap the scaling observation wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :return: the wrapped atari environment.
    """
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    assert hasattr(env.unwrapped, "get_action_meanings")  # for mypy

    wrapped_env: (
        MaxAndSkipEnv
        | EpisodicLifeEnv
        | FireResetEnv
        | WarpFrame
        | ScaledFloatFrame
        | ClipRewardEnv
        | FrameStack
    ) = env
    if episode_life:
        wrapped_env = EpisodicLifeEnv(wrapped_env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        wrapped_env = FireResetEnv(wrapped_env)
    if warp_frame:
        wrapped_env = WarpFrame(wrapped_env)
    if scale:
        wrapped_env = ScaledFloatFrame(wrapped_env)
    if clip_rewards:
        wrapped_env = ClipRewardEnv(wrapped_env)
    if frame_stack:
        wrapped_env = FrameStack(wrapped_env, frame_stack)
    return wrapped_env


def make_atari_env(
    task: str,
    seed: int,
    num_training_envs: int,
    num_test_envs: int,
    scale: int | bool = False,
    frame_stack: int = 4,
    require_envpool: bool = False,
) -> tuple[Env, BaseVectorEnv, BaseVectorEnv]:
    """Wrapper function for Atari env.

    If EnvPool is installed, it will automatically switch to EnvPool's Atari env.
    Legacy tasks like `PongNoFrameskip-v4` are mapped to Gymnasium's modern
    `ALE/Pong-v5` equivalent.

    :return: a tuple of (single env, training envs, test envs).
    """
    env_factory = make_atari_env_factory(
        task,
        frame_stack,
        scale=bool(scale),
        require_envpool=require_envpool,
    )
    envs = env_factory.create_envs(num_training_envs, num_test_envs, seed=seed)
    return envs.env, envs.training_envs, envs.test_envs


class AtariEnvPoolFactory(EnvPoolFactory):
    """Atari-specific EnvPool creation.

    EnvPool internally handles most wrappers implemented by `wrap_deepmind`, so
    we set equivalent creation kwargs during vector-env construction.
    """

    def __init__(self, frame_stack: int, scale: bool) -> None:
        self.frame_stack = frame_stack
        self.scale = scale
        if self.scale:
            warnings.warn(
                "EnvPool does not include ScaledFloatFrame wrapper, "
                "please compensate by scaling inside your network's forward function (e.g. `x = x / 255.0` for Atari)",
            )

    def _transform_task(self, task: str) -> str:
        return _to_envpool_atari_task(task)

    def _transform_kwargs(self, kwargs: dict, mode: EnvMode) -> dict:
        kwargs = super()._transform_kwargs(kwargs, mode)
        is_train = mode == EnvMode.TRAINING
        kwargs["reward_clip"] = is_train
        kwargs["episodic_life"] = is_train
        kwargs["stack_num"] = self.frame_stack
        return kwargs


class GymnasiumAtariEnvFactory(EnvFactoryRegistered):
    def __init__(
        self,
        task: str,
        frame_stack: int,
        scale: bool = False,
        use_envpool_if_available: bool = True,
        require_envpool: bool = False,
        venv_type: VectorEnvType = VectorEnvType.SUBPROC_SHARED_MEM_AUTO,
    ) -> None:
        normalized_task = _to_gymnasium_atari_task(task)
        if normalized_task.startswith("ALE/"):
            _ensure_ale_namespace_registered()
        self.frame_stack = frame_stack
        self.scale = scale
        envpool_factory = None
        if require_envpool and not envpool_is_available:
            raise RuntimeError(
                "EnvPool is required but not available. Install `envpool` in your current "
                "Python environment, then rerun. Note: prebuilt EnvPool wheels are typically "
                "Linux-only, so macOS often cannot use EnvPool acceleration.",
            )
        if use_envpool_if_available:
            if envpool_is_available:
                envpool_factory = AtariEnvPoolFactory(frame_stack=frame_stack, scale=scale)
                log.info("Using envpool, because it is available")
            else:
                log.info("Not using envpool, because it is not available")
        super().__init__(
            task=normalized_task,
            venv_type=venv_type,
            envpool_factory=envpool_factory,
        )

    def _create_env(self, mode: EnvMode) -> gym.Env:
        kwargs = self._create_kwargs(mode)
        # Preserve NoFrameskip-v4 behavior when using ALE v5 ids.
        kwargs.setdefault("frameskip", 1)
        kwargs.setdefault("repeat_action_probability", 0.0)
        env = gym.make(self.task, **kwargs)
        is_train = mode == EnvMode.TRAINING
        return wrap_deepmind(
            env,
            episode_life=is_train,
            clip_rewards=is_train,
            frame_stack=self.frame_stack,
            scale=self.scale,
        )


def make_atari_env_factory(
    task: str,
    frame_stack: int,
    scale: bool = False,
    use_envpool_if_available: bool = True,
    require_envpool: bool = False,
    venv_type: VectorEnvType = VectorEnvType.SUBPROC_SHARED_MEM_AUTO,
) -> EnvFactoryRegistered:
    """Create a modern Atari env factory based on Gymnasium + optional EnvPool."""
    return GymnasiumAtariEnvFactory(
        task=task,
        frame_stack=frame_stack,
        scale=scale,
        use_envpool_if_available=use_envpool_if_available,
        require_envpool=require_envpool,
        venv_type=venv_type,
    )


class AtariEnvFactory(GymnasiumAtariEnvFactory):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "AtariEnvFactory is deprecated, use make_atari_env_factory(...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class AtariEpochStopCallback(EpochStopCallback):
    def __init__(self, task: str) -> None:
        self.task = task

    def should_stop(self, mean_rewards: float, context: TrainingContext) -> bool:
        env = context.envs.env
        if env.spec and env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        if "Pong" in self.task:
            return mean_rewards >= 20
        return False
