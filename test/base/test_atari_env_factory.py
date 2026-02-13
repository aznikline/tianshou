import pytest

import tianshou.env.atari.atari_wrapper as atari_wrapper


def test_legacy_task_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(atari_wrapper, "_ensure_ale_namespace_registered", lambda: None)
    factory = atari_wrapper.make_atari_env_factory(
        task="PongNoFrameskip-v4",
        frame_stack=4,
        use_envpool_if_available=False,
    )
    assert factory.task == "ALE/Pong-v5"


def test_require_envpool_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(atari_wrapper, "_ensure_ale_namespace_registered", lambda: None)
    monkeypatch.setattr(atari_wrapper, "envpool_is_available", False)
    with pytest.raises(RuntimeError, match="EnvPool is required"):
        atari_wrapper.make_atari_env_factory(
            task="ALE/Pong-v5",
            frame_stack=4,
            require_envpool=True,
        )


def test_envpool_factory_is_used_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(atari_wrapper, "_ensure_ale_namespace_registered", lambda: None)
    monkeypatch.setattr(atari_wrapper, "envpool_is_available", True)
    factory = atari_wrapper.make_atari_env_factory(
        task="ALE/Pong-v5",
        frame_stack=4,
        use_envpool_if_available=True,
    )
    assert factory.envpool_factory is not None
    assert isinstance(factory.envpool_factory, atari_wrapper.AtariEnvPoolFactory)
