#!/usr/bin/env python3

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from typing import Literal

from sensai.util import logging

from tianshou.env.atari.atari_network import (
    ActorFactoryAtariDQN,
)
from tianshou.env.atari.atari_wrapper import AtariEpochStopCallback, make_atari_env_factory
from tianshou.highlevel.config import OnPolicyTrainingConfig
from tianshou.highlevel.experiment import (
    A2CExperimentBuilder,
    ExperimentConfig,
)
from tianshou.highlevel.params.algorithm_params import A2CParams
from tianshou.highlevel.params.lr_scheduler import LRSchedulerFactoryFactoryLinear


def main(
    task: str = "ALE/Pong-v5",
    persistence_base_dir: str = "log",
    num_experiments: int = 1,
    experiment_launcher: Literal["sequential", "joblib"] = "sequential",
    max_epochs: int = 100,
    epoch_num_steps: int = 50000,
    require_envpool: bool = True,
) -> None:
    """
    Train an Atari agent with A2C using a throughput-oriented setup.

    This entrypoint is intended for faster wall-clock iteration than the default
    Atari scripts. It can require EnvPool to ensure vectorized simulation speed.
    """
    persistence_base_dir = os.path.abspath(os.path.join(persistence_base_dir, task))
    experiment_config = ExperimentConfig(persistence_base_dir=persistence_base_dir, watch=False)

    training_config = OnPolicyTrainingConfig(
        max_epochs=max_epochs,
        epoch_num_steps=epoch_num_steps,
        batch_size=None,
        num_training_envs=64,
        num_test_envs=10,
        buffer_size=8192,
        collection_step_num_env_steps=256,
        update_step_num_repetitions=2,
        replay_buffer_stack_num=4,
        replay_buffer_ignore_obs_next=True,
        replay_buffer_save_only_last_obs=True,
    )

    env_factory = make_atari_env_factory(
        task,
        frame_stack=4,
        scale=True,
        use_envpool_if_available=True,
        require_envpool=require_envpool,
    )

    experiment_builder = (
        A2CExperimentBuilder(env_factory, experiment_config, training_config)
        .with_a2c_params(
            A2CParams(
                gamma=0.99,
                gae_lambda=0.95,
                return_scaling=False,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                lr=7e-4,
                lr_scheduler=LRSchedulerFactoryFactoryLinear(training_config),
            ),
        )
        .with_actor_factory(ActorFactoryAtariDQN(scale_obs=True, features_only=True))
        .with_critic_factory_use_actor()
        .with_epoch_stop_callback(AtariEpochStopCallback(task))
    )

    experiment_builder.build_and_run(num_experiments=num_experiments, launcher=experiment_launcher)


if __name__ == "__main__":
    result = logging.run_cli(main, level=logging.INFO)
