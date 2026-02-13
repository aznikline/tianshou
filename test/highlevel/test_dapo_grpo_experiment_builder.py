import pytest

from test.highlevel.env_factory import ContinuousTestEnvFactory, DiscreteTestEnvFactory
from tianshou.highlevel.config import OnPolicyTrainingConfig
from tianshou.highlevel.experiment import (
    DAPOExperimentBuilder,
    ExperimentBuilder,
    ExperimentConfig,
    GRPOExperimentBuilder,
)


@pytest.mark.parametrize(
    ("builder_cls", "env_factory"),
    [
        (DAPOExperimentBuilder, ContinuousTestEnvFactory()),
        (GRPOExperimentBuilder, ContinuousTestEnvFactory()),
        (DAPOExperimentBuilder, DiscreteTestEnvFactory()),
        (GRPOExperimentBuilder, DiscreteTestEnvFactory()),
    ],
)
def test_dapo_grpo_experiment_build_and_run(
    builder_cls: type[ExperimentBuilder],
    env_factory: ContinuousTestEnvFactory | DiscreteTestEnvFactory,
) -> None:
    training_config = OnPolicyTrainingConfig(
        max_epochs=1,
        epoch_num_steps=50,
        num_training_envs=1,
        num_test_envs=1,
        collection_step_num_env_steps=50,
        test_step_num_episodes=1,
        batch_size=32,
    )
    experiment_config = ExperimentConfig(
        persistence_enabled=False,
        watch=False,
    )
    builder = builder_cls(
        experiment_config=experiment_config,
        env_factory=env_factory,
        training_config=training_config,
    )
    experiment = builder.build()
    experiment.run(run_name="test_dapo_grpo")
