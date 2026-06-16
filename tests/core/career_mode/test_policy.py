# tests/core/career_mode/test_policy.py
from __future__ import annotations

from pathlib import Path

from fzerox_emulator.arrays import BoolArray, PolicyState
from rl_fzerox.core.career_mode.runner.policy import (
    CareerModePolicyControl,
    _policy_train_config,
)
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.manager import default_managed_run_config
from rl_fzerox.core.manager.models import ManagedRun, ManagedSaveCourseSetup
from rl_fzerox.core.training.inference import LoadedPolicy, PolicyRunner


def test_career_policy_runtime_disables_random_action_dropouts_only(tmp_path: Path) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.action.lean_episode_mask_probability = 1.0
    config.action.air_brake_episode_mask_probability = 1.0
    config.action.spin_episode_mask_probability = 1.0
    config.observation.state_feature_dropouts = tuple(
        feature.model_copy(update={"dropout_prob": 1.0})
        for feature in config.observation.state_feature_dropouts
    )
    run = ManagedRun(
        id="policy-run",
        name="Policy Run",
        status="finished",
        config=config,
        config_hash="hash",
        run_dir=tmp_path / "policy-run",
        created_at="2026-01-01T00:00:00Z",
        lineage_id="policy-run",
    )
    policy_path = run.run_dir / "checkpoints" / "latest.zip"
    policy_path.parent.mkdir(parents=True)
    policy_path.touch()
    runner = PolicyRunner(
        LoadedPolicy(run_dir=run.run_dir, policy_path=policy_path, artifact="latest"),
        policy=_PolicyStub(),
    )
    control = CareerModePolicyControl(
        course_setup=ManagedSaveCourseSetup(
            id="course-setup",
            save_game_id="save",
            policy_run_id=run.id,
            policy_artifact="latest",
            engine_setting_raw_value=50,
            difficulty="novice",
            cup_id="jack",
            course_id="mute_city",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        ),
        policy_run=run,
        runner=runner,
    )

    train_config = _policy_train_config(control)

    assert train_config.env.action.lean_episode_mask_probability == 0.0
    assert train_config.env.action.air_brake_episode_mask_probability == 0.0
    assert train_config.env.action.spin_episode_mask_probability == 0.0
    assert train_config.train.state_feature_dropout_groups
    assert all(
        group.dropout_prob == 1.0 for group in train_config.train.state_feature_dropout_groups
    )
    assert config.action.lean_episode_mask_probability == 1.0
    assert config.action.air_brake_episode_mask_probability == 1.0
    assert config.action.spin_episode_mask_probability == 1.0
    assert all(feature.dropout_prob == 1.0 for feature in config.observation.state_feature_dropouts)


class _PolicyStub:
    def predict(
        self,
        observation: ObservationValue,
        state: PolicyState = None,
        episode_start: BoolArray | None = None,
        deterministic: bool = True,
    ) -> tuple[ActionValue, PolicyState]:
        return 0, state
