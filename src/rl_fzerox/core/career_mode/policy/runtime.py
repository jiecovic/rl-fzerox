# src/rl_fzerox/core/career_mode/policy/runtime.py
"""Policy handoff models for Career Mode races."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fzerox_emulator import RaceControlState, SpinRequest
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches, ActionMaskSnapshot
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.envs.policy_drive import PolicyDriveFrame, PolicyDriveRuntime
from rl_fzerox.core.manager.training import build_managed_train_app_config
from rl_fzerox.core.runtime_spec.schema import TrainAppConfig

if TYPE_CHECKING:
    from fzerox_emulator import Emulator
    from rl_fzerox.core.manager.models import ManagedRun, ManagedSaveCourseSetup
    from rl_fzerox.core.training.inference import PolicyRunner


@dataclass(frozen=True, slots=True)
class CareerModePolicyControl:
    """Loaded trained policy selected for the current Career Mode race."""

    course_setup: ManagedSaveCourseSetup
    policy_run: ManagedRun
    runner: PolicyRunner

    @property
    def key(self) -> tuple[str, str]:
        return (
            self.course_setup.policy_run_id,
            self.course_setup.policy_artifact,
        )


class CareerPolicyRaceDriver:
    """Policy adapter for one Career Mode race handoff."""

    def __init__(
        self,
        *,
        emulator: Emulator,
        policy_control: CareerModePolicyControl,
    ) -> None:
        self.policy_control = policy_control
        self.train_config = _policy_train_config(policy_control)
        self._runtime = PolicyDriveRuntime(
            emulator=emulator,
            train_config=self.train_config,
        )

    @property
    def key(self) -> tuple[str, str]:
        return self.policy_control.key

    @property
    def last_requested_control_state(self) -> RaceControlState:
        return self._runtime.last_requested_control_state

    @property
    def last_gas_level(self) -> float:
        return self._runtime.last_gas_level

    def begin(
        self,
        *,
        seed: int | None,
        course_id: str | None,
    ) -> tuple[ObservationValue, dict[str, object]]:
        observation, info = self._runtime.begin(
            seed=seed,
            course_id=course_id,
        )
        return observation, info

    def step_policy(
        self,
        action: ActionValue,
        *,
        capture_audio: bool = False,
    ) -> PolicyDriveFrame:
        return self._runtime.step_policy(action, capture_audio=capture_audio)

    def step_manual(
        self,
        control_state: RaceControlState,
        *,
        spin_request: SpinRequest = "none",
        capture_audio: bool = False,
    ) -> PolicyDriveFrame:
        return self._runtime.step_manual(
            control_state,
            spin_request=spin_request,
            capture_audio=capture_audio,
        )

    def action_mask_branches(self) -> ActionMaskBranches:
        return self._runtime.action_mask_branches()

    def action_mask_snapshot(self) -> ActionMaskSnapshot:
        return self._runtime.action_mask_snapshot()

    def sync_curriculum_stage(self, stage_index: int | None) -> None:
        self._runtime.sync_curriculum_stage(stage_index)


def _policy_train_config(policy_control: CareerModePolicyControl) -> TrainAppConfig:
    train_config = build_managed_train_app_config(
        policy_control.policy_run.config,
        run_id=policy_control.policy_run.id,
        run_dir=policy_control.policy_run.run_dir,
    )
    return _career_runtime_train_config(train_config)


def _career_runtime_train_config(train_config: TrainAppConfig) -> TrainAppConfig:
    """Return an evaluation-style runtime config for Career Mode policy handoff."""

    # Career attempts should replay the trained policy without episode-scoped
    # randomization, but p=1.0 is not random: it means the policy was trained
    # with that branch always unavailable. Keep the observation/action layout
    # intact so the checkpoint still loads against its original shape.
    # State-feature dropout groups stay in the config because watch inference
    # uses p=1.0 groups as deterministic zeroing metadata, while p<1.0 groups
    # are not sampled there.
    action_config = train_config.env.action.model_copy(
        update={
            "lean_episode_mask_probability": _deterministic_episode_mask_probability(
                train_config.env.action.lean_episode_mask_probability
            ),
            "air_brake_episode_mask_probability": _deterministic_episode_mask_probability(
                train_config.env.action.air_brake_episode_mask_probability
            ),
            "spin_episode_mask_probability": _deterministic_episode_mask_probability(
                train_config.env.action.spin_episode_mask_probability
            ),
        }
    )
    env_config = train_config.env.model_copy(update={"action": action_config})
    return train_config.model_copy(update={"env": env_config})


def _deterministic_episode_mask_probability(probability: float) -> float:
    """Preserve hard action masks while removing stochastic inference changes."""

    return 1.0 if probability >= 1.0 else 0.0
