# src/rl_fzerox/core/career_mode/runner/policy.py
"""Policy handoff models for Career Mode races."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fzerox_emulator import RaceControlState
from fzerox_emulator.arrays import ControllerMaskBatch, DisplayFrames
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine import FZeroXEnvEngine
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches, ActionMaskSnapshot
from rl_fzerox.core.envs.engine.stepping import PolicyDriveStep
from rl_fzerox.core.envs.observations import ObservationValue
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


@dataclass(frozen=True, slots=True)
class CareerPolicyRaceStep:
    """Policy-facing race step data for viewer updates."""

    observation: ObservationValue
    reward: float
    info: dict[str, object]
    display_frames: DisplayFrames
    display_controller_masks: ControllerMaskBatch


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
        self._engine = FZeroXEnvEngine(
            backend=emulator,
            config=self.train_config.env,
            reward_config=self.train_config.reward,
            curriculum_config=self.train_config.curriculum,
        )

    @property
    def key(self) -> tuple[str, str]:
        return self.policy_control.key

    @property
    def last_requested_control_state(self) -> RaceControlState:
        return self._engine.last_requested_control_state

    @property
    def last_gas_level(self) -> float:
        return self._engine.last_gas_level

    def begin(
        self,
        *,
        seed: int | None,
        course_id: str | None,
    ) -> tuple[ObservationValue, dict[str, object]]:
        observation, info = self._engine.begin_policy_drive(
            seed=seed,
            course_id=course_id,
        )
        return observation, _race_start_info(info)

    def step_policy(self, action: ActionValue) -> CareerPolicyRaceStep:
        return self._from_policy_engine_step(self._engine.step_policy_drive(action))

    def step_manual(self, control_state: RaceControlState) -> CareerPolicyRaceStep:
        return self._from_policy_engine_step(self._engine.step_control_policy_drive(control_state))

    def action_mask_branches(self) -> ActionMaskBranches:
        return self._engine.action_mask_branches()

    def action_mask_snapshot(self) -> ActionMaskSnapshot:
        return self._engine.action_mask_snapshot()

    def sync_curriculum_stage(self, stage_index: int | None) -> None:
        self._engine.sync_checkpoint_curriculum_stage(stage_index)

    @staticmethod
    def _from_policy_engine_step(step: PolicyDriveStep) -> CareerPolicyRaceStep:
        return CareerPolicyRaceStep(
            observation=step.observation,
            reward=step.reward,
            info=dict(step.info),
            display_frames=step.display_frames,
            display_controller_masks=step.display_controller_masks,
        )


def _policy_train_config(policy_control: CareerModePolicyControl) -> TrainAppConfig:
    return build_managed_train_app_config(
        policy_control.policy_run.config,
        run_id=policy_control.policy_run.id,
        run_dir=policy_control.policy_run.run_dir,
    )


def _race_start_info(info: dict[str, object]) -> dict[str, object]:
    normalized = dict(info)
    normalized.update(
        {
            "episode_step": 0,
            "step_reward": 0.0,
            "progress_frontier_stalled_frames": 0,
            "stalled_steps": 0,
            "frames_run": 0,
            "repeat_index": 0,
        }
    )
    normalized.pop("reward_breakdown", None)
    return normalized
