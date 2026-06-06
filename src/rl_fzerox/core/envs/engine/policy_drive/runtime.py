# src/rl_fzerox/core/envs/engine/policy_drive/runtime.py
"""Policy-drive runtime around shared emulator stepping components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fzerox_emulator import RaceControlState
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches, ActionMaskSnapshot
from rl_fzerox.core.envs.engine.policy_drive.frame import PolicyDriveFrame
from rl_fzerox.core.envs.engine.runtime import FZeroXEnvEngine
from rl_fzerox.core.envs.engine.stepping import PolicyDriveStep
from rl_fzerox.core.envs.observations import ObservationValue

if TYPE_CHECKING:
    from fzerox_emulator import Emulator
    from rl_fzerox.core.runtime_spec.schema import TrainAppConfig


class PolicyDriveRuntime:
    """Policy-race adapter with no Gym episode lifecycle contract."""

    def __init__(
        self,
        *,
        emulator: Emulator,
        train_config: TrainAppConfig,
    ) -> None:
        self.train_config = train_config
        self._engine = FZeroXEnvEngine(
            backend=emulator,
            config=train_config.env,
            reward_config=train_config.reward,
            curriculum_config=train_config.curriculum,
        )

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
        return self._engine.begin_policy_drive(seed=seed, course_id=course_id)

    def step_policy(self, action: ActionValue) -> PolicyDriveFrame:
        return _policy_drive_frame(self._engine.step_policy_drive(action))

    def step_manual(self, control_state: RaceControlState) -> PolicyDriveFrame:
        return _policy_drive_frame(self._engine.step_control_policy_drive(control_state))

    def action_mask_branches(self) -> ActionMaskBranches:
        return self._engine.action_mask_branches()

    def action_mask_snapshot(self) -> ActionMaskSnapshot:
        return self._engine.action_mask_snapshot()

    def sync_curriculum_stage(self, stage_index: int | None) -> None:
        self._engine.sync_checkpoint_curriculum_stage(stage_index)


def _policy_drive_frame(step: PolicyDriveStep) -> PolicyDriveFrame:
    return PolicyDriveFrame(
        observation=step.observation,
        reward=step.reward,
        info=dict(step.info),
        display_frames=step.display_frames,
        display_controller_masks=step.display_controller_masks,
    )
