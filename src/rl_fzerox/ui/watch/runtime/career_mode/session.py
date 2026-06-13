# src/rl_fzerox/ui/watch/runtime/career_mode/session.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fzerox_emulator import Emulator, RaceControlState, SpinRequest
from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.career_mode.runner.policy import (
    CareerModePolicyControl,
    CareerPolicyRaceDriver,
)
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskBranches, ActionMaskSnapshot
from rl_fzerox.core.envs.engine.info import backend_step_info, telemetry_info
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.envs.policy_drive import PolicyDriveFrame
from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.core.seed import seed_process
from rl_fzerox.ui.watch.runtime.timing import _resolve_control_fps, _target_seconds

if TYPE_CHECKING:
    from rl_fzerox.core.policy.auxiliary_state import AuxiliaryStateTargetName


@dataclass(slots=True)
class CareerModeRuntimeSession:
    """Native emulator session with a lazily attached race policy adapter."""

    config: WatchAppConfig
    emulator: Emulator
    native_fps: float
    native_sample_rate: float
    native_control_fps: float
    target_control_fps: float | None
    target_control_seconds: float | None
    watch_zeroed_state_features: frozenset[str]
    auxiliary_target_names: tuple[AuxiliaryStateTargetName, ...]
    _policy_race: CareerPolicyRaceDriver | None = field(init=False, repr=False)
    _active_policy_key: tuple[str, str] | None = field(init=False, repr=False)
    _active_policy_control: CareerModePolicyControl | None = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._policy_race: CareerPolicyRaceDriver | None = None
        self._active_policy_key: tuple[str, str] | None = None
        self._active_policy_control: CareerModePolicyControl | None = None

    def close(self) -> None:
        self.emulator.close()

    @property
    def backend(self) -> Emulator:
        return self.emulator

    def render(self) -> RgbFrame:
        # The display renderer applies the native crop/aspect policy. The
        # custom dimensions only satisfy the binding's layout argument; display
        # output still uses the crop-correct natural preview size.
        return self.emulator.render_display(height=1, width=1)

    def action_mask_branches(self) -> ActionMaskBranches:
        if self._policy_race is None:
            return {}
        return self._policy_race.action_mask_branches()

    def action_mask_snapshot(self) -> ActionMaskSnapshot:
        return self._require_policy_race().action_mask_snapshot()

    @property
    def last_requested_control_state(self) -> RaceControlState:
        if self._policy_race is None:
            return RaceControlState()
        return self._policy_race.last_requested_control_state

    @property
    def last_gas_level(self) -> float:
        if self._policy_race is None:
            return 0.0
        return self._policy_race.last_gas_level

    def begin_policy_race(
        self,
        *,
        policy_control: CareerModePolicyControl,
        seed: int | None,
        course_id: str | None,
    ) -> tuple[ObservationValue, dict[str, object]]:
        self._ensure_policy_race(policy_control)
        return self._require_policy_race().begin(seed=seed, course_id=course_id)

    def step_policy(
        self,
        action: ActionValue,
        *,
        capture_audio: bool = False,
    ) -> PolicyDriveFrame:
        return self._require_policy_race().step_policy(action, capture_audio=capture_audio)

    def step_manual_race(
        self,
        control_state: RaceControlState,
        *,
        spin_request: SpinRequest = "none",
        capture_audio: bool = False,
    ) -> PolicyDriveFrame:
        return self._require_policy_race().step_manual(
            control_state,
            spin_request=spin_request,
            capture_audio=capture_audio,
        )

    def sync_policy_curriculum_stage(self, stage_index: int | None) -> None:
        self._require_policy_race().sync_curriculum_stage(stage_index)

    def menu_info(self) -> dict[str, object]:
        telemetry = self.emulator.try_read_telemetry()
        info = backend_step_info(self.emulator)
        if telemetry is not None:
            info.update(telemetry_info(telemetry))
        return info

    def snapshot_config(self, base_config: WatchAppConfig) -> WatchAppConfig:
        """Return the viewer config for the currently active policy race."""

        policy_race = self._policy_race
        policy_control = self._active_policy_control
        if policy_race is None or policy_control is None:
            return base_config
        train_config = policy_race.train_config
        return base_config.model_copy(
            update={
                "track": train_config.track,
                "env": train_config.env,
                "reward": train_config.reward,
                "policy": train_config.policy,
                "curriculum": train_config.curriculum,
                "train": train_config.train,
                "watch": base_config.watch.model_copy(
                    update={
                        "policy_run_dir": policy_control.policy_run.run_dir,
                        "policy_artifact": policy_control.course_setup.policy_artifact,
                        "policy_algorithm": train_config.train.algorithm,
                    }
                ),
            }
        )

    def _ensure_policy_race(self, policy_control: CareerModePolicyControl) -> None:
        if self._active_policy_key == policy_control.key and self._policy_race is not None:
            return
        self._policy_race = CareerPolicyRaceDriver(
            emulator=self.emulator,
            policy_control=policy_control,
        )
        self._active_policy_key = policy_control.key
        self._active_policy_control = policy_control
        train_config = self._policy_race.train_config
        self.watch_zeroed_state_features = frozenset()
        self.auxiliary_target_names = tuple(
            loss.name for loss in train_config.policy.auxiliary_state.losses
        )

    def _require_policy_race(self) -> CareerPolicyRaceDriver:
        if self._policy_race is None:
            raise RuntimeError("Career Mode policy race is not active")
        return self._policy_race


def open_career_mode_runtime_session(config: WatchAppConfig) -> CareerModeRuntimeSession:
    """Open the Career Mode emulator without constructing a policy environment."""

    if config.watch.attempt_seed is not None:
        seed_process(config.watch.attempt_seed)
    emulator = Emulator(
        core_path=config.emulator.core_path,
        rom_path=config.emulator.rom_path,
        runtime_dir=config.emulator.runtime_dir,
        baseline_state_path=None,
        renderer=config.emulator.renderer,
    )
    native_fps = emulator.native_fps
    native_sample_rate = emulator.native_sample_rate
    native_control_fps = native_fps
    target_control_fps = _resolve_control_fps(
        config.watch.control_fps,
        native_control_fps=native_control_fps,
    )
    return CareerModeRuntimeSession(
        config=config,
        emulator=emulator,
        native_fps=native_fps,
        native_sample_rate=native_sample_rate,
        native_control_fps=native_control_fps,
        target_control_fps=target_control_fps,
        target_control_seconds=_target_seconds(target_control_fps),
        watch_zeroed_state_features=frozenset(),
        auxiliary_target_names=(),
    )
