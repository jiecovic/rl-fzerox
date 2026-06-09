# src/rl_fzerox/core/career_mode/runner/camera.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rl_fzerox.core.career_mode.runner.menu import (
    CareerPhase,
    MenuFacts,
    MenuInput,
    RawMenuStep,
    in_gp_race,
    raw_step,
)
from rl_fzerox.core.domain.camera import CameraSettingName
from rl_fzerox.core.envs.engine.info import telemetry_info
from rl_fzerox.core.envs.engine.reset.camera import (
    CAMERA_SYNC_CONTROLS,
    CameraSyncBackend,
    sync_camera_setting,
)


class CareerCameraRuntimeSession(Protocol):
    @property
    def emulator(self) -> CameraSyncBackend: ...


@dataclass(slots=True)
class CareerCameraSync:
    """Own the camera-setting target and the real input sync side effects."""

    _target: CameraSettingName | None = None
    _synced: bool = True
    _tap_count: int = 0

    @property
    def target(self) -> CameraSettingName | None:
        return self._target

    @property
    def synced(self) -> bool:
        return self._synced

    @property
    def tap_count(self) -> int:
        return self._tap_count

    def set_target(self, target: CameraSettingName | None) -> None:
        self._target = target
        self.reset()

    def reset(self) -> None:
        self._synced = self._target is None
        self._tap_count = 0

    def ready(self, info: dict[str, object]) -> bool:
        facts = MenuFacts.from_info(info)
        if self._target is None:
            return True
        if facts.camera_setting == self._target:
            self._synced = True
            return True
        return self._synced

    def next_menu_step(self, info: dict[str, object]) -> RawMenuStep | None:
        facts = MenuFacts.from_info(info)
        if self._target is None or self._synced:
            return None
        if facts.camera_setting == self._target:
            self._synced = True
            return None
        if not race_intro_ready_for_camera(info):
            return raw_step(
                MenuInput.NEUTRAL,
                1,
                phase="camera_sync:wait_intro",
            )
        return raw_step(MenuInput.NEUTRAL, 1, phase="camera_sync:apply")

    def sync_before_policy_handoff(
        self,
        *,
        session: CareerCameraRuntimeSession,
        info: dict[str, object],
        phase: CareerPhase,
    ) -> bool:
        if (
            self._target is None
            or self._synced
            or phase not in {CareerPhase.ENTER_RACE, CareerPhase.CONTINUE_AFTER_RACE}
            or not in_gp_race(info)
            or not race_intro_ready_for_camera(info)
        ):
            return False

        telemetry = session.emulator.try_read_telemetry()
        try:
            telemetry = sync_camera_setting(
                session.emulator,
                target_name=self._target,
                telemetry=telemetry,
                info=info,
            )
        except RuntimeError as exc:
            info["camera_setting_sync"] = "retry"
            info["camera_setting_sync_reason"] = str(exc)
            return False
        if telemetry is None:
            return False
        info.update(telemetry_info(telemetry))
        if telemetry.camera_setting_name == self._target:
            self._synced = True
            tap_count = info.get("camera_setting_taps")
            self._tap_count = tap_count if isinstance(tap_count, int) else 0
        return True


def race_intro_ready_for_camera(info: dict[str, object]) -> bool:
    transition_state = info.get("menu_transition_state_raw")
    if isinstance(transition_state, int) and not isinstance(transition_state, bool):
        if transition_state != 0:
            return False
    timer = info.get("race_intro_timer")
    if isinstance(timer, bool) or not isinstance(timer, int):
        return False
    return 0 < timer < CAMERA_SYNC_CONTROLS.ready_intro_timer
