# src/rl_fzerox/core/envs/engine/camera.py
from __future__ import annotations

from fzerox_emulator import ControllerState, EmulatorBackend, FZeroXTelemetry
from rl_fzerox.core.domain.camera import (
    CAMERA_SETTING_BY_NAME,
    CAMERA_SETTINGS,
    CameraSettingName,
)

_CAMERA_SETTING_COUNT = len(CAMERA_SETTINGS)
_CAMERA_RIGHT_TAP = ControllerState(right_stick_x=1.0)
_IDLE_INPUT = ControllerState()


def sync_camera_setting(
    backend: EmulatorBackend,
    *,
    target_name: CameraSettingName | None,
    telemetry: FZeroXTelemetry | None,
    info: dict[str, object],
) -> FZeroXTelemetry | None:
    """Advance the in-game camera setting with C-Right taps until telemetry matches."""

    if target_name is None:
        return telemetry

    info["camera_setting_target"] = target_name
    target = CAMERA_SETTING_BY_NAME[target_name]
    if telemetry is None:
        info["camera_setting_sync"] = "skipped"
        info["camera_setting_sync_reason"] = "telemetry_unavailable"
        return telemetry

    for tap_count in range(_CAMERA_SETTING_COUNT + 1):
        if telemetry.camera_setting_raw == target.raw:
            info["camera_setting_sync"] = "already_set" if tap_count == 0 else "changed"
            info["camera_setting_taps"] = tap_count
            return telemetry
        if tap_count == _CAMERA_SETTING_COUNT:
            break
        telemetry = _tap_next_camera_setting(backend) or telemetry

    raise RuntimeError(
        "Failed to apply configured camera setting "
        f"{target_name!r}; telemetry still reports {telemetry.camera_setting_name!r} "
        f"({telemetry.camera_setting_raw}) after {_CAMERA_SETTING_COUNT} C-Right taps."
    )


def _tap_next_camera_setting(backend: EmulatorBackend) -> FZeroXTelemetry | None:
    # F-Zero X handles camera changes through BTN_CRIGHT. Tapping the real input
    # lets the game update the associated FOV/distance/pitch state coherently.
    backend.set_controller_state(_CAMERA_RIGHT_TAP)
    backend.step_frames(1, capture_video=False)
    backend.set_controller_state(_IDLE_INPUT)
    backend.step_frames(1, capture_video=True)
    return backend.try_read_telemetry()
