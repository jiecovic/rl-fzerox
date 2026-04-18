# src/rl_fzerox/core/envs/engine/info.py
from __future__ import annotations

from collections.abc import Mapping

from fzerox_emulator import EmulatorBackend, FZeroXTelemetry, ObservationSpec
from rl_fzerox.core.envs.laps import completed_race_laps
from rl_fzerox.core.envs.observations import (
    ActionHistoryControl,
    ObservationStackMode,
    ObservationStateProfile,
    image_observation_shape,
    state_feature_count,
    state_feature_names,
)


def has_custom_baseline(info: Mapping[str, object]) -> bool:
    """Return whether reset info points at a custom user baseline."""

    return info.get("baseline_kind") == "custom"


def read_live_telemetry(backend: EmulatorBackend) -> FZeroXTelemetry | None:
    """Read the latest live telemetry snapshot, if one is available."""

    return backend.try_read_telemetry()


def set_observation_info(
    info: dict[str, object],
    *,
    observation_shape: tuple[int, ...],
    observation_spec: ObservationSpec,
    frame_stack: int,
    observation_stack_mode: ObservationStackMode,
    observation_mode: str,
    observation_state_profile: ObservationStateProfile,
    action_history_len: int | None,
    action_history_controls: tuple[ActionHistoryControl, ...],
) -> None:
    """Attach observation metadata used by watch/debug surfaces."""

    expected_image_shape = image_observation_shape(
        observation_spec,
        frame_stack=frame_stack,
        stack_mode=observation_stack_mode,
    )
    if observation_shape != expected_image_shape:
        raise ValueError(
            "Rendered observation shape did not match native observation spec: "
            f"got={observation_shape}, expected={expected_image_shape}"
        )

    info["observation_mode"] = observation_mode
    info["observation_shape"] = observation_shape
    info["observation_frame_shape"] = (
        observation_spec.height,
        observation_spec.width,
        observation_spec.channels,
    )
    info["observation_stack"] = frame_stack
    info["observation_stack_mode"] = observation_stack_mode
    if observation_mode == "image_state":
        info["observation_state_profile"] = observation_state_profile
        info["observation_action_history_len"] = action_history_len
        info["observation_action_history_controls"] = action_history_controls
        info["observation_state_shape"] = (
            state_feature_count(
                observation_state_profile,
                action_history_len=action_history_len,
                action_history_controls=action_history_controls,
            ),
        )
        info["observation_state_features"] = state_feature_names(
            observation_state_profile,
            action_history_len=action_history_len,
            action_history_controls=action_history_controls,
        )


def telemetry_info(telemetry: FZeroXTelemetry) -> dict[str, object]:
    """Project pickle-safe episode info from the live telemetry snapshot."""

    race_laps_completed = completed_race_laps(telemetry)
    return {
        "game_mode": telemetry.game_mode_name,
        "course_index": telemetry.course_index,
        "camera_setting": telemetry.camera_setting_name,
        "camera_setting_raw": telemetry.camera_setting_raw,
        "race_intro_timer": telemetry.race_intro_timer,
        "race_time_ms": telemetry.player.race_time_ms,
        "race_distance": telemetry.player.race_distance,
        "speed_kph": telemetry.player.speed_kph,
        "position": telemetry.player.position,
        "total_racers": telemetry.total_racers,
        "lap": telemetry.player.lap,
        "laps_completed": race_laps_completed,
        "race_laps_completed": race_laps_completed,
        "raw_laps_completed": telemetry.player.laps_completed,
        "energy": telemetry.player.energy,
    }


def telemetry_can_boost(telemetry: FZeroXTelemetry | None) -> bool:
    """Return whether the game currently allows manual boost."""

    if telemetry is None:
        return False
    return bool(telemetry.player.can_boost)


def telemetry_energy_fraction(telemetry: FZeroXTelemetry | None) -> float | None:
    """Return player energy normalized to [0, 1], if telemetry is usable."""

    if telemetry is None:
        return None
    max_energy = float(telemetry.player.max_energy)
    if max_energy <= 0.0:
        return None
    return max(0.0, min(1.0, float(telemetry.player.energy) / max_energy))


def backend_step_info(backend: EmulatorBackend) -> dict[str, object]:
    """Return backend-side timing/display metadata for the current frame."""

    return {
        "backend": backend.name,
        "frame_index": backend.frame_index,
        "display_aspect_ratio": backend.display_aspect_ratio,
        "native_fps": backend.native_fps,
    }


def set_curriculum_info(
    info: dict[str, object],
    *,
    stage_index: int | None,
    stage_name: str | None,
) -> None:
    """Attach the active curriculum stage in a watch- and callback-friendly form."""

    info["curriculum_stage"] = stage_index
    info["curriculum_stage_name"] = stage_name
