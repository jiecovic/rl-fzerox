# src/rl_fzerox/core/envs/engine/info.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Protocol

from fzerox_emulator import FZeroXTelemetry, ObservationSpec
from rl_fzerox.core.envs.laps import completed_race_laps
from rl_fzerox.core.envs.observation_image import image_observation_shape
from rl_fzerox.core.envs.observations import (
    ActionHistoryControl,
    ObservationCourseContext,
    ObservationGroundEffectContext,
    ObservationStackMode,
    ObservationStateProfile,
    StateComponentsSettings,
    state_feature_count,
    state_feature_names,
)


class BackendInfoReader(Protocol):
    """Narrow backend surface needed for frame/debug info."""

    @property
    def name(self) -> str: ...

    @property
    def native_fps(self) -> float: ...

    @property
    def display_aspect_ratio(self) -> float: ...

    @property
    def frame_index(self) -> int: ...

    def vehicle_setup_info(self) -> dict[str, object]: ...


class TelemetryBackend(Protocol):
    """Backend surface needed to read live telemetry."""

    def try_read_telemetry(self) -> FZeroXTelemetry | None: ...


def has_custom_baseline(info: Mapping[str, object]) -> bool:
    """Return whether reset info points at a custom user baseline."""

    return info.get("baseline_kind") == "custom"


def read_live_telemetry(backend: TelemetryBackend) -> FZeroXTelemetry | None:
    """Read the latest live telemetry snapshot, if one is available."""

    return backend.try_read_telemetry()


def set_observation_info(
    info: dict[str, object],
    *,
    observation_shape: tuple[int, ...],
    observation_spec: ObservationSpec,
    frame_stack: int,
    observation_stack_mode: ObservationStackMode,
    observation_minimap_layer: bool,
    observation_resize_filter: str,
    observation_minimap_resize_filter: str,
    observation_mode: str,
    observation_state_profile: ObservationStateProfile,
    observation_course_context: ObservationCourseContext,
    observation_ground_effect_context: ObservationGroundEffectContext,
    action_history_len: int | None,
    action_history_controls: tuple[ActionHistoryControl, ...],
    observation_state_components: StateComponentsSettings | None,
    observation_zeroed_state_components: tuple[str, ...] = (),
) -> None:
    """Attach observation metadata used by watch/debug surfaces."""

    expected_image_shape = image_observation_shape(
        observation_spec,
        frame_stack=frame_stack,
        stack_mode=observation_stack_mode,
        minimap_layer=observation_minimap_layer,
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
    info["observation_minimap_layer"] = observation_minimap_layer
    info["observation_resize_filter"] = observation_resize_filter
    info["observation_minimap_resize_filter"] = observation_minimap_resize_filter
    if observation_mode == "image_state":
        info["observation_state_profile"] = observation_state_profile
        info["observation_course_context"] = observation_course_context
        info["observation_ground_effect_context"] = observation_ground_effect_context
        info["observation_action_history_len"] = action_history_len
        info["observation_action_history_controls"] = action_history_controls
        if observation_state_components is not None:
            info["observation_state_components"] = tuple(
                asdict(component) for component in observation_state_components
            )
        info["observation_zeroed_state_components"] = observation_zeroed_state_components
        info["observation_state_shape"] = (
            state_feature_count(
                observation_state_profile,
                course_context=observation_course_context,
                ground_effect_context=observation_ground_effect_context,
                action_history_len=action_history_len,
                action_history_controls=action_history_controls,
                state_components=observation_state_components,
            ),
        )
        info["observation_state_features"] = state_feature_names(
            observation_state_profile,
            course_context=observation_course_context,
            ground_effect_context=observation_ground_effect_context,
            action_history_len=action_history_len,
            action_history_controls=action_history_controls,
            state_components=observation_state_components,
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


def backend_step_info(backend: BackendInfoReader) -> dict[str, object]:
    """Return backend-side timing/display metadata for the current frame."""

    info: dict[str, object] = {
        "backend": backend.name,
        "frame_index": backend.frame_index,
        "display_aspect_ratio": backend.display_aspect_ratio,
        "native_fps": backend.native_fps,
    }
    info.update(vehicle_setup_info(backend))
    return info


def vehicle_setup_info(backend: BackendInfoReader) -> dict[str, object]:
    """Read native-decoded player machine setup info for HUD/debug checks."""

    try:
        return dict(backend.vehicle_setup_info())
    except (OSError, RuntimeError, ValueError):
        return {}


def set_curriculum_info(
    info: dict[str, object],
    *,
    stage_index: int | None,
    stage_name: str | None,
) -> None:
    """Attach the active curriculum stage in a watch- and callback-friendly form."""

    info["curriculum_stage"] = stage_index
    info["curriculum_stage_name"] = stage_name
