# src/rl_fzerox/core/envs/engine/info.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Protocol

from fzerox_emulator import FZeroXTelemetry, ObservationSpec
from rl_fzerox.core.envs.laps import completed_race_laps
from rl_fzerox.core.envs.observations import (
    ActionHistoryControl,
    ObservationStackMode,
    StateComponentsSettings,
    image_observation_shape,
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
    action_history_len: int | None,
    action_history_controls: tuple[ActionHistoryControl, ...],
    observation_state_components: StateComponentsSettings | None,
    split_lean_history: bool,
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
        if observation_state_components is None:
            raise ValueError("image_state observations require state components in debug info")
        info["observation_action_history_len"] = action_history_len
        info["observation_action_history_controls"] = action_history_controls
        info["observation_state_components"] = tuple(
            asdict(component) for component in observation_state_components
        )
        info["observation_state_shape"] = (
            state_feature_count(
                state_components=observation_state_components,
                split_lean_history=split_lean_history,
            ),
        )
        info["observation_state_features"] = state_feature_names(
            state_components=observation_state_components,
            split_lean_history=split_lean_history,
        )


def telemetry_info(telemetry: FZeroXTelemetry) -> dict[str, object]:
    """Project pickle-safe episode info from the live telemetry snapshot."""

    race_laps_completed = completed_race_laps(telemetry)
    info = _telemetry_race_context_info(telemetry)
    info.update(_telemetry_player_race_info(telemetry))
    info.update(_telemetry_player_state_info(telemetry))
    info.update(_telemetry_lap_progress_info(telemetry, race_laps_completed))
    return info


def _telemetry_race_context_info(telemetry: FZeroXTelemetry) -> dict[str, object]:
    return {
        "game_mode": telemetry.game_mode_name,
        "game_mode_raw": telemetry.game_mode_raw,
        "difficulty": telemetry.difficulty_name,
        "difficulty_name": telemetry.difficulty_name,
        "difficulty_raw": telemetry.difficulty_raw,
        "course_index": telemetry.course_index,
        "camera_setting": telemetry.camera_setting_name,
        "camera_setting_raw": telemetry.camera_setting_raw,
        "race_intro_timer": telemetry.race_intro_timer,
        "menu_selected_mode_raw": telemetry.menu_selected_mode_raw,
        "menu_difficulty_state_raw": telemetry.menu_difficulty_state_raw,
        "menu_difficulty_cursor_raw": telemetry.menu_difficulty_cursor_raw,
        "menu_transition_state_raw": telemetry.menu_transition_state_raw,
        "menu_current_ghost_type_raw": telemetry.menu_current_ghost_type_raw,
        "queued_game_mode_raw": telemetry.queued_game_mode_raw,
    }


def _telemetry_player_race_info(telemetry: FZeroXTelemetry) -> dict[str, object]:
    return {
        "total_lap_count": telemetry.total_lap_count,
        "race_time_ms": telemetry.player.race_time_ms,
        "race_distance": telemetry.player.race_distance,
        "speed_kph": telemetry.player.speed_kph,
        "position": telemetry.player.position,
        "gp_final_rank": telemetry.gp_final_rank,
        "gp_points": telemetry.gp_points,
        "ko_star_count": telemetry.player.ko_star_count,
        "total_racers": telemetry.total_racers,
    }


def _telemetry_player_state_info(telemetry: FZeroXTelemetry) -> dict[str, object]:
    return {
        "termination_reason": telemetry.player.terminal_reason,
        "finished": telemetry.player.finished,
        "retired": telemetry.player.retired,
        "crashed": telemetry.player.crashed,
    }


def _telemetry_lap_progress_info(
    telemetry: FZeroXTelemetry,
    race_laps_completed: int,
) -> dict[str, object]:
    return {
        "lap": telemetry.player.lap,
        "laps_completed": race_laps_completed,
        "race_laps_completed": race_laps_completed,
        "raw_laps_completed": telemetry.player.laps_completed,
        "episode_completion_fraction": _episode_completion_fraction(telemetry),
        "energy": telemetry.player.energy,
    }


def _episode_completion_fraction(telemetry: FZeroXTelemetry) -> float:
    course_length = float(telemetry.course_length)
    total_lap_count = int(telemetry.total_lap_count)
    if course_length <= 0.0 or total_lap_count <= 0:
        return 0.0
    total_race_distance = course_length * total_lap_count
    if total_race_distance <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(telemetry.player.race_distance) / total_race_distance))


def telemetry_boost_unlocked(telemetry: FZeroXTelemetry | None) -> bool:
    """Return the game's manual-boost unlock flag.

    The emulator field is named ``can_boost``, but in game semantics it means
    boost has been unlocked after lap one. It is not the stricter
    ``can_boost`` policy/reward signal.
    """

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
