# tests/support/native_objects.py
from __future__ import annotations

from collections.abc import Iterable

from fzerox_emulator import (
    FZeroXTelemetry,
    PlayerTelemetry,
    StepStatus,
    StepSummary,
)
from fzerox_emulator import (
    encode_state_flags as _encode_state_flags,
)


def encode_state_flags(state_labels: Iterable[str]) -> int:
    return int(_encode_state_flags(list(state_labels)))


def make_player_telemetry(
    *,
    state_labels: tuple[str, ...] = ("active",),
    state_flags: int | None = None,
    speed_kph: float = 100.0,
    energy: float = 178.0,
    max_energy: float = 178.0,
    boost_timer: int = 0,
    reverse_timer: int = 0,
    race_distance: float = 0.0,
    lap_distance: float | None = None,
    race_time_ms: int = 0,
    lap: int = 1,
    laps_completed: int = 0,
    position: int = 30,
) -> PlayerTelemetry:
    resolved_state_flags = encode_state_flags(state_labels) if state_flags is None else state_flags
    return PlayerTelemetry(
        state_flags=resolved_state_flags,
        speed_kph=speed_kph,
        energy=energy,
        max_energy=max_energy,
        boost_timer=boost_timer,
        reverse_timer=reverse_timer,
        race_distance=race_distance,
        lap_distance=race_distance if lap_distance is None else lap_distance,
        race_time_ms=race_time_ms,
        lap=lap,
        laps_completed=laps_completed,
        position=position,
    )


def make_telemetry(
    *,
    total_lap_count: int = 3,
    game_mode_raw: int = 1,
    game_mode_name: str = "gp_race",
    in_race_mode: bool = True,
    total_racers: int = 30,
    course_index: int = 0,
    difficulty_raw: int = 0,
    difficulty_name: str = "novice",
    camera_setting_raw: int = 2,
    camera_setting_name: str = "regular",
    player: PlayerTelemetry | None = None,
    state_labels: tuple[str, ...] = ("active",),
    state_flags: int | None = None,
    speed_kph: float = 100.0,
    energy: float = 178.0,
    max_energy: float = 178.0,
    boost_timer: int = 0,
    reverse_timer: int = 0,
    race_distance: float = 0.0,
    lap_distance: float | None = None,
    race_time_ms: int = 0,
    lap: int = 1,
    laps_completed: int = 0,
    position: int = 30,
) -> FZeroXTelemetry:
    resolved_player = player or make_player_telemetry(
        state_labels=state_labels,
        state_flags=state_flags,
        speed_kph=speed_kph,
        energy=energy,
        max_energy=max_energy,
        boost_timer=boost_timer,
        reverse_timer=reverse_timer,
        race_distance=race_distance,
        lap_distance=lap_distance,
        race_time_ms=race_time_ms,
        lap=lap,
        laps_completed=laps_completed,
        position=position,
    )
    return FZeroXTelemetry(
        total_lap_count=total_lap_count,
        game_mode_raw=game_mode_raw,
        game_mode_name=game_mode_name,
        in_race_mode=in_race_mode,
        total_racers=total_racers,
        course_index=course_index,
        player=resolved_player,
        difficulty_raw=difficulty_raw,
        difficulty_name=difficulty_name,
        camera_setting_raw=camera_setting_raw,
        camera_setting_name=camera_setting_name,
    )


def make_step_summary(
    *,
    frames_run: int = 1,
    max_race_distance: float,
    reverse_active_frames: int = 0,
    low_speed_frames: int = 0,
    energy_loss_total: float = 0.0,
    energy_gain_total: float = 0.0,
    consecutive_low_speed_frames: int = 0,
    entered_state_labels: tuple[str, ...] = (),
    entered_state_flags: int | None = None,
    final_frame_index: int = 1,
) -> StepSummary:
    resolved_entered_state_flags = (
        encode_state_flags(entered_state_labels)
        if entered_state_flags is None
        else entered_state_flags
    )
    return StepSummary(
        frames_run=frames_run,
        max_race_distance=max_race_distance,
        reverse_active_frames=reverse_active_frames,
        low_speed_frames=low_speed_frames,
        energy_loss_total=energy_loss_total,
        energy_gain_total=energy_gain_total,
        consecutive_low_speed_frames=consecutive_low_speed_frames,
        entered_state_flags=resolved_entered_state_flags,
        final_frame_index=final_frame_index,
    )


def make_step_status(
    *,
    step_count: int,
    stalled_steps: int = 0,
    reverse_timer: int = 0,
    progress_frontier_stalled_frames: int = 0,
    termination_reason: str | None = None,
    truncation_reason: str | None = None,
) -> StepStatus:
    return StepStatus(
        step_count=step_count,
        stalled_steps=stalled_steps,
        reverse_timer=reverse_timer,
        progress_frontier_stalled_frames=progress_frontier_stalled_frames,
        termination_reason=termination_reason,
        truncation_reason=truncation_reason,
    )
