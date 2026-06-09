# tests/core/game/reward_support.py
from __future__ import annotations

from fzerox_emulator import FZeroXTelemetry, StepStatus, StepSummary
from tests.support.native_objects import encode_state_flags, make_step_summary, make_telemetry

_COURSE_EFFECT_PIT = 1
_COURSE_EFFECT_DIRT = 2
_COURSE_EFFECT_DASH = 3
_COURSE_EFFECT_ICE = 4


def _telemetry(
    *,
    race_distance: float = 0.0,
    game_mode_name: str = "gp_race",
    state_labels: tuple[str, ...] = ("active",),
    total_racers: int = 30,
    position: int = 30,
    energy: float = 178.0,
    ko_star_count: int = 0,
    boost_timer: int = 0,
    race_time_ms: int = 0,
    speed_kph: float = 100.0,
    laps_completed: int = 0,
    lap: int | None = None,
    reverse_timer: int = 0,
    on_energy_refill: bool = False,
    course_effect_raw: int = 0,
    height_above_ground: float = 0.0,
    signed_lateral_offset: float = 0.0,
    lateral_distance: float = 0.0,
    current_radius_left: float = 0.0,
    current_radius_right: float = 0.0,
    future_local_nearest_segment_index: int | None = None,
    future_local_nearest_segment_distance: float = 0.0,
) -> FZeroXTelemetry:
    state_flags = encode_state_flags(state_labels)
    state_flags |= course_effect_raw
    if on_energy_refill:
        state_flags |= _COURSE_EFFECT_PIT
    return make_telemetry(
        game_mode_name=game_mode_name,
        total_racers=total_racers,
        race_distance=race_distance,
        state_labels=state_labels,
        state_flags=state_flags,
        speed_kph=speed_kph,
        energy=energy,
        ko_star_count=ko_star_count,
        boost_timer=boost_timer,
        race_time_ms=race_time_ms,
        position=position,
        laps_completed=laps_completed,
        lap=max(laps_completed + 1, 1) if lap is None else lap,
        reverse_timer=reverse_timer,
        height_above_ground=height_above_ground,
        signed_lateral_offset=signed_lateral_offset,
        lateral_distance=lateral_distance,
        current_radius_left=current_radius_left,
        current_radius_right=current_radius_right,
        future_local_nearest_segment_index=future_local_nearest_segment_index,
        future_local_nearest_segment_distance=future_local_nearest_segment_distance,
    )


def _summary(
    *,
    max_race_distance: float,
    max_race_distance_speed_kph: float = 760.0,
    frames_run: int = 1,
    airborne_frames: int = 0,
    reverse_active_frames: int = 0,
    low_speed_frames: int = 0,
    energy_loss_total: float = 0.0,
    energy_gain_total: float = 0.0,
    damage_taken_frames: int = 0,
    entered_state_labels: tuple[str, ...] = (),
    entered_course_effects: int = 0,
) -> StepSummary:
    return make_step_summary(
        frames_run=frames_run,
        max_race_distance=max_race_distance,
        max_race_distance_speed_kph=max_race_distance_speed_kph,
        reverse_active_frames=reverse_active_frames,
        low_speed_frames=low_speed_frames,
        energy_loss_total=energy_loss_total,
        energy_gain_total=energy_gain_total,
        damage_taken_frames=damage_taken_frames,
        entered_state_labels=entered_state_labels,
        entered_course_effects=entered_course_effects,
        final_frame_index=frames_run,
        airborne_frames=airborne_frames,
    )


def _status(
    *,
    step_count: int,
    stalled_steps: int = 0,
    reverse_timer: int = 0,
    termination_reason: str | None = None,
    truncation_reason: str | None = None,
) -> StepStatus:
    return StepStatus(
        {
            "step_count": step_count,
            "stalled_steps": stalled_steps,
            "reverse_timer": reverse_timer,
            "termination_reason": termination_reason,
            "truncation_reason": truncation_reason,
        }
    )


def _entered_course_effects(*effects: int) -> int:
    bitset = 0
    for effect in effects:
        bitset |= 1 << effect
    return bitset
