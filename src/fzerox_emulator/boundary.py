# src/fzerox_emulator/boundary.py
"""Typed dictionary payloads exchanged with the compiled native extension."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from fzerox_emulator.control.spin import SpinRequest

if TYPE_CHECKING:
    # Keep the runtime module import-free from `_native`; these types document
    # payload shape without forcing extension loading during lightweight imports.
    from fzerox_emulator._native import PlayerTelemetry


class ObservationSpecDict(TypedDict):
    """Native-resolved observation geometry returned as a plain mapping."""

    preset: str
    width: int
    height: int
    channels: int
    display_width: int
    display_height: int


class FrameObservationOptionsDict(TypedDict, total=False):
    """Optional image-rendering arguments accepted by the native frame renderer."""

    stack_mode: Literal["rgb", "gray", "luma_chroma"]
    minimap_layer: bool
    resize_filter: Literal["nearest", "bilinear"]
    minimap_resize_filter: Literal["nearest", "bilinear"]
    height: int
    width: int


class ObservationImageRequestDict(TypedDict, total=False):
    """One observation render request passed into repeated-step native calls."""

    preset: str
    frame_stack: int
    stack_mode: Literal["rgb", "gray", "luma_chroma"]
    minimap_layer: bool
    resize_filter: Literal["nearest", "bilinear"]
    minimap_resize_filter: Literal["nearest", "bilinear"]
    height: int
    width: int


class VehicleSetupInfoDict(TypedDict):
    """Native-decoded machine setup values read back from live RAM."""

    player_character_index_ram: int
    racer_character_index_ram: int
    engine_setting_ram: float
    engine_setting_raw_value_ram: int
    engine_setting_percent_ram: float
    character_engine_setting_ram: float | None
    racer_engine_curve_ram: float | None


class RaceStartRequestDict(TypedDict, total=False):
    """Race-start setup request with raw values already normalized by Python."""

    mode: str
    course_index: int
    character_index: int
    machine_skin_index: int
    engine_setting_raw_value: int
    total_lap_count: int
    gp_difficulty_raw_value: int


class RepeatStepRequestDict(TypedDict, total=False):
    """Gameplay control and stopping-rule request for one repeated native step.

    The control fields are F-Zero X semantics. Rust maps them to the configured
    Mupen64Plus-Next/libretro profile:

    - ``gas`` -> N64 A
    - ``air_brake`` -> N64 C-Down
    - ``boost`` -> N64 B
    - ``lean_left`` -> N64 L
    - ``lean_right`` -> N64 R
    """

    action_repeat: int
    stuck_min_speed_kph: float
    energy_loss_epsilon: float
    max_episode_steps: int
    progress_frontier_stall_limit_frames: int | None
    progress_frontier_epsilon: float
    terminate_on_energy_depleted: bool
    lean_timer_assist: bool
    spin_request: SpinRequest
    spin_cooldown_frames: int
    gas: bool
    air_brake: bool
    boost: bool
    lean_left: bool
    lean_right: bool
    stick_x: float
    pitch: float


class RepeatObservationStepRequestDict(TypedDict):
    """Repeated-step request with a single final observation render."""

    step: RepeatStepRequestDict
    observation: ObservationImageRequestDict
    capture_audio: NotRequired[bool]


class RepeatMultiObservationStepRequestDict(TypedDict):
    """Repeated-step request with several final observation renders."""

    step: RepeatStepRequestDict
    observations: list[ObservationImageRequestDict]


class PlayerTelemetryDict(TypedDict, total=False):
    """Native player telemetry payload used to construct `PlayerTelemetry`."""

    state_flags: int
    speed_kph: float
    energy: float
    max_energy: float
    boost_timer: int
    recoil_tilt_magnitude: float
    reverse_timer: int
    race_distance: float
    lap_distance: float
    race_time_ms: int
    lap: int
    laps_completed: int
    position: int
    ko_star_count: int
    damage_rumble_counter: int
    segment_index: int | None
    segment_t: float
    segment_length_proportion: float
    world_pos_x: float
    world_pos_y: float
    world_pos_z: float
    segment_center_x: float
    segment_center_y: float
    segment_center_z: float
    local_lateral_velocity: float
    signed_lateral_offset: float
    lateral_distance: float
    lateral_displacement_magnitude: float
    current_radius_left: float
    current_radius_right: float
    height_above_ground: float
    future_local_nearest_segment_index: int | None
    future_local_nearest_segment_distance: float
    velocity_magnitude: float
    acceleration_magnitude: float
    acceleration_force: float
    drift_attack_force: float
    collision_mass: float
    machine_character_index: int
    machine_body_stat: int
    machine_boost_stat: int
    machine_grip_stat: int
    machine_weight: int
    engine_setting: float


class FZeroXTelemetryDict(TypedDict, total=False):
    """Native race telemetry payload used to construct `FZeroXTelemetry`."""

    total_lap_count: int
    game_mode_raw: int
    game_mode_name: str
    in_race_mode: bool
    total_racers: int
    gp_final_rank: int
    course_index: int
    player: PlayerTelemetry
    course_length: float
    course_segment_count: int
    difficulty_raw: int
    difficulty_name: str | None
    camera_setting_raw: int
    camera_setting_name: str | None
    race_intro_timer: int
    menu_selected_mode_raw: int
    menu_difficulty_state_raw: int
    menu_difficulty_cursor_raw: int
    menu_transition_state_raw: int
    menu_current_ghost_type_raw: int
    queued_game_mode_raw: int


class StepSummaryDict(TypedDict, total=False):
    """Per-step counters accumulated inside the native repeated-step loop."""

    frames_run: int
    max_race_distance: float
    max_race_distance_speed_kph: float
    reverse_active_frames: int
    collision_recoil_active_frames: int
    low_speed_frames: int
    energy_loss_total: float
    energy_gain_total: float
    damage_taken_frames: int
    consecutive_low_speed_frames: int
    entered_state_flags: int
    entered_course_effects: int
    final_frame_index: int
    airborne_frames: int
    outside_track_min_height_above_ground: float | None
    spin_macro_started: bool
    spin_macro_active_frames: int
    lean_macro_owned_frames: int
    impact_frames: int | None


class StepStatusDict(TypedDict, total=False):
    """Step-limit status returned by native repeated-step execution."""

    step_count: int
    stalled_steps: int
    reverse_timer: int
    progress_frontier_stalled_frames: int
    termination_reason: str | None
    truncation_reason: str | None
    spin_macro_active: bool
    spin_macro_frames_remaining: int
    spin_macro_cooldown_frames: int
