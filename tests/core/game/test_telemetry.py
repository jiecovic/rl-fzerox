# tests/core/game/test_telemetry.py
from __future__ import annotations

from fzerox_emulator import (
    FZeroXTelemetry,
    PlayerTelemetry,
    StepSummary,
    encode_state_flags,
)


def test_native_player_telemetry_exposes_state_helpers() -> None:
    player = PlayerTelemetry(
        state_flags=(1 << 20) | (1 << 30),
        speed_kph=123.5,
        energy=92.25,
        max_energy=100.0,
        boost_timer=0,
        recoil_tilt_magnitude=0.5,
        reverse_timer=12,
        race_distance=12_345.5,
        lap_distance=2_345.5,
        race_time_ms=12_345,
        lap=2,
        laps_completed=1,
        position=3,
        machine_body_stat=4,
        machine_boost_stat=3,
        machine_grip_stat=2,
        machine_weight=1260,
        engine_setting=0.7,
    )

    assert player.can_boost is True
    assert player.active is True
    assert player.finished is False
    assert player.recoil_tilt_magnitude == 0.5
    assert player.course_effect_raw == 0
    assert player.course_effect_name == "none"
    assert player.on_energy_refill is False
    assert player.state_labels == ("can_boost", "active")
    assert player.machine_body_stat == 4
    assert player.machine_boost_stat == 3
    assert player.machine_grip_stat == 2
    assert player.machine_weight == 1260
    assert abs(player.engine_setting - 0.7) < 1e-6


def test_native_player_telemetry_decodes_course_effect_low_bits() -> None:
    player = PlayerTelemetry(
        state_flags=1 | (1 << 30),
        speed_kph=123.5,
        energy=92.25,
        max_energy=100.0,
        boost_timer=0,
        recoil_tilt_magnitude=0.5,
        reverse_timer=12,
        race_distance=12_345.5,
        lap_distance=2_345.5,
        race_time_ms=12_345,
        lap=2,
        laps_completed=1,
        position=3,
    )

    assert player.course_effect_raw == 1
    assert player.course_effect_name == "pit"
    assert player.on_energy_refill is True
    assert player.state_labels == ("active",)


def test_native_player_telemetry_ignores_refill_when_energy_is_full() -> None:
    player = PlayerTelemetry(
        state_flags=1 | (1 << 30),
        speed_kph=123.5,
        energy=100.0,
        max_energy=100.0,
        boost_timer=0,
        recoil_tilt_magnitude=0.5,
        reverse_timer=12,
        race_distance=12_345.5,
        lap_distance=2_345.5,
        race_time_ms=12_345,
        lap=2,
        laps_completed=1,
        position=3,
    )

    assert player.course_effect_raw == 1
    assert player.course_effect_name == "pit"
    assert player.on_energy_refill is False


def test_native_telemetry_to_dict_includes_nested_player_state() -> None:
    telemetry = FZeroXTelemetry(
        total_lap_count=3,
        game_mode_raw=1,
        game_mode_name="gp_race",
        in_race_mode=True,
        total_racers=30,
        course_index=0,
        course_length=80_000.0,
        player=PlayerTelemetry(
            state_flags=(1 << 20) | (1 << 30),
            speed_kph=123.5,
            energy=92.25,
            max_energy=100.0,
            boost_timer=0,
            recoil_tilt_magnitude=0.5,
            reverse_timer=12,
            race_distance=12_345.5,
            lap_distance=2_345.5,
            race_time_ms=12_345,
            lap=2,
            laps_completed=1,
            position=3,
            local_lateral_velocity=-9.5,
            signed_lateral_offset=42.0,
            machine_body_stat=4,
            machine_boost_stat=3,
            machine_grip_stat=2,
            machine_weight=1260,
            engine_setting=0.7,
        ),
        difficulty_raw=2,
        difficulty_name="expert",
        camera_setting_raw=3,
        camera_setting_name="wide",
        race_intro_timer=39,
    )

    payload = telemetry.to_dict()

    assert payload["total_lap_count"] == 3
    assert payload["difficulty_raw"] == 2
    assert payload["difficulty_name"] == "expert"
    assert payload["camera_setting_raw"] == 3
    assert payload["camera_setting_name"] == "wide"
    assert payload["race_intro_timer"] == 39
    assert payload["game_mode_name"] == "gp_race"
    assert payload["total_racers"] == 30
    assert payload["course_length"] == 80_000.0
    player_payload = payload["player"]
    assert isinstance(player_payload, dict)
    assert player_payload["recoil_tilt_magnitude"] == 0.5
    assert player_payload["machine_body_stat"] == 4
    assert player_payload["machine_boost_stat"] == 3
    assert player_payload["machine_grip_stat"] == 2
    assert player_payload["machine_weight"] == 1260
    assert isinstance(player_payload["engine_setting"], float)
    assert abs(player_payload["engine_setting"] - 0.7) < 1e-6
    assert player_payload["local_lateral_velocity"] == -9.5
    assert player_payload["signed_lateral_offset"] == 42.0
    assert player_payload["course_effect_raw"] == 0
    assert player_payload["course_effect_name"] == "none"
    assert player_payload["on_energy_refill"] is False
    assert player_payload["state_labels"] == ("can_boost", "active")


def test_native_step_summary_exposes_entered_state_helpers() -> None:
    summary = StepSummary(
        frames_run=2,
        max_race_distance=42.0,
        reverse_active_frames=1,
        low_speed_frames=2,
        energy_loss_total=4.0,
        energy_gain_total=2.5,
        damage_taken_frames=1,
        consecutive_low_speed_frames=2,
        entered_state_flags=(1 << 13) | (1 << 25),
        final_frame_index=12,
    )

    assert summary.entered_collision_recoil is True
    assert summary.entered_finished is True
    assert summary.entered_crashed is False
    assert summary.energy_gain_total == 2.5
    assert summary.damage_taken_frames == 1
    assert summary.reverse_active_frames == 1
    assert summary.low_speed_frames == 2
    assert summary.entered_state_labels == ("collision_recoil", "finished")


def test_encode_state_flags_builds_bitmask_from_labels() -> None:
    encoded = encode_state_flags(["collision_recoil", "finished"])

    assert encoded == (1 << 13) | (1 << 25)
