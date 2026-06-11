# tests/ui/test_viewer_game_panel_records.py
from rl_fzerox.ui.watch.records import track_record_key
from rl_fzerox.ui.watch.view.panels.core.model import _build_panel_columns
from rl_fzerox.ui.watch.view.panels.rendering.draw import _record_tab_sections
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from tests.ui.viewer_game_panel_support import (
    race_control_state,
)
from tests.ui.viewer_support import record_book, record_entry
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


def test_records_section_shows_non_agg_reference_records() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_non_agg_best_time_ms": 48035,
            "track_non_agg_best_player": "Daniel",
            "track_non_agg_worst_time_ms": 75846,
            "track_non_agg_worst_player": "FTQ",
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        track_pool_records=(
            {
                "track_display_name": "Big Blue Time Attack - Blue Falcon Engine 50",
                "track_id": "big_blue",
                "track_non_agg_best_time_ms": 48035,
                "track_non_agg_worst_time_ms": 75846,
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    header_line = next(line for line in records_section.lines if line.label == "Big Blue")
    pb_line = next(line for line in records_section.lines if line.label == "Best time")
    wr_line = next(line for line in records_section.lines if line.label == "WR")
    assert header_line.value == ""
    assert header_line.status_icon == "none"
    assert header_line.status_text == ""
    assert pb_line.value == "--"
    assert wr_line.value == "48.035 - 1:15.846"


def test_records_section_shows_watch_best_for_track_pool() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_id": "silence",
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        track_record_book=record_book(
            {
                "silence": record_entry(
                    best_finish_time_ms=98765,
                    latest_finish_rank=5,
                    latest_finish_time_ms=101234,
                    latest_finish_setup={
                        "vehicle_name": "Blue Falcon",
                        "engine_setting_raw_value": 40,
                    },
                    attempt_stats={
                        "attempts": 3,
                        "finishes": 1,
                        "completion_samples": 3,
                        "completion_sum": 2.25,
                        "best_completion": 1.0,
                    },
                )
            }
        ),
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence Time Attack - Blue Falcon Engine 50",
                "track_mode": "gp_race",
                "track_non_agg_best_time_ms": 60638,
                "track_non_agg_worst_time_ms": 63279,
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    header_line = next(line for line in records_section.lines if line.label == "> Silence")
    pb_line = next(line for line in records_section.lines if line.label == "Best time")
    latest_line = next(line for line in records_section.lines if line.label == "Latest")
    attempts_line = next(line for line in records_section.lines if line.label == "Attempts")
    assert header_line.value == ""
    assert header_line.status_icon == "outside"
    assert header_line.status_text == "+35.5s"
    assert pb_line.value == "1:38.765"
    assert latest_line.value == "P5 · 1:41.234 (+2.5s) · Blue Falcon / Engine 40"
    assert attempts_line.value == "3 · finish 33.3% · comp 75.0%"


def test_records_section_shows_gp_best_rank_with_watch_best_time() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_id": "silence",
            "track_mode": "gp_race",
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        track_record_book=record_book(
            {
                "silence": record_entry(
                    best_finish_rank=1,
                    best_finish_rank_time_ms=101_000,
                    best_finish_rank_setup={
                        "vehicle_name": "Deep Claw",
                        "engine_setting_raw_value": 60,
                    },
                    best_finish_time_ms=98765,
                    best_finish_time_rank=2,
                    best_finish_time_setup={
                        "vehicle_name": "Blue Falcon",
                        "engine_setting_raw_value": 50,
                    },
                )
            }
        ),
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence GP Race - Blue Falcon Engine 50",
                "track_mode": "gp_race",
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    best_time_line = next(line for line in records_section.lines if line.label == "Best time")
    best_position_line = next(line for line in records_section.lines if line.label == "Best pos")

    assert best_time_line.value == "1:38.765 · P2 · Blue Falcon / Engine 50"
    assert best_position_line.value == "P1 · 1:41.000 · Deep Claw / Engine 60"


def test_records_section_groups_track_pool_by_cup() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        track_pool_records=(
            {
                "track_id": "port_town_2",
                "track_course_ref": "queen/port_town_2",
                "track_course_name": "Port Town II",
            },
            {
                "track_id": "mute_city",
                "track_course_ref": "jack/mute_city",
                "track_course_name": "Mute City",
            },
            {
                "track_id": "silence",
                "track_course_ref": "jack/silence",
                "track_course_name": "Silence",
            },
        ),
    )

    assert [section.title for section in columns.records] == ["Jack Cup", "Queen Cup"]
    assert [line.label for line in columns.records[0].lines if line.heading] == [
        "Mute City",
        "Silence",
    ]
    assert [line.label for line in columns.records[1].lines if line.heading] == ["Port Town II"]
    assert [section.title for section in _record_tab_sections(columns.records, 1)] == ["Queen Cup"]


def test_records_section_groups_generated_x_cup_records() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        track_pool_records=(
            {
                "track_id": "mute_city",
                "track_course_ref": "jack/mute_city",
                "track_course_name": "Mute City",
            },
            {
                "track_id": "x_cup_d6a1a626",
                "track_course_id": "x_cup_d6a1a626",
                "track_runtime_course_key": "x_cup_slot_1",
                "track_course_name": "Space Plant",
                "track_course_index": 48,
                "track_generated_course_kind": "x_cup",
            },
        ),
    )

    assert [section.title for section in columns.records] == ["Jack Cup", "X Cup"]
    assert [line.label for line in columns.records[1].lines if line.heading] == ["Space Plant"]


def test_records_section_dedupes_course_variants_to_one_course_row() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={"frame_index": 0, "native_fps": 60.0, "track_course_id": "mute_city"},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        track_record_book=record_book({"mute_city": record_entry(best_finish_time_ms=88_000)}),
        track_pool_records=(
            {
                "track_id": "mute_city_blue_falcon",
                "track_course_id": "mute_city",
                "track_course_name": "Mute City",
                "track_vehicle": "blue_falcon",
            },
            {
                "track_id": "mute_city_fire_stingray",
                "track_course_id": "mute_city",
                "track_course_name": "Mute City",
                "track_vehicle": "fire_stingray",
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    headings = [line.label for line in records_section.lines if line.heading]

    assert headings == ["> Mute City"]


def test_records_section_follows_selected_gp_difficulty() -> None:
    novice_record: dict[str, object] = {
        "track_id": "mute_city_novice",
        "track_course_id": "mute_city",
        "track_course_name": "Mute City",
        "track_mode": "gp_race",
        "track_gp_difficulty": "novice",
    }
    expert_record: dict[str, object] = {
        "track_id": "mute_city_expert",
        "track_course_id": "mute_city",
        "track_course_name": "Mute City",
        "track_mode": "gp_race",
        "track_gp_difficulty": "expert",
    }
    novice_key = track_record_key(novice_record)
    expert_key = track_record_key(expert_record)
    assert novice_key is not None
    assert expert_key is not None
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_course_id": "mute_city",
            "track_mode": "gp_race",
            "track_gp_difficulty": "expert",
            "watch_selected_gp_difficulty": "expert",
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(difficulty_name="expert", difficulty_raw=2),
        track_record_book=record_book(
            {
                novice_key: record_entry(
                    best_finish_rank=3,
                    best_finish_rank_time_ms=92_000,
                    best_finish_time_ms=92_000,
                    best_finish_time_rank=3,
                ),
                expert_key: record_entry(
                    best_finish_rank=1,
                    best_finish_rank_time_ms=102_000,
                    best_finish_time_ms=98_000,
                    best_finish_time_rank=1,
                ),
            }
        ),
        track_pool_records=(novice_record, expert_record),
    )

    assert [section.title for section in columns.records] == ["Records"]
    headings = [line.label for line in columns.records[0].lines if line.heading]
    expert_pb_line = next(line for line in columns.records[0].lines if line.label == "Best time")
    expert_pos_line = next(line for line in columns.records[0].lines if line.label == "Best pos")

    assert headings == ["> Mute City"]
    assert expert_pb_line.value == "1:38.000 · P1"
    assert expert_pos_line.value == "P1 · 1:42.000"


def test_records_section_highlights_current_track_heading() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_id": "silence",
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        track_pool_records=(
            {
                "track_id": "mute_city",
                "track_course_name": "Mute City",
            },
            {
                "track_id": "silence",
                "track_course_name": "Silence",
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    mute_line = next(line for line in records_section.lines if line.label == "Mute City")
    silence_line = next(line for line in records_section.lines if line.label == "> Silence")

    assert mute_line.label_color is None
    assert silence_line.label_color == PALETTE.text_accent
    assert silence_line.status_text == "LIVE"


def test_records_section_shows_latest_improvement_against_previous_pb() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_id": "silence",
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        track_record_book=record_book(
            {
                "silence": record_entry(
                    best_finish_time_ms=97_530,
                    latest_finish_time_ms=97_530,
                    latest_finish_delta_ms=-1_235,
                )
            }
        ),
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence Time Attack - Blue Falcon Engine 50",
                "track_non_agg_best_time_ms": 60638,
                "track_non_agg_worst_time_ms": 63279,
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    latest_line = next(line for line in records_section.lines if line.label == "Latest")
    assert latest_line.value == "1:37.530 (-1.2s)"


def test_records_section_marks_watch_best_inside_reference_range() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_id": "silence",
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        track_record_book=record_book({"silence": record_entry(best_finish_time_ms=62_000)}),
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence Time Attack - Blue Falcon Engine 50",
                "track_non_agg_best_time_ms": 60638,
                "track_non_agg_worst_time_ms": 63279,
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    header_line = next(line for line in records_section.lines if line.label == "> Silence")
    pb_line = next(line for line in records_section.lines if line.label == "Best time")
    assert header_line.value == ""
    assert header_line.status_icon == "in_range"
    assert header_line.status_text == "+1.4s"
    assert pb_line.value == "1:02.000"


def test_records_section_formats_minute_scale_reference_gap() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={
            "frame_index": 0,
            "native_fps": 60.0,
            "track_id": "silence",
        },
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=race_control_state(),
        policy_curriculum_stage=None,
        policy_action=None,
        policy_reload_age_seconds=None,
        policy_reload_error=None,
        action_repeat=3,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
        track_record_book=record_book({"silence": record_entry(best_finish_time_ms=138_379)}),
        track_pool_records=(
            {
                "track_id": "silence",
                "track_display_name": "Silence Time Attack - Blue Falcon Engine 50",
                "track_non_agg_best_time_ms": 60638,
                "track_non_agg_worst_time_ms": 63279,
            },
        ),
    )

    records_section = next(section for section in columns.records if section.title == "Records")
    header_line = next(line for line in records_section.lines if line.label == "> Silence")
    assert header_line.status_icon == "outside"
    assert header_line.status_text == "+1min 15.1s"
