# tests/ui/test_viewer_runtime_records.py
"""Watch runtime tests for track records and track-pool metadata.

The cases cover record-book aggregation, GP/X-Cup record identity, record-panel
rows, and the track-pool metadata projected into Watch side panels.
"""

from pathlib import Path

from rl_fzerox.core.runtime_spec.schema import (
    CareerModeRaceSetupConfig,
    EmulatorConfig,
    EnvConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.ui.watch.records import TrackRecordBook, track_record_key
from rl_fzerox.ui.watch.view.panels.content.records import track_record_sections
from rl_fzerox.ui.watch.view.screen.render import (
    _add_config_track_info,
    _track_pool_records,
)
from tests.ui.viewer_support import record_book, record_entry
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


def test_track_record_book_tracks_successful_finishes_per_track() -> None:
    book = TrackRecordBook()
    book = book.update(
        {"termination_reason": "crashed", "race_time_ms": 98_000, "track_id": "mute"},
        _sample_telemetry(race_time_ms=98_000, position=4),
        episode_done=True,
    )
    assert book.best_finish_position is None
    assert book.entries["mute"].failed_attempt is True

    book = book.update(
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=98_000, position=8),
        episode_done=True,
    )
    book = book.update(
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=101_000, position=12),
        episode_done=True,
    )
    book = book.update(
        {"termination_reason": "finished", "track_id": "silence"},
        _sample_telemetry(race_time_ms=105_000, position=5),
        episode_done=True,
    )
    book = book.update(
        {"termination_reason": "finished", "track_id": "mute"},
        _sample_telemetry(race_time_ms=95_000, position=3),
        episode_done=True,
    )

    assert book.best_finish_position == 3
    assert book.entries["mute"].best_finish_time_ms == 95_000
    assert book.entries["mute"].best_finish_time_rank == 3
    assert book.entries["mute"].best_finish_rank == 3
    assert book.entries["mute"].best_finish_rank_time_ms == 95_000
    assert book.entries["mute"].latest_finish_time_ms == 95_000
    assert book.entries["mute"].latest_finish_delta_ms == -3_000
    assert book.entries["mute"].failed_attempt is False
    assert book.entries["silence"].best_finish_time_ms == 105_000


def test_track_record_book_metadata_tracks_the_finish_that_set_the_record() -> None:
    book = TrackRecordBook()
    info = {
        "termination_reason": "finished",
        "track_id": "mute",
        "race_time_ms": 98_000,
        "position": 3,
        "track_vehicle_name": "Deep Claw",
        "track_engine_setting_raw_value": 60,
    }

    book = book.update(info, None, episode_done=True)
    slower_info: dict[str, object] = dict(
        info,
        race_time_ms=101_000,
        position=1,
        track_vehicle_name="Blue Falcon",
        track_engine_setting_raw_value=40,
    )
    book = book.update(
        slower_info,
        None,
        episode_done=True,
    )
    faster_same_rank_info: dict[str, object] = dict(
        info,
        race_time_ms=97_000,
        position=1,
        track_vehicle_name="Twin Noritta",
        track_engine_setting_raw_value=70,
    )
    book = book.update(
        faster_same_rank_info,
        None,
        episode_done=True,
    )

    entry = book.entries["mute"]
    assert entry.best_finish_time_ms == 97_000
    assert entry.best_finish_time_rank == 1
    assert entry.best_finish_time_setup == {
        "vehicle_name": "Twin Noritta",
        "engine_setting_raw_value": 70,
    }
    assert entry.best_finish_rank == 1
    assert entry.best_finish_rank_time_ms == 97_000
    assert entry.best_finish_rank_setup == {
        "vehicle_name": "Twin Noritta",
        "engine_setting_raw_value": 70,
    }
    assert entry.latest_finish_rank == 1
    assert entry.latest_finish_delta_ms == -1_000
    assert entry.latest_finish_setup == {
        "vehicle_name": "Twin Noritta",
        "engine_setting_raw_value": 70,
    }


def test_track_record_book_tracks_gp_difficulties_separately() -> None:
    novice_info: dict[str, object] = {
        "termination_reason": "finished",
        "track_id": "mute",
        "track_mode": "gp_race",
        "track_gp_difficulty": "novice",
    }
    expert_info: dict[str, object] = {
        "termination_reason": "finished",
        "track_id": "mute",
        "track_mode": "gp_race",
        "track_gp_difficulty": "expert",
    }
    novice_key = track_record_key(novice_info)
    expert_key = track_record_key(expert_info)
    assert novice_key is not None
    assert expert_key is not None

    book = TrackRecordBook()
    book = book.update(novice_info, _sample_telemetry(race_time_ms=98_000), episode_done=True)
    book = book.update(expert_info, _sample_telemetry(race_time_ms=101_000), episode_done=True)
    book = book.update(novice_info, _sample_telemetry(race_time_ms=95_000), episode_done=True)

    assert book.entries[novice_key].best_finish_time_ms == 95_000
    assert book.entries[expert_key].best_finish_time_ms == 101_000


def test_x_cup_record_key_uses_generated_hash_and_difficulty() -> None:
    assert (
        track_record_key(
            {
                "track_course_id": "x_cup_slot_1",
                "track_generated_course_kind": "x_cup",
                "track_generated_course_hash": "abcd1234",
                "track_gp_difficulty": "expert",
            }
        )
        == "x_cup:abcd1234#difficulty=expert"
    )


def test_track_record_book_tracks_failed_attempts_and_attempt_stats() -> None:
    book = TrackRecordBook()
    book = book.update(
        {
            "termination_reason": "crashed",
            "track_id": "mute",
            "episode_completion_fraction": 0.25,
        },
        None,
        episode_done=True,
    )
    book = book.update(
        {
            "termination_reason": "finished",
            "track_id": "mute",
            "episode_completion_fraction": 1.0,
        },
        None,
        episode_done=True,
    )
    book = book.update(
        {
            "termination_reason": "crashed",
            "track_id": "mute",
            "episode_completion_fraction": 0.5,
        },
        None,
        episode_done=False,
    )

    entry = book.entries["mute"]
    assert entry.failed_attempt is False
    assert entry.attempt_stats.as_mapping() == {
        "attempts": 2,
        "finishes": 1,
        "completion_samples": 2,
        "completion_sum": 1.25,
        "best_completion": 1.0,
    }


def test_record_panel_marks_failed_watch_attempts_until_success() -> None:
    records: tuple[dict[str, object], ...] = (
        {
            "track_id": "mute",
            "track_course_id": "mute_city",
            "track_course_name": "Mute City",
        },
    )

    failed_section = track_record_sections(
        current_info={},
        track_pool_records=records,
        track_record_book=record_book({"mute": record_entry(failed_attempt=True)}),
    )[0]
    success_section = track_record_sections(
        current_info={},
        track_pool_records=records,
        track_record_book=record_book(
            {
                "mute": record_entry(
                    best_finish_time_ms=95_000,
                    latest_finish_time_ms=95_000,
                    failed_attempt=True,
                )
            }
        ),
    )[0]

    assert failed_section.lines[0].status_text == "FAILED"
    assert failed_section.lines[2].value == "failed"
    assert success_section.lines[0].status_text == ""
    assert success_section.lines[2].value == "1:35.000"


def test_config_track_info_uses_registry_name_for_course_index(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    baseline_path = tmp_path / "mute.state"
    core_path.touch()
    rom_path.touch()
    baseline_path.write_bytes(b"baseline")
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute_city",
                        display_name="Mute City Time Attack - Blue Falcon Engine 50",
                        baseline_state_path=baseline_path,
                        course_index=0,
                    ),
                ),
            )
        ),
    )
    info: dict[str, object] = {"course_index": 0}

    _add_config_track_info(info, config)

    assert info["track_entry_id"] == "mute_city"
    assert info["track_id"] == "mute_city"
    assert info["track_course_key"] == "course:0"
    assert info["track_display_name"] == "Mute City Time Attack - Blue Falcon Engine 50"


def test_track_sampling_records_prefer_refreshed_watch_snapshot_state(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    old_baseline_path = tmp_path / "old.state"
    new_baseline_path = tmp_path / "new.state"
    core_path.touch()
    rom_path.touch()
    old_baseline_path.write_bytes(b"old")
    new_baseline_path.write_bytes(b"new")
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="old",
                        course_id="x_cup_old",
                        runtime_course_key="x_cup_slot_1",
                        baseline_state_path=old_baseline_path,
                    ),
                ),
            )
        ),
    )
    refreshed = TrackSamplingConfig(
        enabled=True,
        entries=(
            TrackSamplingEntryConfig(
                id="new",
                course_id="x_cup_new",
                runtime_course_key="x_cup_slot_1",
                baseline_state_path=new_baseline_path,
                mode="gp_race",
                course_index=48,
                generated_course_kind="x_cup",
                generated_course_seed=123,
                generated_course_hash="newhash",
                generated_course_slot=0,
                generated_course_generation=2,
            ),
        ),
    )

    records = _track_pool_records(config, active_track_sampling=refreshed)

    assert records[0]["track_entry_id"] == "new"
    assert records[0]["track_id"] == "new"
    assert records[0]["track_course_key"] == "x_cup_slot_1"
    assert records[0]["track_course_id"] == "x_cup_new"
    assert records[0]["track_reset_course_key"] == "x_cup_slot_1"
    assert records[0]["track_generated_course_kind"] == "x_cup"
    assert records[0]["track_generated_course_generation"] == 2


def test_track_sampling_records_count_in_memory_alt_baselines(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    base_path = tmp_path / "base.state"
    alt_a_path = tmp_path / "alt-a.state"
    alt_b_path = tmp_path / "alt-b.state"
    core_path.touch()
    rom_path.touch()
    base_path.write_bytes(b"base")
    alt_a_path.write_bytes(b"alt-a")
    alt_b_path.write_bytes(b"alt-b")
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute_city",
                        display_name="Mute City",
                        course_id="mute_city",
                        baseline_state_path=base_path,
                    ),
                    TrackSamplingEntryConfig(
                        id="mute_city__alt_alt-a",
                        display_name="Mute City alt A",
                        course_id="mute_city",
                        baseline_state_path=alt_a_path,
                        alt_baseline_id="alt-a",
                        alt_baseline_label="frame 100",
                        alt_baseline_source_entry_id="mute_city",
                    ),
                    TrackSamplingEntryConfig(
                        id="mute_city__alt_alt-b",
                        display_name="Mute City alt B",
                        course_id="mute_city",
                        baseline_state_path=alt_b_path,
                        alt_baseline_id="alt-b",
                        alt_baseline_label="frame 200",
                        alt_baseline_source_entry_id="mute_city",
                    ),
                ),
            )
        ),
    )

    records = _track_pool_records(config)

    assert records[0]["track_alt_baseline_count"] == 2
    assert records[1]["track_alt_baseline_count"] == 0
    assert records[1]["track_alt_baseline_id"] == "alt-a"


def test_career_mode_track_pool_records_cover_selected_cup(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        watch=WatchConfig(
            managed_save_game_id="save-a",
            career_mode_race_setup=CareerModeRaceSetupConfig(
                difficulty="master",
                cup_id="joker",
                vehicle_id="blue_falcon",
                vehicle_display_name="Blue Falcon",
                character_index=0,
                machine_select_slot=0,
                machine_select_row=0,
                machine_select_column=0,
                engine_setting_raw_value=80,
            ),
        ),
    )

    records = _track_pool_records(config)

    assert [record["track_course_name"] for record in records] == [
        "Rainbow Road",
        "Devil's Forest 3",
        "Space Plant",
        "Sand Ocean 2",
        "Port Town 2",
        "Big Hand",
    ]
    assert {record["track_gp_difficulty"] for record in records} == {"master"}
    assert {record["track_vehicle_name"] for record in records} == {"Blue Falcon"}
    assert {record["track_engine_setting_raw_value"] for record in records} == {80}
    best_time = records[0]["track_non_agg_best_time_ms"]
    assert isinstance(best_time, int)
    assert best_time > 0


def test_record_rows_click_stable_runtime_course_key() -> None:
    section = track_record_sections(
        current_info={},
        track_pool_records=(
            {
                "track_id": "generated",
                "track_course_id": "x_cup_generated",
                "track_reset_course_key": "x_cup_slot_1",
            },
        ),
        track_record_book=record_book(),
    )[0]

    assert section.lines[0].click_course_id == "x_cup_slot_1"


def test_record_rows_can_disable_course_jump_clicks() -> None:
    section = track_record_sections(
        current_info={},
        track_pool_records=(
            {
                "track_id": "mute_city",
                "track_course_id": "mute_city",
                "track_reset_course_key": "mute_city",
            },
        ),
        track_record_book=record_book(),
        allow_course_jumps=False,
    )[0]

    assert section.lines[0].click_course_id is None
