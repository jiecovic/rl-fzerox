# tests/core/envs/test_track_reset_selector.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.engine_tuning import EngineTuningArmState, EngineTuningRuntimeState
from rl_fzerox.core.envs.engine.reset import TrackResetSelector, select_reset_track_by_course_id
from rl_fzerox.core.runtime_spec.schema import (
    AdaptiveEngineTuningConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
)


def test_select_reset_track_by_course_id_uses_matching_entry() -> None:
    config = TrackSamplingConfig(
        enabled=True,
        sampling_mode="balanced",
        entries=(
            TrackSamplingEntryConfig(
                id="mute_city",
                course_id="mute_city",
                course_name="Mute City",
                baseline_state_path=Path("mute_city.state"),
            ),
            TrackSamplingEntryConfig(
                id="silence",
                course_id="silence",
                course_name="Silence",
                baseline_state_path=Path("silence.state"),
            ),
        ),
    )

    selected = select_reset_track_by_course_id(config, course_id="silence")

    assert selected is not None
    assert selected.id == "silence"
    assert selected.course_id == "silence"
    assert selected.sampling_mode == "locked"


def test_select_reset_track_by_course_id_can_label_external_scheduler() -> None:
    config = TrackSamplingConfig(
        enabled=True,
        sampling_mode="balanced",
        entries=(
            TrackSamplingEntryConfig(
                id="silence",
                course_id="silence",
                course_name="Silence",
                baseline_state_path=Path("silence.state"),
            ),
        ),
    )

    selected = select_reset_track_by_course_id(
        config,
        course_id="silence",
        sampling_mode="deficit_budget",
    )

    assert selected is not None
    assert selected.sampling_mode == "deficit_budget"


def test_select_reset_track_by_course_id_returns_none_without_match() -> None:
    config = TrackSamplingConfig(
        enabled=True,
        entries=(
            TrackSamplingEntryConfig(
                id="mute_city",
                course_id="mute_city",
                baseline_state_path=Path("mute_city.state"),
            ),
        ),
    )

    assert select_reset_track_by_course_id(config, course_id="silence") is None


def test_track_reset_selector_resyncs_when_entry_metadata_changes() -> None:
    selector = TrackResetSelector()
    original = TrackSamplingConfig(
        enabled=True,
        sampling_mode="balanced",
        entries=(
            TrackSamplingEntryConfig(
                id="mute_city",
                course_id="mute_city",
                baseline_state_path=Path("mute_city.state"),
                engine_setting_raw_value=20,
            ),
        ),
    )
    updated = original.model_copy(
        update={
            "entries": (
                original.entries[0].model_copy(
                    update={
                        "engine_setting_raw_value": 80,
                    }
                ),
            )
        }
    )

    first = selector.select(original, seed=123)
    second = selector.select(updated, seed=123)

    assert first is not None
    assert second is not None
    assert first.engine_setting_raw_value == 20
    assert second.engine_setting_raw_value == 80


def test_track_reset_selector_applies_adaptive_engine_choice() -> None:
    config = TrackSamplingConfig(
        enabled=True,
        sampling_mode="balanced",
        engine_tuning=AdaptiveEngineTuningConfig(
            enabled=True,
            min_raw_value=60,
            max_raw_value=70,
            bin_size=10,
            prior_mean=0.0,
            prior_strength=0.0,
            exploration_scale=0.0,
            uniform_exploration=0.0,
        ),
        entries=(
            TrackSamplingEntryConfig(
                id="mute_city",
                course_id="mute_city",
                source_vehicle="blue_falcon",
                baseline_state_path=Path("mute_city.state"),
            ),
        ),
    )
    state = EngineTuningRuntimeState(
        version=1,
        update_count=1,
        arms=(
            EngineTuningArmState(
                context_key="mute_city|blue_falcon",
                course_key="mute_city",
                vehicle_id="blue_falcon",
                engine_setting_raw_value=70,
                attempts=2,
                decayed_count=2.0,
                decayed_score_total=3.0,
            ),
        ),
    )

    selected = TrackResetSelector().select(config, seed=123, engine_tuning_state=state)

    assert selected is not None
    assert selected.engine_setting_raw_value == 70
    assert selected.engine_tuning_context_key == "mute_city|blue_falcon"
    assert selected.engine_tuning_course_key == "mute_city"
    assert selected.engine_tuning_vehicle_id == "blue_falcon"


def test_fixed_env_track_sampling_pins_course_by_env_index() -> None:
    config = TrackSamplingConfig(
        enabled=True,
        sampling_mode="fixed_env",
        entries=(
            TrackSamplingEntryConfig(
                id="mute_city",
                course_id="mute_city",
                baseline_state_path=Path("mute_city.state"),
            ),
            TrackSamplingEntryConfig(
                id="silence",
                course_id="silence",
                baseline_state_path=Path("silence.state"),
            ),
            TrackSamplingEntryConfig(
                id="sand_ocean",
                course_id="sand_ocean",
                baseline_state_path=Path("sand_ocean.state"),
            ),
        ),
    )

    selected_by_env = tuple(
        TrackResetSelector(env_index=env_index).select(config, seed=123) for env_index in range(4)
    )

    selected_course_ids = [
        selected.course_id if selected is not None else None for selected in selected_by_env
    ]
    assert selected_course_ids == ["mute_city", "silence", "sand_ocean", "mute_city"]
    assert selected_by_env[0] is not None
    assert selected_by_env[0].sampling_mode == "fixed_env"
    assert selected_by_env[0].cycle_position == 0
    assert selected_by_env[3] is not None
    assert selected_by_env[3].cycle_position == 0


def test_deficit_budget_track_sampling_falls_back_to_env_index_assignment() -> None:
    config = TrackSamplingConfig(
        enabled=True,
        sampling_mode="deficit_budget",
        entries=(
            TrackSamplingEntryConfig(
                id="mute_city",
                course_id="mute_city",
                baseline_state_path=Path("mute_city.state"),
            ),
            TrackSamplingEntryConfig(
                id="silence",
                course_id="silence",
                baseline_state_path=Path("silence.state"),
            ),
        ),
    )

    selected = TrackResetSelector(env_index=3).select(config, seed=123)

    assert selected is not None
    assert selected.course_id == "silence"
    assert selected.sampling_mode == "deficit_budget"
    assert selected.cycle_position == 1
