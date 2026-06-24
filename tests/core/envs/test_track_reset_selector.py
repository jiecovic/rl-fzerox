# tests/core/envs/test_track_reset_selector.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.engine_tuning import (
    EngineTuningContext,
    EngineTuningResetCandidate,
    EngineTuningResetContext,
    EngineTuningResetSampler,
)
from rl_fzerox.core.envs.engine.reset import (
    TrackResetSelector,
    select_reset_track_by_course_id,
)
from rl_fzerox.core.runtime_spec.schema import (
    AdaptiveEngineTuningConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
)


def test_select_reset_track_by_course_id_uses_matching_entry() -> None:
    config = TrackSamplingConfig(
        enabled=True,
        sampling_mode="equal",
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
        sampling_mode="equal",
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


def test_alt_baseline_selection_reports_base_track_id(tmp_path: Path) -> None:
    alt_state_path = tmp_path / "mute-alt.state"
    alt_state_path.write_bytes(b"state")
    config = TrackSamplingConfig(
        enabled=True,
        entries=(
            TrackSamplingEntryConfig(
                id="mute_city_gp_race_novice_blue_falcon__alt_alt-a",
                course_id="mute_city",
                baseline_state_path=alt_state_path,
                alt_baseline_id="alt-a",
                alt_baseline_label="chicane approach",
                alt_baseline_source_entry_id="mute_city_gp_race_novice_blue_falcon",
            ),
        ),
    )

    selected = TrackResetSelector().select(config, seed=123)

    assert selected is not None
    info = selected.info()
    assert info["track_entry_id"] == "mute_city_gp_race_novice_blue_falcon__alt_alt-a"
    assert info["track_id"] == "mute_city_gp_race_novice_blue_falcon"
    assert info["track_course_key"] == "mute_city"
    assert info["track_alt_baseline_id"] == "alt-a"


def test_reset_selector_skips_missing_alt_baseline_entries(tmp_path: Path) -> None:
    base_state_path = tmp_path / "mute.state"
    base_state_path.write_bytes(b"state")
    config = TrackSamplingConfig(
        enabled=True,
        entries=(
            TrackSamplingEntryConfig(
                id="mute_city_gp_race_novice_blue_falcon",
                course_id="mute_city",
                baseline_state_path=base_state_path,
                weight=1.0,
            ),
            TrackSamplingEntryConfig(
                id="mute_city_gp_race_novice_blue_falcon__alt_missing",
                course_id="mute_city",
                baseline_state_path=tmp_path / "missing.state",
                weight=100.0,
                alt_baseline_id="missing",
                alt_baseline_source_entry_id="mute_city_gp_race_novice_blue_falcon",
            ),
        ),
    )

    selected = TrackResetSelector().select(config, seed=123)

    assert selected is not None
    assert selected.id == "mute_city_gp_race_novice_blue_falcon"


def test_track_reset_selector_resyncs_when_entry_metadata_changes() -> None:
    selector = TrackResetSelector()
    original = TrackSamplingConfig(
        enabled=True,
        sampling_mode="equal",
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
        sampling_mode="equal",
        engine_tuning=AdaptiveEngineTuningConfig(
            enabled=True,
            min_raw_value=60,
            max_raw_value=70,
            prior_finish_time_seconds=200.0,
            observation_noise_seconds=0.25,
            curve_lengthscale_raw=1.0,
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
    context = EngineTuningContext(course_key="mute_city", vehicle_id="blue_falcon")
    sampler = EngineTuningResetSampler(
        contexts=(
            EngineTuningResetContext(
                context=context,
                candidates=(
                    EngineTuningResetCandidate(
                        engine_setting_raw_value=70,
                        probability=1.0,
                        mean_score=-80.0,
                        sampled_score=-80.0,
                        score_count=2,
                        finish_count=2,
                        estimated_finish_time_ms=80_000,
                        best_finish_time_ms=80_000,
                    ),
                ),
                greedy_engine_setting_raw_value=70,
            ),
        ),
    )

    selected = TrackResetSelector().select(config, seed=123, engine_tuning_sampler=sampler)

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
