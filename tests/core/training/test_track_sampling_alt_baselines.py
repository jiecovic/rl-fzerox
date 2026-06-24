# tests/core/training/test_track_sampling_alt_baselines.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.domain.courses import X_CUP_COURSE
from rl_fzerox.core.runtime_spec.schema import TrackSamplingConfig, TrackSamplingEntryConfig
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingAltBaseline,
    apply_alt_baselines_to_track_sampling,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.alt_baselines import (
    alt_baseline_reset_variant_key,
)


def test_alt_baseline_projection_normalizes_within_base_candidate(
    tmp_path: Path,
) -> None:
    base_state = _touched(tmp_path / "base.state")
    first_alt = _touched(tmp_path / "first-alt.state")
    second_alt = _touched(tmp_path / "second-alt.state")
    config = TrackSamplingConfig(
        enabled=True,
        entries=(
            TrackSamplingEntryConfig(
                id="mute_city_gp_race_novice_blue_falcon",
                course_id="mute_city",
                mode="gp_race",
                gp_difficulty="novice",
                vehicle="blue_falcon",
                baseline_state_path=base_state,
                weight=3.0,
            ),
        ),
    )

    projected = apply_alt_baselines_to_track_sampling(
        config,
        (
            _baseline(
                baseline_id="alt-a",
                source_entry_id="mute_city_gp_race_novice_blue_falcon",
                state_path=first_alt,
                label="chicane approach",
            ),
            _baseline(
                baseline_id="alt-b",
                source_entry_id="mute_city_gp_race_novice_blue_falcon",
                state_path=second_alt,
                label="final lap",
            ),
        ),
    )

    assert [entry.id for entry in projected.entries] == [
        "mute_city_gp_race_novice_blue_falcon",
        "mute_city_gp_race_novice_blue_falcon__alt_alt-a",
        "mute_city_gp_race_novice_blue_falcon__alt_alt-b",
    ]
    assert [float(entry.weight) for entry in projected.entries] == [1.0, 1.0, 1.0]
    assert {entry.baseline_group_id for entry in projected.entries} == {
        "mute_city_gp_race_novice_blue_falcon"
    }
    assert projected.entries[1].baseline_state_path == first_alt
    assert projected.entries[1].alt_baseline_id == "alt-a"

    projected_again = apply_alt_baselines_to_track_sampling(projected, ())

    assert projected_again.entries == config.entries


def test_alt_baseline_projection_skips_missing_and_generated_x_cup_entries(
    tmp_path: Path,
) -> None:
    stable_state = _touched(tmp_path / "stable.state")
    missing_alt = tmp_path / "missing-alt.state"
    x_cup_state = _touched(tmp_path / "x-cup.state")
    config = TrackSamplingConfig(
        enabled=True,
        entries=(
            TrackSamplingEntryConfig(
                id="silence_gp_race_novice_blue_falcon",
                course_id="silence",
                mode="gp_race",
                gp_difficulty="novice",
                vehicle="blue_falcon",
                baseline_state_path=stable_state,
            ),
            TrackSamplingEntryConfig(
                id="x_cup_slot_1_gp_race_novice_blue_falcon",
                runtime_course_key="x_cup_slot_1",
                course_id="x_cup_seed",
                mode=X_CUP_COURSE.race_mode,
                course_index=X_CUP_COURSE.course_index,
                gp_difficulty="novice",
                vehicle="blue_falcon",
                baseline_state_path=x_cup_state,
                generated_course_kind=X_CUP_COURSE.generated_kind,
                generated_course_seed=123,
                generated_course_hash="seed",
                generated_course_slot=0,
                generated_course_generation=1,
            ),
        ),
    )

    projected = apply_alt_baselines_to_track_sampling(
        config,
        (
            _baseline(
                baseline_id="missing",
                source_entry_id="silence_gp_race_novice_blue_falcon",
                course_key="silence",
                state_path=missing_alt,
            ),
            _baseline(
                baseline_id="x-cup",
                source_entry_id="x_cup_slot_1_gp_race_novice_blue_falcon",
                course_key="x_cup_slot_1",
                state_path=x_cup_state,
            ),
        ),
    )

    assert projected.entries == config.entries


def _baseline(
    *,
    baseline_id: str,
    source_entry_id: str,
    state_path: Path,
    course_key: str = "mute_city",
    label: str = "watch snapshot",
) -> TrackSamplingAltBaseline:
    return TrackSamplingAltBaseline(
        id=baseline_id,
        run_id="run",
        course_key=course_key,
        reset_variant_key=alt_baseline_reset_variant_key(
            mode="gp_race",
            gp_difficulty="novice",
            vehicle="blue_falcon",
        ),
        source_entry_id=source_entry_id,
        label=label,
        state_path=state_path,
        weight=1.0,
        enabled=True,
        created_at="2026-06-13T10:00:00+00:00",
        updated_at="2026-06-13T10:00:00+00:00",
    )


def _touched(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"state")
    return path.resolve()
