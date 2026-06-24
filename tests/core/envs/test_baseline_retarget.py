# tests/core/envs/test_baseline_retarget.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.domain.race_difficulty import RaceDifficultyName
from rl_fzerox.core.envs.engine.reset.gp_race import retarget_gp_race_baseline
from rl_fzerox.core.envs.engine.reset.time_attack import retarget_time_attack_baseline
from rl_fzerox.core.envs.engine.reset.tracks import SelectedTrack
from tests.support.fakes import SyntheticBackend


def test_gp_race_retarget_skips_canonical_noop_engine_patch() -> None:
    backend = SyntheticBackend()
    info: dict[str, object] = {}

    retarget_gp_race_baseline(
        backend=backend,
        selected_track=_selected_track(mode="gp_race", alt_baseline_id=None),
        telemetry=None,
        info=info,
    )

    assert getattr(backend, "last_engine_settings", None) is None
    assert "track_gp_race_retargeted" not in info


def test_gp_race_retarget_patches_alt_baseline_even_when_source_matches() -> None:
    backend = SyntheticBackend()
    info: dict[str, object] = {}

    retarget_gp_race_baseline(
        backend=backend,
        selected_track=_selected_track(mode="gp_race", alt_baseline_id="alt-a"),
        telemetry=None,
        info=info,
    )

    assert backend.last_engine_settings == {
        "mode": "gp_race",
        "engine_setting_raw_value": 50,
    }
    assert info["track_gp_race_retargeted"] is True


def test_time_attack_retarget_skips_canonical_noop_engine_patch() -> None:
    backend = SyntheticBackend()
    info: dict[str, object] = {}

    retarget_time_attack_baseline(
        backend=backend,
        selected_track=_selected_track(mode="time_attack", alt_baseline_id=None),
        telemetry=None,
        info=info,
    )

    assert getattr(backend, "last_engine_settings", None) is None
    assert "track_time_attack_retargeted" not in info


def test_time_attack_retarget_patches_alt_baseline_even_when_source_matches() -> None:
    backend = SyntheticBackend()
    info: dict[str, object] = {}

    retarget_time_attack_baseline(
        backend=backend,
        selected_track=_selected_track(mode="time_attack", alt_baseline_id="alt-a"),
        telemetry=None,
        info=info,
    )

    assert backend.last_engine_settings == {
        "mode": "time_attack",
        "engine_setting_raw_value": 50,
    }
    assert info["track_time_attack_retargeted"] is True


def _selected_track(*, mode: str, alt_baseline_id: str | None) -> SelectedTrack:
    gp_difficulty: RaceDifficultyName | None = "master" if mode == "gp_race" else None
    return SelectedTrack(
        id="mute_city",
        display_name="Mute City",
        course_ref=None,
        course_id="mute_city",
        runtime_course_key="mute_city",
        course_name="Mute City",
        baseline_state_path=Path("mute.state"),
        weight=1.0,
        course_index=0,
        mode=mode,
        gp_difficulty=gp_difficulty,
        vehicle="blue_falcon",
        vehicle_name="Blue Falcon",
        engine_setting_raw_value=50,
        engine_setting_min_raw_value=None,
        engine_setting_max_raw_value=None,
        engine_tuning_context_key=None,
        engine_tuning_course_key=None,
        engine_tuning_vehicle_id=None,
        engine_tuning_sampled_score=None,
        engine_tuning_mean_score=None,
        engine_tuning_finish_count=None,
        source_vehicle="blue_falcon",
        source_course_index=0,
        source_gp_difficulty=gp_difficulty,
        source_engine_setting_raw_value=50,
        baseline_group_id=None,
        baseline_group_weight=None,
        baseline_variant_index=None,
        baseline_variant_count=None,
        baseline_variant_seed=None,
        alt_baseline_id=alt_baseline_id,
        alt_baseline_label=None,
        alt_baseline_source_entry_id="mute_city_base" if alt_baseline_id is not None else None,
        generated_course_kind=None,
        generated_course_seed=None,
        generated_course_hash=None,
        generated_course_slot=None,
        generated_course_generation=None,
        generated_course_segment_count=None,
        generated_course_length=None,
        log_per_course=False,
        records=None,
        sampling_mode="equal",
    )
