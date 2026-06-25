# tests/core/evaluation/test_targets.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.evaluation.models import EvaluationTargetSpec
from rl_fzerox.core.evaluation.targets import single_course_targets_from_config
from rl_fzerox.core.manager import default_managed_run_config
from rl_fzerox.core.manager.training import build_managed_train_app_config
from rl_fzerox.core.runtime_spec.schema import (
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
)


def test_single_course_targets_filter_gp_entries_by_target_metadata(
    tmp_path: Path,
) -> None:
    config = _config_with_entries(
        tmp_path,
        (
            TrackSamplingEntryConfig(
                id="mute-city-master-blue",
                course_ref="jack/mute_city",
                course_id="mute_city",
                course_name="Mute City",
                mode="gp_race",
                gp_difficulty="master",
                vehicle="blue_falcon",
                baseline_state_path=tmp_path / "mute-city.state",
                engine_setting_raw_value=64,
            ),
            TrackSamplingEntryConfig(
                id="mute-city-expert-blue",
                course_ref="jack/mute_city",
                course_id="mute_city",
                mode="gp_race",
                gp_difficulty="expert",
                vehicle="blue_falcon",
            ),
            TrackSamplingEntryConfig(
                id="port-town-master-blue",
                course_ref="queen/port_town",
                course_id="port_town",
                mode="gp_race",
                gp_difficulty="master",
                vehicle="blue_falcon",
            ),
            TrackSamplingEntryConfig(
                id="mute-city-master-fox",
                course_ref="jack/mute_city",
                course_id="mute_city",
                mode="gp_race",
                gp_difficulty="master",
                vehicle="golden_fox",
            ),
            TrackSamplingEntryConfig(
                id="mute-city-master-blue-alt",
                course_ref="jack/mute_city",
                course_id="mute_city",
                mode="gp_race",
                gp_difficulty="master",
                vehicle="blue_falcon",
                alt_baseline_id="variant-b",
            ),
            TrackSamplingEntryConfig(
                id="mute-city-time-attack",
                course_id="mute_city",
                mode="time_attack",
                vehicle="blue_falcon",
            ),
        ),
    )

    targets = single_course_targets_from_config(
        config,
        EvaluationTargetSpec(
            mode="gp_course",
            cup_ids=("jack",),
            difficulties=("master",),
            vehicle_ids=("blue_falcon",),
        ),
    )

    assert len(targets) == 1
    assert targets[0].target_id == "mute-city-master-blue"
    assert targets[0].course_id == "mute_city"
    assert targets[0].course_name == "Mute City"
    assert targets[0].cup_id == "jack"
    assert targets[0].difficulty == "master"
    assert targets[0].vehicle_id == "blue_falcon"
    assert targets[0].baseline_state_path == str(tmp_path / "mute-city.state")
    assert targets[0].engine_setting_raw_value == 64


def test_single_course_targets_match_time_attack_by_course_id_or_entry_id(
    tmp_path: Path,
) -> None:
    config = _config_with_entries(
        tmp_path,
        (
            TrackSamplingEntryConfig(
                id="slot-a",
                course_id="mute_city",
                course_name="Mute City",
                mode="time_attack",
                vehicle="blue_falcon",
            ),
            TrackSamplingEntryConfig(
                id="slot-b",
                course_id="silence",
                course_name="Silence",
                mode="time_attack",
                vehicle="blue_falcon",
                engine_setting_raw_value=85,
            ),
            TrackSamplingEntryConfig(
                id="slot-c",
                course_id="sand_ocean",
                mode="gp_race",
                gp_difficulty="master",
                vehicle="blue_falcon",
            ),
        ),
    )

    targets = single_course_targets_from_config(
        config,
        EvaluationTargetSpec(
            mode="time_attack_course",
            course_ids=("mute_city", "slot-b"),
            vehicle_ids=("blue_falcon",),
        ),
    )

    assert [target.target_id for target in targets] == ["slot-a", "slot-b"]
    assert [target.course_id for target in targets] == ["mute_city", "silence"]
    assert targets[1].engine_setting_raw_value == 85


def _config_with_entries(
    tmp_path: Path,
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> TrainAppConfig:
    config = build_managed_train_app_config(
        default_managed_run_config(),
        run_id="eval-target-test",
        run_dir=tmp_path / "eval-target-test",
    )
    return config.model_copy(
        update={
            "env": config.env.model_copy(
                update={
                    "track_sampling": TrackSamplingConfig(
                        enabled=True,
                        entries=entries,
                    )
                }
            )
        }
    )
