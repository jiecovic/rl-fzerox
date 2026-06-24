# tests/core/manager/test_training_bridge_tracks_fork.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.domain.courses import X_CUP_COURSE
from rl_fzerox.core.manager import ManagedRunConfig, default_managed_run_config
from rl_fzerox.core.manager.training import (
    assert_managed_fork_compatible,
    build_managed_fork_train_app_config,
    build_managed_resume_train_app_config,
    build_managed_train_app_config,
)


def test_manager_training_bridge_supports_vehicle_pools_and_random_engine_ranges(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.vehicle.selection_mode = "pool"
    config.vehicle.selected_vehicle_ids = ("blue_falcon", "golden_fox")
    config.vehicle.engine_mode = "random_range"
    config.vehicle.engine_setting_min_raw_value = 20
    config.vehicle.engine_setting_max_raw_value = 80
    config.tracks.selected_course_ids = ("mute_city", "silence")

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-pool",
        run_dir=tmp_path / "runs" / "bridge-pool_0001",
    )

    entries = train_config.env.track_sampling.entries
    assert len(entries) == 4
    assert {entry.vehicle for entry in entries} == {"blue_falcon", "golden_fox"}
    assert {entry.course_id for entry in entries} == {"mute_city", "silence"}
    assert {entry.source_vehicle for entry in entries} == {"blue_falcon"}
    assert {entry.engine_setting_min_raw_value for entry in entries} == {20}
    assert {entry.engine_setting_max_raw_value for entry in entries} == {80}
    assert {entry.source_engine_setting_raw_value for entry in entries} == {50}


def test_manager_training_bridge_supports_built_in_gp_race_launch(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.race_mode = "gp_race"
    config.tracks.selected_course_ids = ("mute_city", "silence")

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-gp",
        run_dir=tmp_path / "runs" / "bridge-gp_0001",
    )

    assert {entry.mode for entry in train_config.env.track_sampling.entries} == {"gp_race"}
    assert {entry.gp_difficulty for entry in train_config.env.track_sampling.entries} == {"novice"}
    assert {entry.course_id for entry in train_config.env.track_sampling.entries} == {
        "mute_city",
        "silence",
    }
    assert {entry.course_index for entry in train_config.env.track_sampling.entries} == {0, 1}
    assert {entry.source_course_index for entry in train_config.env.track_sampling.entries} == {
        0,
        1,
    }


def test_manager_training_bridge_adds_generated_x_cup_entries(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.race_mode = "gp_race"
    config.tracks.include_x_cup = True
    config.tracks.x_cup_course_count = 2
    config.tracks.selected_course_ids = ()

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-x-cup",
        run_dir=tmp_path / "runs" / "bridge-x-cup_0001",
    )

    entries = train_config.env.track_sampling.entries
    assert len(entries) == 2
    assert {entry.generated_course_kind for entry in entries} == {X_CUP_COURSE.generated_kind}
    assert {entry.course_index for entry in entries} == {X_CUP_COURSE.course_index}
    assert {entry.source_course_index for entry in entries} == {X_CUP_COURSE.course_index}
    assert {entry.log_per_course for entry in entries} == {False}
    assert len({entry.course_id for entry in entries}) == 2
    assert all(entry.generated_course_seed is not None for entry in entries)
    assert all(entry.generated_course_hash is not None for entry in entries)
    assert {entry.generated_course_slot for entry in entries} == {0, 1}
    assert {entry.generated_course_generation for entry in entries} == {1}


def test_manager_training_bridge_groups_generated_x_cup_difficulty_baselines(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.race_mode = "gp_race"
    config.tracks.gp_difficulties = ("novice", "expert")
    config.tracks.include_x_cup = True
    config.tracks.x_cup_course_count = 2
    config.tracks.selected_course_ids = ()

    train_config = build_managed_train_app_config(
        ManagedRunConfig.model_validate(config.model_dump(mode="python")),
        run_id="bridge-x-cup-difficulties",
        run_dir=tmp_path / "runs" / "bridge-x-cup-difficulties_0001",
    )

    entries = train_config.env.track_sampling.entries
    assert len(entries) == 4
    assert {entry.gp_difficulty for entry in entries} == {"novice", "expert"}
    assert {entry.runtime_course_key for entry in entries} == {
        "x_cup_slot_1",
        "x_cup_slot_2",
    }
    assert len({entry.course_id for entry in entries}) == 2
    assert len({entry.generated_course_hash for entry in entries}) == 2
    for slot in (0, 1):
        slot_entries = tuple(entry for entry in entries if entry.generated_course_slot == slot)
        assert {entry.gp_difficulty for entry in slot_entries} == {"novice", "expert"}
        assert len({entry.course_id for entry in slot_entries}) == 1
        assert len({entry.generated_course_hash for entry in slot_entries}) == 1


def test_manager_config_omits_gp_difficulties_outside_gp_race() -> None:
    config = default_managed_run_config()

    dumped = config.model_dump(mode="json")

    assert dumped["tracks"]["race_mode"] == "time_attack"
    assert "gp_difficulties" not in dumped["tracks"]


def test_manager_config_rejects_legacy_gp_difficulty() -> None:
    with pytest.raises(ValueError, match="gp_difficulty"):
        ManagedRunConfig.model_validate(
            {
                **default_managed_run_config().model_dump(mode="json"),
                "tracks": {
                    **default_managed_run_config().tracks.model_dump(mode="json"),
                    "race_mode": "gp_race",
                    "gp_difficulty": "expert",
                },
            }
        )


def test_manager_config_rejects_removed_engine_tuner_objective_alias() -> None:
    with pytest.raises(ValueError, match="adaptive_engine_tuner_objective"):
        ManagedRunConfig.model_validate(
            {
                **default_managed_run_config().model_dump(mode="json"),
                "vehicle": {
                    **default_managed_run_config().vehicle.model_dump(mode="json"),
                    "adaptive_engine_tuner_objective": "completion",
                },
            }
        )


def test_manager_training_bridge_projects_configured_gp_difficulties(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.race_mode = "gp_race"
    config.tracks.gp_difficulties = ("standard", "master")
    config.tracks.selected_course_ids = ("mute_city",)

    train_config = build_managed_train_app_config(
        ManagedRunConfig.model_validate(config.model_dump(mode="python")),
        run_id="bridge-gp-pool",
        run_dir=tmp_path / "runs" / "bridge-gp-pool_0001",
    )

    entries = train_config.env.track_sampling.entries
    assert len(entries) == 2
    assert {entry.mode for entry in entries} == {"gp_race"}
    assert {entry.gp_difficulty for entry in entries} == {"standard", "master"}
    assert {entry.source_gp_difficulty for entry in entries} == {"standard", "master"}
    assert {entry.course_id for entry in entries} == {"mute_city"}


def test_manager_training_bridge_supports_weight_fork_from_parent_run(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config()
    run_dir = tmp_path / "runs" / "bridge-child"
    source_run_dir = tmp_path / "runs" / "bridge-parent"

    train_config = build_managed_fork_train_app_config(
        config,
        run_id="bridge-child",
        run_dir=run_dir,
        source_run_dir=source_run_dir,
        source_artifact="best",
        source_config=config,
        tensorboard_step_offset=816_040,
    )

    assert train_config.train.explicit_run_dir == run_dir
    assert train_config.train.continue_run_dir is None
    assert train_config.train.resume_run_dir == source_run_dir
    assert train_config.train.resume_source_algorithm == train_config.train.algorithm
    assert (
        train_config.train.resume_source_auxiliary_state_enabled
        == config.policy.auxiliary_state_enabled
    )
    assert train_config.train.resume_source_auxiliary_state_head_arch == (
        config.policy.auxiliary_state_head_arch
    )
    assert train_config.train.resume_source_metadata_required is True
    assert train_config.train.resume_artifact == "best"
    assert train_config.train.resume_mode == "weights_only"
    assert train_config.train.tensorboard_step_offset == 816_040


def test_manager_training_bridge_preserves_tensorboard_offset_for_full_resume(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config()
    run_dir = tmp_path / "runs" / "bridge-resume"

    train_config = build_managed_resume_train_app_config(
        config,
        run_id="bridge-resume",
        run_dir=run_dir,
        tensorboard_step_offset=816_040,
    )

    assert train_config.train.explicit_run_dir == run_dir
    assert train_config.train.continue_run_dir == run_dir
    assert train_config.train.resume_run_dir == run_dir
    assert train_config.train.resume_source_algorithm == train_config.train.algorithm
    assert (
        train_config.train.resume_source_auxiliary_state_enabled
        == config.policy.auxiliary_state_enabled
    )
    assert train_config.train.resume_source_auxiliary_state_head_arch == (
        config.policy.auxiliary_state_head_arch
    )
    assert train_config.train.resume_source_metadata_required is True
    assert train_config.train.resume_mode == "full_model"
    assert train_config.train.tensorboard_step_offset == 816_040


def test_fork_compatibility_allows_nonstructural_observation_and_policy_edits() -> None:
    source_config = default_managed_run_config()
    candidate_config = source_config.model_copy(deep=True)
    candidate_config.observation.resize_filter = "nearest"
    candidate_config.observation.minimap_resize_filter = "bilinear"
    candidate_config.policy.activation = "gelu"
    candidate_config.policy.image_projection_activation = "gelu"
    candidate_config.policy.state_activation = "gelu"
    candidate_config.policy.fusion_activation = "tanh"
    candidate_config.policy.layer_norm_activation = "gelu"

    assert_managed_fork_compatible(source_config, candidate_config)


def test_fork_compatibility_rejects_pitch_output_shape_changes() -> None:
    source_config = default_managed_run_config()
    candidate_config = source_config.model_copy(deep=True)
    candidate_config.action.pitch_mode = "discrete"
    candidate_config.action.pitch_buckets = 9

    with pytest.raises(ValueError, match="action layout"):
        assert_managed_fork_compatible(source_config, candidate_config)


def test_fork_compatibility_rejects_cnn_architecture_changes() -> None:
    source_config = default_managed_run_config()
    candidate_config = source_config.model_copy(deep=True)
    candidate_config.policy.features_dim = 640

    with pytest.raises(ValueError, match="policy architecture"):
        assert_managed_fork_compatible(source_config, candidate_config)
