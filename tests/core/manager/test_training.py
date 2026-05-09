# tests/core/manager/test_training.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.domain.observation_image import ObservationCustomResolution
from rl_fzerox.core.envs.observations.state import state_feature_names
from rl_fzerox.core.manager import default_managed_run_config
from rl_fzerox.core.manager.training import (
    assert_managed_fork_compatible,
    build_managed_fork_train_app_config,
    build_managed_resume_train_app_config,
    build_managed_train_app_config,
)


def test_default_manager_training_bridge_uses_configured_hybrid_defaults(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config()

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-default",
        run_dir=tmp_path / "runs" / "bridge-default_0001",
    )

    assert train_config.train.algorithm == "maskable_hybrid_recurrent_ppo"
    assert train_config.reward.name == "reward_main"
    assert train_config.reward.suspend_progress_while_outside_track_bounds is True
    assert train_config.train.explicit_run_dir == tmp_path / "runs" / "bridge-default_0001"
    assert train_config.emulator.renderer == "gliden64"
    assert train_config.env.action.runtime().name == "configured_hybrid"
    assert train_config.env.action.mask_air_brake_on_ground is False
    assert train_config.env.action.layout_continuous_axes == ("steer",)
    assert train_config.env.action.layout_discrete_axes == (
        "gas",
        "air_brake",
        "boost",
        "lean",
        "pitch",
    )
    assert len(train_config.env.track_sampling.entries) == len(config.tracks.selected_course_ids)


def test_manager_training_bridge_supports_discrete_and_continuous_mixed_actions(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.action.steering_mode = "discrete"
    config.action.drive_mode = "pwm"
    config.action.force_full_throttle = True
    config.action.include_air_brake = False
    config.action.include_boost = False
    config.action.include_lean = True
    config.action.lean_output_mode = "independent_buttons"
    config.action.include_pitch = True
    config.action.pitch_mode = "continuous"

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-mixed",
        run_dir=tmp_path / "runs" / "bridge-mixed_0001",
    )

    assert train_config.train.algorithm == "maskable_hybrid_recurrent_ppo"
    assert train_config.env.action.runtime().name == "configured_hybrid"
    assert train_config.env.action.force_full_throttle is True
    assert train_config.env.action.layout_continuous_axes == ("drive", "pitch")
    assert train_config.env.action.layout_discrete_axes == ("steer", "lean")
    assert train_config.env.action.independent_lean_buttons is True


def test_manager_training_bridge_can_mask_air_brake_on_ground(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.action.mask_air_brake_on_ground = True

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-air-brake-ground-mask",
        run_dir=tmp_path / "runs" / "bridge-air-brake-ground-mask_0001",
    )

    assert train_config.env.action.mask_air_brake_on_ground is True


def test_manager_training_bridge_supports_continuous_air_brake_lane(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.action.air_brake_mode = "pwm"
    config.action.enable_air_brake = False
    config.action.continuous_air_brake_deadzone = 0.2
    config.action.continuous_air_brake_full_threshold = 0.9
    config.action.continuous_air_brake_min_duty = 0.35

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-air-brake-pwm",
        run_dir=tmp_path / "runs" / "bridge-air-brake-pwm_0001",
    )

    assert train_config.env.action.runtime().name == "configured_hybrid"
    assert train_config.env.action.layout_continuous_axes == ("steer", "air_brake")
    assert train_config.env.action.layout_discrete_axes == ("gas", "boost", "lean", "pitch")
    assert train_config.env.action.continuous_air_brake_deadzone == pytest.approx(0.2)
    assert train_config.env.action.continuous_air_brake_full_threshold == pytest.approx(0.9)
    assert train_config.env.action.continuous_air_brake_min_duty == pytest.approx(0.35)
    assert train_config.env.action.continuous_air_brake_mode == "off"


def test_manager_training_bridge_can_override_renderer(tmp_path: Path) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.environment.renderer = "angrylion"

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-renderer",
        run_dir=tmp_path / "runs" / "bridge-renderer_0001",
    )

    assert train_config.emulator.renderer == "angrylion"


def test_manager_training_bridge_uses_explicit_state_component_membership(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.observation.state_components = tuple(
        component
        for component in config.observation.state_components
        if component.name != "machine_context"
    )
    config.policy.state_net_arch = ()

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-component-membership",
        run_dir=tmp_path / "runs" / "bridge-component-membership_0001",
    )

    assert "machine_context" not in {
        component.name for component in train_config.env.observation.state_components or ()
    }
    assert train_config.policy.extractor.resolved_state_net_arch() == ()


def test_manager_training_bridge_projects_individual_state_features(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.observation.state_components = tuple(
        component.model_copy(update={"included_features": ("track_position.edge_ratio",)})
        if component.name == "track_position"
        else component
        for component in config.observation.state_components
        if component.name == "track_position"
    )
    config.observation.state_feature_dropouts = ()

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-feature-membership",
        run_dir=tmp_path / "runs" / "bridge-feature-membership_0001",
    )
    components = train_config.env.observation.state_components
    assert components is not None

    assert state_feature_names(
        state_components=tuple(component.data() for component in components)
    ) == ("track_position.edge_ratio",)


def test_manager_training_bridge_supports_episode_state_feature_dropout(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.observation.state_feature_dropouts = (
        config.observation.state_feature_dropouts[0].model_copy(update={"dropout_prob": 0.25}),
    )

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-state-feature-dropout",
        run_dir=tmp_path / "runs" / "bridge-state-feature-dropout_0001",
    )

    assert tuple(
        group.model_dump(mode="python") for group in train_config.train.state_feature_dropout_groups
    ) == ({"dropout_prob": 0.25, "feature_names": ("track_position.edge_ratio",)},)


def test_manager_training_bridge_projects_custom_observation_resolution(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.observation.resolution_mode = "custom"
    config.observation.custom_resolution = ObservationCustomResolution(height=72, width=96)

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-custom-resolution",
        run_dir=tmp_path / "runs" / "bridge-custom-resolution_0001",
    )

    assert train_config.env.observation.resolution_mode == "custom"
    assert train_config.env.observation.custom_resolution is not None
    assert train_config.env.observation.custom_resolution.height == 72
    assert train_config.env.observation.custom_resolution.width == 96


def test_manager_training_bridge_supports_nature_conv_profile_for_custom_resolution() -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.observation.resolution_mode = "custom"
    config.observation.custom_resolution = ObservationCustomResolution(height=72, width=96)

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-custom-resolution-nature-conv",
        run_dir=Path("unused"),
    )

    assert train_config.policy.extractor.conv_profile == "nature"


def test_manager_training_bridge_supports_multilayer_state_mlp(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.policy.state_net_arch = (128, 64)

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-state-mlp",
        run_dir=tmp_path / "runs" / "bridge-state-mlp_0001",
    )

    assert train_config.policy.extractor.resolved_state_net_arch() == (128, 64)


def test_manager_training_bridge_switches_to_discrete_ppo_when_no_continuous_axes(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.policy.recurrent_enabled = False
    config.action.steering_mode = "discrete"
    config.action.drive_mode = "on_off"

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-discrete",
        run_dir=tmp_path / "runs" / "bridge-discrete_0001",
    )

    assert train_config.train.algorithm == "maskable_ppo"
    assert train_config.env.action.runtime().name == "configured_discrete"
    assert train_config.env.action.layout_continuous_axes == ()
    assert train_config.env.action.layout_discrete_axes == (
        "steer",
        "gas",
        "air_brake",
        "boost",
        "lean",
        "pitch",
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


def test_manager_config_omits_gp_difficulty_outside_gp_race() -> None:
    config = default_managed_run_config()

    dumped = config.model_dump(mode="json")

    assert dumped["tracks"]["race_mode"] == "time_attack"
    assert "gp_difficulty" not in dumped["tracks"]


def test_manager_training_bridge_projects_configured_gp_difficulty(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.race_mode = "gp_race"
    config.tracks.gp_difficulty = "master"
    config.tracks.selected_course_ids = ("mute_city",)

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-gp-master",
        run_dir=tmp_path / "runs" / "bridge-gp-master_0001",
    )

    entries = train_config.env.track_sampling.entries
    assert len(entries) == 1
    assert entries[0].mode == "gp_race"
    assert entries[0].gp_difficulty == "master"
    assert entries[0].source_gp_difficulty == "master"


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
        tensorboard_step_offset=816_040,
    )

    assert train_config.train.explicit_run_dir == run_dir
    assert train_config.train.continue_run_dir is None
    assert train_config.train.resume_run_dir == source_run_dir
    assert train_config.train.resume_source_algorithm is None
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
    assert train_config.train.resume_source_algorithm is None
    assert train_config.train.resume_mode == "full_model"
    assert train_config.train.tensorboard_step_offset == 816_040


def test_fork_compatibility_allows_nonstructural_observation_and_policy_edits() -> None:
    source_config = default_managed_run_config()
    candidate_config = source_config.model_copy(deep=True)
    candidate_config.observation.resize_filter = "nearest"
    candidate_config.observation.minimap_resize_filter = "bilinear"
    candidate_config.policy.activation = "gelu"

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
