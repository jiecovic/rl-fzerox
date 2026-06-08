# tests/core/manager/test_training.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from rl_fzerox.core.domain.observation_image import (
    CustomResolutionChoice,
    SourceCropResolutionChoice,
)
from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.envs.observations.state import state_feature_names
from rl_fzerox.core.manager import ManagedRunConfig, default_managed_run_config
from rl_fzerox.core.manager.architecture.preview import policy_architecture_preview
from rl_fzerox.core.manager.training import (
    assert_managed_fork_compatible,
    build_managed_fork_train_app_config,
    build_managed_resume_train_app_config,
    build_managed_train_app_config,
)


def _manager_config_data_with_control_history_features(
    included_features: tuple[str, ...],
    *,
    lean_output_mode: Literal["three_way", "four_way_categorical", "independent_buttons"],
) -> dict[str, object]:
    return {
        "action": {"lean_output_mode": lean_output_mode},
        "observation": {
            "state_components": [
                {
                    "name": "control_history",
                    "length": 1,
                    "controls": ["lean"],
                    "included_features": list(included_features),
                }
            ],
            "state_feature_dropouts": [],
        },
    }


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
    assert train_config.reward.position_progress_min_multiplier == pytest.approx(1.0)
    assert train_config.reward.position_progress_max_multiplier == pytest.approx(1.0)
    assert train_config.reward.spin_request_penalty == pytest.approx(0.0)
    assert train_config.train.explicit_run_dir == tmp_path / "runs" / "bridge-default_0001"
    assert train_config.emulator.renderer == "gliden64"
    assert train_config.env.camera_setting == "close_behind"
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


def test_manager_training_bridge_projects_equal_sampling_to_balanced_cycle(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.sampling_mode = "equal"

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-equal-track-sampling",
        run_dir=tmp_path / "runs" / "bridge-equal-track-sampling_0001",
    )

    assert train_config.env.track_sampling.sampling_mode == "balanced"


def test_manager_training_bridge_projects_fixed_env_sampling(tmp_path: Path) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.train.num_envs = len(config.tracks.selected_course_ids)
    config.tracks.sampling_mode = "fixed_env"

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-fixed-env-track-sampling",
        run_dir=tmp_path / "runs" / "bridge-fixed-env-track-sampling_0001",
    )

    assert train_config.env.track_sampling.sampling_mode == "fixed_env"


def test_fixed_env_sampling_allows_env_multiples() -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.train.num_envs = 24
    config.tracks.sampling_mode = "fixed_env"
    config.tracks.selected_course_ids = config.tracks.selected_course_ids[:12]

    validated = ManagedRunConfig.model_validate(config.model_dump(mode="python"))

    assert validated.tracks.active_course_count() == 12


def test_fixed_env_sampling_rejects_non_divisible_env_counts() -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.train.num_envs = 24
    config.tracks.sampling_mode = "fixed_env"
    config.tracks.selected_course_ids = config.tracks.selected_course_ids[:10]

    with pytest.raises(ValueError, match="fixed_env"):
        ManagedRunConfig.model_validate(config.model_dump(mode="python"))


def test_fixed_env_sampling_rejects_too_few_envs() -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.train.num_envs = 12
    config.tracks.sampling_mode = "fixed_env"

    with pytest.raises(ValueError, match="at least the active course count"):
        ManagedRunConfig.model_validate(config.model_dump(mode="python"))


def test_fixed_env_sampling_counts_x_cup_slots() -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.train.num_envs = 24
    config.tracks.race_mode = "gp_race"
    config.tracks.include_x_cup = True
    config.tracks.x_cup_course_count = 2
    config.tracks.sampling_mode = "fixed_env"
    config.tracks.selected_course_ids = config.tracks.selected_course_ids[:22]

    validated = ManagedRunConfig.model_validate(config.model_dump(mode="python"))

    assert validated.tracks.active_course_count() == 24


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
    assert train_config.env.action.layout_discrete_axes == (
        "steer",
        "lean_left",
        "lean_right",
    )
    assert train_config.env.action.lean_output_mode == "independent_buttons"
    assert train_config.env.action.runtime().split_lean_action_branches is True


def test_manager_training_bridge_projects_action_entropy_and_actor_loss(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.action.steering_mode = "discrete"
    config.action.drive_mode = "pwm"
    config.action.force_full_throttle = True
    config.action.include_air_brake = False
    config.action.include_boost = False
    config.action.include_lean = False
    config.action.include_pitch = True
    config.action.pitch_mode = "continuous"
    config.train.entropy_group_weights = {
        "drive": 1.5,
        "ghost": 2.0,
        "pitch": 0.25,
    }
    config.train.actor_regularization.grounded_pitch_neutral_loss_weight = 0.02
    config.train.actor_regularization.pitch_std_cap_loss_weight = 0.05
    config.train.actor_regularization.pitch_std_cap = 0.5

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-action-loss-controls",
        run_dir=tmp_path / "runs" / "bridge-action-loss-controls_0001",
    )

    assert train_config.train.entropy_group_weights == {
        "drive": pytest.approx(1.5),
        "pitch": pytest.approx(0.25),
    }
    assert (
        train_config.train.actor_regularization.grounded_pitch_neutral_loss_weight
        == pytest.approx(0.02)
    )
    assert train_config.train.actor_regularization.pitch_std_cap_loss_weight == pytest.approx(0.05)
    assert train_config.train.actor_regularization.pitch_std_cap == pytest.approx(0.5)

    config.action.pitch_mode = "discrete"
    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-action-loss-controls-discrete",
        run_dir=tmp_path / "runs" / "bridge-action-loss-controls-discrete_0001",
    )

    assert train_config.train.actor_regularization.grounded_pitch_neutral_loss_weight == 0.0
    assert train_config.train.actor_regularization.pitch_std_cap_loss_weight == 0.0


def test_manager_training_bridge_supports_four_way_categorical_lean(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.action.lean_output_mode = "four_way_categorical"

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-four-way-lean",
        run_dir=tmp_path / "runs" / "bridge-four-way-lean_0001",
    )

    assert train_config.env.action.layout_discrete_axes == (
        "gas",
        "air_brake",
        "boost",
        "lean",
        "pitch",
    )
    assert train_config.env.action.lean_output_mode == "four_way_categorical"
    assert train_config.env.action.runtime().split_lean_history is True


def test_manager_training_bridge_supports_optional_spin_macro(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.action.include_spin = True
    config.action.spin_cooldown_frames = 13

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-spin",
        run_dir=tmp_path / "runs" / "bridge-spin_0001",
    )

    assert train_config.env.action.layout_discrete_axes == (
        "gas",
        "air_brake",
        "boost",
        "lean",
        "spin",
        "pitch",
    )
    assert train_config.env.action.spin_cooldown_frames == 13


def test_manager_action_config_disables_spin_outside_three_way_lean() -> None:
    config = ManagedRunConfig.model_validate(
        {
            "action": {
                "lean_output_mode": "four_way_categorical",
                "include_spin": True,
                "enable_spin": True,
            }
        }
    )

    assert config.action.include_spin is False
    assert config.action.enable_spin is False


def test_manager_run_config_accepts_independent_lean_history_features(
    tmp_path: Path,
) -> None:
    config_data = _manager_config_data_with_control_history_features(
        (
            "control_history.prev_lean_left_1",
            "control_history.prev_lean_right_1",
        ),
        lean_output_mode="independent_buttons",
    )

    config = ManagedRunConfig.model_validate(config_data)
    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-independent-lean-history",
        run_dir=tmp_path / "runs" / "bridge-independent-lean-history_0001",
    )
    components = train_config.env.observation.state_components
    assert components is not None

    assert state_feature_names(
        state_components=tuple(component.data() for component in components),
        split_lean_history=train_config.env.action.runtime().split_lean_history,
    ) == (
        "control_history.prev_lean_left_1",
        "control_history.prev_lean_right_1",
    )


def test_manager_run_config_rejects_split_lean_history_for_three_way_lean() -> None:
    config_data = _manager_config_data_with_control_history_features(
        ("control_history.prev_lean_left_1",),
        lean_output_mode="three_way",
    )

    with pytest.raises(ValueError, match="control_history.prev_lean_left_1"):
        ManagedRunConfig.model_validate(config_data)


def test_manager_training_bridge_can_mask_air_brake_on_ground(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.action.mask_air_brake_on_ground = True
    config.action.mask_boost_when_airborne = False
    config.action.boost_decision_interval_steps = 4

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-air-brake-ground-mask",
        run_dir=tmp_path / "runs" / "bridge-air-brake-ground-mask_0001",
    )

    assert train_config.env.action.mask_air_brake_on_ground is True
    assert train_config.env.action.mask_boost_when_airborne is False
    assert train_config.env.action.boost_decision_interval_frames == 8


def test_manager_training_bridge_projects_outside_track_recovery_reward(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.reward.outside_track_recovery_reward = 0.0025
    config.reward.outside_track_recovery_reward_cap = 0.075
    config.reward.outside_track_recovery_airborne_grace_frames = 45
    config.reward.airborne_landing_grace_frames = 60
    config.reward.position_progress_min_multiplier = 0.95
    config.reward.position_progress_max_multiplier = 1.15

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-outside-track-recovery",
        run_dir=tmp_path / "runs" / "bridge-outside-track-recovery_0001",
    )

    assert train_config.reward.outside_track_recovery_reward == pytest.approx(0.0025)
    assert train_config.reward.outside_track_recovery_reward_cap == pytest.approx(0.075)
    assert train_config.reward.outside_track_recovery_airborne_grace_frames == 45
    assert train_config.reward.airborne_landing_grace_frames == 60
    assert train_config.reward.position_progress_min_multiplier == pytest.approx(0.95)
    assert train_config.reward.position_progress_max_multiplier == pytest.approx(1.15)


def test_manager_training_bridge_projects_boost_reward_energy_shaping(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.reward.manual_boost_reward = 7.0
    config.reward.manual_boost_reward_energy_shaping = True
    config.reward.manual_boost_reward_min_energy_fraction = 0.15
    config.reward.manual_boost_reward_min_energy_value = -1.4
    config.reward.manual_boost_reward_full_energy_fraction = 0.75
    config.reward.manual_boost_reward_energy_curve = "smoothstep"

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-boost-reward-energy-shaping",
        run_dir=tmp_path / "runs" / "bridge-boost-reward-energy-shaping_0001",
    )

    assert train_config.reward.manual_boost_reward == pytest.approx(7.0)
    assert train_config.reward.manual_boost_reward_energy_shaping is True
    assert train_config.reward.manual_boost_reward_min_energy_fraction == pytest.approx(0.15)
    assert train_config.reward.manual_boost_reward_min_energy_value == pytest.approx(-1.4)
    assert train_config.reward.manual_boost_reward_full_energy_fraction == pytest.approx(0.75)
    assert train_config.reward.manual_boost_reward_energy_curve == "smoothstep"


def test_manager_training_bridge_projects_adaptive_track_sampling_settings(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.sampling_mode = "adaptive_step_balanced"
    config.tracks.step_balance_update_episodes = 7
    config.tracks.step_balance_ema_alpha = 0.2
    config.tracks.step_balance_max_weight_scale = 3.5
    config.tracks.adaptive_step_balance_completion_weight = 0.45
    config.tracks.adaptive_step_balance_target_completion = 0.85
    config.tracks.adaptive_step_balance_min_confidence_episodes = 32
    config.tracks.adaptive_step_balance_confidence_scale = 2.5

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-adaptive-track-sampling",
        run_dir=tmp_path / "runs" / "bridge-adaptive-track-sampling_0001",
    )

    assert train_config.env.track_sampling.sampling_mode == "adaptive_step_balanced"
    assert train_config.env.track_sampling.step_balance_update_episodes == 7
    assert train_config.env.track_sampling.step_balance_ema_alpha == pytest.approx(0.2)
    assert train_config.env.track_sampling.step_balance_max_weight_scale == pytest.approx(3.5)
    assert train_config.env.track_sampling.adaptive_step_balance_completion_weight == (
        pytest.approx(0.45)
    )
    assert train_config.env.track_sampling.adaptive_step_balance_target_completion == (
        pytest.approx(0.85)
    )
    assert train_config.env.track_sampling.adaptive_step_balance_min_confidence_episodes == 32
    assert train_config.env.track_sampling.adaptive_step_balance_confidence_scale == (
        pytest.approx(2.5)
    )


def test_manager_training_bridge_projects_deficit_budget_track_sampling_settings(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.sampling_mode = "deficit_budget"
    config.tracks.deficit_budget_uniform_fraction = 0.6
    config.tracks.deficit_budget_min_weight = 1.2
    config.tracks.deficit_budget_max_weight = 4.0
    config.tracks.deficit_budget_ema_alpha = 0.01
    config.tracks.deficit_budget_weight_update_rollouts = 30

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-deficit-track-sampling",
        run_dir=tmp_path / "runs" / "bridge-deficit-track-sampling_0001",
    )

    assert train_config.env.track_sampling.sampling_mode == "deficit_budget"
    assert train_config.env.track_sampling.deficit_budget_uniform_fraction == pytest.approx(0.6)
    assert train_config.env.track_sampling.deficit_budget_min_weight == pytest.approx(1.2)
    assert train_config.env.track_sampling.deficit_budget_max_weight == pytest.approx(4.0)
    assert train_config.env.track_sampling.deficit_budget_ema_alpha == pytest.approx(0.01)
    assert train_config.env.track_sampling.deficit_budget_weight_update_rollouts == 30


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
    config.environment.camera_setting = "regular"

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-renderer",
        run_dir=tmp_path / "runs" / "bridge-renderer_0001",
    )

    assert train_config.emulator.renderer == "angrylion"
    assert train_config.env.camera_setting == "regular"


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


def test_manager_training_bridge_can_disable_fusion_mlp(tmp_path: Path) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.policy.fusion_features_dim = None

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-no-fusion",
        run_dir=tmp_path / "runs" / "bridge-no-fusion_0001",
    )
    preview = policy_architecture_preview(config)
    fusion_group = next(group for group in preview.parameter_groups if group.name == "Fusion")
    fusion_node = next(
        node for lane in preview.architecture_lanes for node in lane.nodes if node.id == "fusion"
    )

    assert train_config.policy.extractor.fusion_features_dim is None
    assert preview.extractor_output_dim == preview.fusion_input_dim
    assert fusion_group.params == 0
    assert fusion_node.detail == f"identity {preview.fusion_input_dim}"
    assert fusion_node.tone == "muted"


def test_policy_architecture_preview_labels_extractor_activations() -> None:
    config = default_managed_run_config()
    preview = policy_architecture_preview(config)
    node_by_id = {node.id: node for lane in preview.architecture_lanes for node in lane.nodes}

    assert node_by_id["cnn"].detail == "nature → 3136"
    assert node_by_id["image_projection"].detail == "identity 3136"
    assert node_by_id["state_mlp"].detail.endswith(", relu")
    assert node_by_id["fusion"].detail.endswith(", relu")
    assert node_by_id["policy_head"].detail.endswith(", relu")


def test_manager_training_bridge_projects_extractor_activations(tmp_path: Path) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.policy.features_dim = 512
    config.policy.image_projection_activation = "gelu"
    config.policy.state_activation = "tanh"
    config.policy.fusion_activation = "tanh"
    config.policy.layer_norm_activation = "gelu"

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-activations",
        run_dir=tmp_path / "runs" / "bridge-activations",
    )
    preview = policy_architecture_preview(config)
    node_by_id = {node.id: node for lane in preview.architecture_lanes for node in lane.nodes}

    assert train_config.policy.extractor.image_projection_activation == "gelu"
    assert train_config.policy.extractor.state_activation == "tanh"
    assert train_config.policy.extractor.fusion_activation == "tanh"
    assert train_config.policy.extractor.layer_norm_activation == "gelu"
    assert node_by_id["image_projection"].detail.endswith(", gelu")
    assert node_by_id["state_mlp"].detail.endswith(", tanh")
    assert node_by_id["fusion"].detail.endswith(", tanh")
    assert node_by_id["layer_norm"].detail == "on, gelu"


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
    config.observation.resolution = CustomResolutionChoice(height=72, width=96)

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-custom-resolution",
        run_dir=tmp_path / "runs" / "bridge-custom-resolution_0001",
    )

    assert train_config.env.observation.resolution == CustomResolutionChoice(height=72, width=96)


def test_manager_training_bridge_projects_source_crop_observation_resolution(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.environment.renderer = "angrylion"
    config.observation.resolution = SourceCropResolutionChoice()

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-source-crop-resolution",
        run_dir=tmp_path / "runs" / "bridge-source-crop-resolution_0001",
    )

    assert train_config.env.observation.resolution == SourceCropResolutionChoice()
    assert train_config.env.observation.image_geometry(renderer=train_config.emulator.renderer) == (
        208,
        592,
    )


def test_manager_training_bridge_supports_nature_conv_profile_for_custom_resolution() -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.observation.resolution = CustomResolutionChoice(height=72, width=96)

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
