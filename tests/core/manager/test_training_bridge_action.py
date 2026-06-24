# tests/core/manager/test_training_bridge_action.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.envs.observations.state import state_feature_names
from rl_fzerox.core.manager import ManagedRunConfig, default_managed_run_config
from rl_fzerox.core.manager.training import (
    build_managed_train_app_config,
)
from tests.core.manager.manager_training_support import (
    _manager_config_data_with_control_history_features,
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

    assert train_config.env.track_sampling.sampling_mode == "equal"


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
    config.tracks.gp_difficulties = ("novice", "expert")
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
    config.train.entropy_coefficients = {
        "drive": 0.015,
        "ghost": 0.02,
        "pitch": 0.0025,
    }
    config.train.actor_regularization.grounded_pitch_neutral_loss_weight = 0.02
    config.train.actor_regularization.pitch_std_cap_loss_weight = 0.07
    config.train.actor_regularization.grounded_pitch_std_cap = 0.35
    config.train.actor_regularization.airborne_pitch_std_cap = 0.8
    config.train.actor_regularization.steer_std_cap_loss_weight = 0.03
    config.train.actor_regularization.steer_std_cap = 0.9
    config.train.actor_regularization.steer_signed_balance_loss_weight = 0.04
    config.train.actor_regularization.steer_signed_balance_deadzone = 0.2
    config.train.actor_regularization.lean_signed_balance_loss_weight = 0.05
    config.train.actor_regularization.lean_signed_balance_deadzone = 0.1

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-action-loss-controls",
        run_dir=tmp_path / "runs" / "bridge-action-loss-controls_0001",
    )

    assert train_config.train.ent_coef == pytest.approx(1.0)
    assert train_config.train.entropy_group_weights == {
        "drive": pytest.approx(0.015),
        "pitch": pytest.approx(0.0025),
        "steer": pytest.approx(0.01),
    }
    assert (
        train_config.train.actor_regularization.grounded_pitch_neutral_loss_weight
        == pytest.approx(0.02)
    )
    assert train_config.train.actor_regularization.pitch_std_cap_loss_weight == pytest.approx(0.07)
    assert train_config.train.actor_regularization.grounded_pitch_std_cap == pytest.approx(0.35)
    assert train_config.train.actor_regularization.airborne_pitch_std_cap == pytest.approx(0.8)
    assert train_config.train.actor_regularization.steer_std_cap_loss_weight == 0.0
    assert train_config.train.actor_regularization.steer_std_cap == pytest.approx(0.9)
    assert train_config.train.actor_regularization.steer_signed_balance_loss_weight == 0.0
    assert train_config.train.actor_regularization.steer_signed_balance_deadzone == pytest.approx(
        0.2
    )
    assert train_config.train.actor_regularization.lean_signed_balance_loss_weight == 0.0
    assert train_config.train.actor_regularization.lean_signed_balance_deadzone == pytest.approx(
        0.1
    )

    config.action.steering_mode = "continuous"
    config.action.include_lean = True
    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-action-loss-controls-continuous-steer",
        run_dir=tmp_path / "runs" / "bridge-action-loss-controls-continuous-steer_0001",
    )

    assert train_config.train.actor_regularization.steer_std_cap_loss_weight == pytest.approx(0.03)
    assert (
        train_config.train.actor_regularization.steer_signed_balance_loss_weight
        == pytest.approx(0.04)
    )
    assert train_config.train.actor_regularization.lean_signed_balance_loss_weight == pytest.approx(
        0.05
    )

    config.action.steering_mode = "discrete"
    config.action.pitch_mode = "discrete"
    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-action-loss-controls-discrete",
        run_dir=tmp_path / "runs" / "bridge-action-loss-controls-discrete_0001",
    )

    assert (
        train_config.train.actor_regularization.grounded_pitch_neutral_loss_weight
        == pytest.approx(0.02)
    )
    assert train_config.train.actor_regularization.pitch_std_cap_loss_weight == pytest.approx(0.07)

    config.action.include_pitch = False
    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-action-loss-controls-no-pitch",
        run_dir=tmp_path / "runs" / "bridge-action-loss-controls-no-pitch_0001",
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
    config.action.spin_episode_mask_probability = 0.25
    config.policy.spin_idle_logit = 1.0

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
    assert train_config.env.action.spin_episode_mask_probability == pytest.approx(0.25)
    assert train_config.policy.action_bias.spin_idle_logit == pytest.approx(1.0)


def test_manager_training_bridge_supports_discrete_air_brake_bias(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.action.include_air_brake = True
    config.action.air_brake_mode = "on_off"
    config.action.air_brake_episode_mask_probability = 0.5
    config.action.air_brake_pulse_frames = 12
    config.policy.air_brake_on_logit = 2.0

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-air-brake-bias",
        run_dir=tmp_path / "runs" / "bridge-air-brake-bias_0001",
    )

    assert "air_brake" in train_config.env.action.layout_discrete_axes
    assert train_config.env.action.air_brake_episode_mask_probability == pytest.approx(0.5)
    assert train_config.env.action.air_brake_pulse_frames == 12
    assert train_config.policy.action_bias.air_brake_on_logit == pytest.approx(2.0)


def test_manager_training_bridge_ignores_air_brake_bias_without_discrete_branch(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.action.include_air_brake = True
    config.action.air_brake_mode = "pwm"
    config.action.air_brake_episode_mask_probability = 0.5
    config.action.air_brake_pulse_frames = 12
    config.policy.air_brake_on_logit = 2.0

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-air-brake-pwm-bias",
        run_dir=tmp_path / "runs" / "bridge-air-brake-pwm-bias_0001",
    )

    assert "air_brake" not in train_config.env.action.layout_discrete_axes
    assert train_config.env.action.air_brake_episode_mask_probability == pytest.approx(0.0)
    assert train_config.env.action.air_brake_pulse_frames == 0
    assert train_config.policy.action_bias.air_brake_on_logit == pytest.approx(0.0)


def test_manager_training_bridge_ignores_spin_idle_bias_without_spin_branch(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.action.include_spin = False
    config.action.spin_episode_mask_probability = 0.25
    config.policy.spin_idle_logit = 1.0

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-no-spin",
        run_dir=tmp_path / "runs" / "bridge-no-spin_0001",
    )

    assert "spin" not in train_config.env.action.layout_discrete_axes
    assert train_config.env.action.spin_episode_mask_probability == pytest.approx(0.0)
    assert train_config.policy.action_bias.spin_idle_logit == pytest.approx(0.0)


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
    config.reward.outside_track_dip_penalty = -0.125
    config.reward.outside_track_dip_height_threshold = -8.0
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
    assert train_config.reward.outside_track_dip_penalty == pytest.approx(-0.125)
    assert train_config.reward.outside_track_dip_height_threshold == pytest.approx(-8.0)
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


def test_manager_training_bridge_projects_step_balanced_track_sampling_settings(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.sampling_mode = "step_balanced"
    config.tracks.step_balance_update_episodes = 7
    config.tracks.step_balance_ema_alpha = 0.2
    config.tracks.step_balance_max_weight_scale = 3.5

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-step-balanced-track-sampling",
        run_dir=tmp_path / "runs" / "bridge-step-balanced-track-sampling_0001",
    )

    assert train_config.env.track_sampling.sampling_mode == "step_balanced"
    assert train_config.env.track_sampling.step_balance_update_episodes == 7
    assert train_config.env.track_sampling.step_balance_ema_alpha == pytest.approx(0.2)
    assert train_config.env.track_sampling.step_balance_max_weight_scale == pytest.approx(3.5)


def test_manager_training_bridge_projects_deficit_budget_track_sampling_settings(
    tmp_path: Path,
) -> None:
    config = default_managed_run_config().model_copy(deep=True)
    config.tracks.sampling_mode = "deficit_budget"
    config.tracks.deficit_budget_uniform_fraction = 0.6
    config.tracks.deficit_budget_focus_sharpness = 2.0
    config.tracks.deficit_budget_ema_alpha = 0.01
    config.tracks.deficit_budget_weight_update_rollouts = 30
    config.tracks.deficit_budget_difficulty_metric = "finish_ema"
    config.tracks.deficit_budget_warmup_min_episodes_per_course = 12
    config.tracks.deficit_budget_uniform_staleness_rotations = 1.5

    train_config = build_managed_train_app_config(
        config,
        run_id="bridge-deficit-track-sampling",
        run_dir=tmp_path / "runs" / "bridge-deficit-track-sampling_0001",
    )

    assert train_config.env.track_sampling.sampling_mode == "deficit_budget"
    assert train_config.env.track_sampling.deficit_budget_uniform_fraction == pytest.approx(0.6)
    assert train_config.env.track_sampling.deficit_budget_focus_sharpness == pytest.approx(2.0)
    assert train_config.env.track_sampling.deficit_budget_ema_alpha == pytest.approx(0.01)
    assert train_config.env.track_sampling.deficit_budget_weight_update_rollouts == 30
    assert train_config.env.track_sampling.deficit_budget_difficulty_metric == "finish_ema"
    assert train_config.env.track_sampling.deficit_budget_warmup_min_episodes_per_course == 12
    assert train_config.env.track_sampling.deficit_budget_uniform_staleness_rotations == (
        pytest.approx(1.5)
    )


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
