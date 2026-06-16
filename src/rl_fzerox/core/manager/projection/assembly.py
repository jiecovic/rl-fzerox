# src/rl_fzerox/core/manager/projection/assembly.py
"""Assemble concrete training-schema fragments from manager config sections."""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS
from rl_fzerox.core.manager.artifacts.paths import manager_runs_root
from rl_fzerox.core.manager.projection.actions import (
    build_action_data,
    continuous_action_axes,
    discrete_action_axes,
)
from rl_fzerox.core.manager.projection.observations import (
    build_observation_data,
    build_state_feature_dropout_groups,
)
from rl_fzerox.core.manager.projection.policy import build_policy_data
from rl_fzerox.core.manager.projection.tracks import build_track_sampling_data
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.runtime_spec.paths import project_root_dir


def train_config_payload(
    config: ManagedRunConfig,
    *,
    run_id: str,
    run_dir: Path,
) -> dict[str, object]:
    return {
        "seed": config.seed,
        "emulator": emulator_data(config),
        "track": {},
        "env": env_data(config),
        "reward": reward_data(config),
        "policy": build_policy_data(config),
        "train": train_data(config, run_id=run_id, run_dir=run_dir),
    }


def env_data(config: ManagedRunConfig) -> dict[str, object]:
    return {
        "action_repeat": config.action.action_repeat,
        "max_episode_steps": config.environment.max_episode_steps,
        "stuck_min_speed_kph": 60.0,
        "progress_frontier_stall_limit_frames": (
            config.environment.progress_frontier_stall_limit_frames
        ),
        "progress_frontier_epsilon": config.environment.progress_frontier_epsilon,
        "boost_min_energy_fraction": config.action.boost_min_energy_fraction,
        "randomize_game_rng_on_reset": True,
        "randomize_game_rng_requires_race_mode": True,
        "randomize_gp_lives_on_reset": config.environment.randomize_gp_lives_on_reset,
        "gp_lives_jitter_min": config.environment.gp_lives_jitter_min,
        "gp_lives_jitter_max": config.environment.gp_lives_jitter_max,
        "camera_setting": config.environment.camera_setting,
        "reset_to_race": True,
        "race_intro_target_timer": 39,
        "cache_track_baselines": True,
        "track_sampling": build_track_sampling_data(config),
        "action": build_action_data(config),
        "observation": build_observation_data(config),
    }


def emulator_data(config: ManagedRunConfig) -> dict[str, object]:
    return {
        "core_path": default_core_path(),
        "rom_path": default_rom_path(),
        "renderer": config.environment.renderer,
    }


def reward_data(config: ManagedRunConfig) -> dict[str, object]:
    reward = config.reward
    return {
        "name": "reward_main",
        "time_penalty_per_frame": reward.time_penalty_per_frame,
        "progress_bucket_distance": reward.progress_bucket_distance,
        "progress_bucket_reward": reward.progress_bucket_reward,
        "progress_reward_interval_frames": reward.progress_reward_interval_frames,
        "suspend_progress_while_outside_track_bounds": (
            reward.suspend_progress_while_outside_track_bounds
        ),
        "progress_track_distance_tolerance": reward.progress_track_distance_tolerance,
        "progress_speed_min_kph": reward.progress_speed_min_kph,
        "progress_speed_min_multiplier": reward.progress_speed_min_multiplier,
        "progress_speed_reference_kph": reward.progress_speed_reference_kph,
        "progress_speed_max_kph": reward.progress_speed_max_kph,
        "progress_speed_max_multiplier": reward.progress_speed_max_multiplier,
        "progress_speed_curve_power": reward.progress_speed_curve_power,
        "position_progress_min_multiplier": reward.position_progress_min_multiplier,
        "position_progress_max_multiplier": reward.position_progress_max_multiplier,
        "outside_track_recovery_reward": reward.outside_track_recovery_reward,
        "outside_track_recovery_reward_cap": reward.outside_track_recovery_reward_cap,
        "outside_track_recovery_airborne_grace_frames": (
            reward.outside_track_recovery_airborne_grace_frames
        ),
        "lap_completion_bonus": reward.lap_completion_bonus,
        "lap_position_scale": reward.lap_position_scale,
        "ko_star_reward": reward.ko_star_reward,
        "energy_loss_epsilon": reward.energy_loss_epsilon,
        "energy_refill_progress_multiplier": reward.energy_refill_progress_multiplier,
        "dirt_progress_multiplier": reward.dirt_progress_multiplier,
        "ice_progress_multiplier": reward.ice_progress_multiplier,
        "dirt_entry_penalty": reward.dirt_entry_penalty,
        "ice_entry_penalty": reward.ice_entry_penalty,
        "energy_refill_collision_cooldown_frames": reward.energy_refill_collision_cooldown_frames,
        "air_brake_request_penalty": reward.air_brake_request_penalty,
        "spin_request_penalty": reward.spin_request_penalty,
        "manual_boost_reward": reward.manual_boost_reward,
        "manual_boost_reward_energy_shaping": reward.manual_boost_reward_energy_shaping,
        "manual_boost_reward_min_energy_fraction": (reward.manual_boost_reward_min_energy_fraction),
        "manual_boost_reward_min_energy_value": reward.manual_boost_reward_min_energy_value,
        "manual_boost_reward_full_energy_fraction": (
            reward.manual_boost_reward_full_energy_fraction
        ),
        "manual_boost_reward_energy_curve": reward.manual_boost_reward_energy_curve,
        "boost_pad_reward_before_unlock": reward.boost_pad_reward_before_unlock,
        "boost_pad_reward_after_unlock": reward.boost_pad_reward_after_unlock,
        "boost_pad_reward_progress_window": reward.boost_pad_reward_progress_window,
        "lean_request_penalty": reward.lean_request_penalty,
        "lean_activation_penalty": reward.lean_activation_penalty,
        "grounded_pitch_penalty": reward.grounded_pitch_penalty,
        "impact_frame_penalty": reward.impact_frame_penalty,
        "energy_loss_penalty": reward.energy_loss_penalty,
        "energy_gain_reward": reward.energy_gain_reward,
        "airborne_landing_reward": reward.airborne_landing_reward,
        "airborne_landing_grace_frames": reward.airborne_landing_grace_frames,
        "airborne_landing_min_peak_height": reward.airborne_landing_min_peak_height,
        "failure_penalty": reward.failure_penalty,
        "truncation_penalty": reward.truncation_penalty,
        "step_reward_clip_min": reward.step_reward_clip_min,
        "step_reward_clip_max": reward.step_reward_clip_max,
    }


def train_data(config: ManagedRunConfig, *, run_id: str, run_dir: Path) -> dict[str, object]:
    train = config.train
    return {
        "algorithm": effective_train_algorithm(config),
        "vec_env": "subproc",
        "num_envs": train.num_envs,
        "total_timesteps": train.total_timesteps,
        "n_steps": train.n_steps,
        "n_epochs": train.n_epochs,
        "batch_size": train.batch_size,
        "learning_rate": train.learning_rate,
        "gamma": train.gamma,
        "gae_lambda": train.gae_lambda,
        "clip_range": train.clip_range,
        "clip_range_vf": train.clip_range_vf,
        "ent_coef": train.ent_coef,
        "vf_coef": train.vf_coef,
        "max_grad_norm": train.max_grad_norm,
        "normalize_advantage": train.normalize_advantage,
        "target_kl": train.target_kl,
        "entropy_group_weights": _entropy_group_weights(config),
        "actor_regularization": _actor_regularization_data(config),
        "stats_window_size": train.stats_window_size,
        "checkpoint_every_rollouts": train.checkpoint_every_rollouts,
        "save_latest_checkpoint": train.save_latest_checkpoint,
        "save_best_checkpoint": train.save_best_checkpoint,
        "save_recent_checkpoints": train.save_recent_checkpoints,
        "recent_checkpoint_limit": train.recent_checkpoint_limit,
        "state_feature_dropout_groups": build_state_feature_dropout_groups(config),
        "verbose": 0,
        "device": "cuda",
        "save_freq": 10_000,
        "output_root": manager_runs_root(output_root=run_dir.parent),
        "run_name": run_id,
        "explicit_run_dir": run_dir,
    }


def _entropy_group_weights(config: ManagedRunConfig) -> dict[str, float]:
    action_groups = set(continuous_action_axes(config)) | set(discrete_action_axes(config))
    return {
        name: float(weight)
        for name, weight in config.train.entropy_group_weights.items()
        if name in action_groups
    }


def _actor_regularization_data(config: ManagedRunConfig) -> dict[str, float]:
    actor_regularization = config.train.actor_regularization
    grounded_pitch_neutral_loss_weight = actor_regularization.grounded_pitch_neutral_loss_weight
    pitch_std_cap_loss_weight = actor_regularization.pitch_std_cap_loss_weight
    steer_std_cap_loss_weight = actor_regularization.steer_std_cap_loss_weight
    steer_signed_balance_loss_weight = actor_regularization.steer_signed_balance_loss_weight
    lean_signed_balance_loss_weight = actor_regularization.lean_signed_balance_loss_weight
    continuous_action_groups = set(continuous_action_axes(config))
    discrete_action_groups = set(discrete_action_axes(config))
    action_groups = continuous_action_groups | discrete_action_groups
    if "pitch" not in action_groups:
        grounded_pitch_neutral_loss_weight = 0.0
        pitch_std_cap_loss_weight = 0.0
    if "steer" not in continuous_action_groups:
        steer_std_cap_loss_weight = 0.0
        steer_signed_balance_loss_weight = 0.0
    if not (
        "lean" in discrete_action_groups
        or {"lean_left", "lean_right"}.issubset(discrete_action_groups)
    ):
        lean_signed_balance_loss_weight = 0.0
    return {
        "grounded_pitch_neutral_loss_weight": float(grounded_pitch_neutral_loss_weight),
        "pitch_std_cap_loss_weight": float(pitch_std_cap_loss_weight),
        "grounded_pitch_std_cap": float(actor_regularization.grounded_pitch_std_cap),
        "airborne_pitch_std_cap": float(actor_regularization.airborne_pitch_std_cap),
        "steer_std_cap_loss_weight": float(steer_std_cap_loss_weight),
        "steer_std_cap": float(actor_regularization.steer_std_cap),
        "steer_signed_balance_loss_weight": float(steer_signed_balance_loss_weight),
        "steer_signed_balance_deadzone": float(actor_regularization.steer_signed_balance_deadzone),
        "lean_signed_balance_loss_weight": float(lean_signed_balance_loss_weight),
        "lean_signed_balance_deadzone": float(actor_regularization.lean_signed_balance_deadzone),
    }


def effective_train_algorithm(config: ManagedRunConfig) -> str:
    return (
        TRAINING_ALGORITHMS.maskable_hybrid_recurrent_ppo
        if config.policy.recurrent_enabled
        else TRAINING_ALGORITHMS.maskable_hybrid_action_ppo
    )


def default_core_path() -> Path:
    return (project_root_dir() / "local" / "libretro" / "mupen64plus_next_libretro.so").resolve()


def default_rom_path() -> Path:
    return (project_root_dir() / "local" / "roms" / "F-Zero X (USA).n64").resolve()
