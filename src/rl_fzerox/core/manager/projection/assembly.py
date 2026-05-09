# src/rl_fzerox/core/manager/projection/assembly.py
"""Assemble concrete training-schema fragments from manager config sections."""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS
from rl_fzerox.core.manager.artifacts.paths import manager_runs_root
from rl_fzerox.core.manager.projection.actions import (
    build_action_data,
    continuous_action_axes,
)
from rl_fzerox.core.manager.projection.observations import (
    build_observation_data,
    build_state_feature_dropout_groups,
)
from rl_fzerox.core.manager.projection.policy import build_policy_data
from rl_fzerox.core.manager.projection.tracks import build_track_sampling_data
from rl_fzerox.core.manager.run_spec import ManagedRunConfig
from rl_fzerox.core.runtime_spec.paths import project_root_dir


def validate_launch_support(config: ManagedRunConfig) -> None:
    unsupported: list[str] = []
    if config.tracks.pool_mode != "built_in":
        unsupported.append("x cup launch is not wired into training")
    if unsupported:
        joined = "; ".join(unsupported)
        raise ValueError(f"Cannot launch this config yet: {joined}")


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
        "camera_setting": "close_behind",
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
        "reverse_time_penalty_scale": reward.reverse_time_penalty_scale,
        "slow_speed_time_penalty_scale": reward.slow_speed_time_penalty_scale,
        "slow_speed_time_penalty_start_kph": reward.slow_speed_time_penalty_start_kph,
        "slow_speed_time_penalty_power": reward.slow_speed_time_penalty_power,
        "progress_bucket_distance": reward.progress_bucket_distance,
        "progress_bucket_reward": reward.progress_bucket_reward,
        "progress_reward_interval_frames": reward.progress_reward_interval_frames,
        "suspend_progress_while_outside_track_bounds": (
            reward.suspend_progress_while_outside_track_bounds
        ),
        "outside_bounds_reentry_progress_distance_cap": (
            reward.outside_bounds_reentry_progress_distance_cap
        ),
        "outside_track_frame_penalty": reward.outside_track_frame_penalty,
        "lap_completion_bonus": reward.lap_completion_bonus,
        "lap_position_scale": reward.lap_position_scale,
        "energy_loss_epsilon": reward.energy_loss_epsilon,
        "energy_refill_progress_multiplier": reward.energy_refill_progress_multiplier,
        "dirt_progress_multiplier": reward.dirt_progress_multiplier,
        "ice_progress_multiplier": reward.ice_progress_multiplier,
        "dirt_entry_penalty": reward.dirt_entry_penalty,
        "ice_entry_penalty": reward.ice_entry_penalty,
        "energy_refill_collision_cooldown_frames": reward.energy_refill_collision_cooldown_frames,
        "air_brake_request_penalty": reward.air_brake_request_penalty,
        "manual_boost_reward": reward.manual_boost_reward,
        "boost_pad_reward": reward.boost_pad_reward,
        "boost_pad_reward_progress_window": reward.boost_pad_reward_progress_window,
        "lean_request_penalty": reward.lean_request_penalty,
        "grounded_pitch_penalty": reward.grounded_pitch_penalty,
        "damage_taken_frame_penalty": reward.damage_taken_frame_penalty,
        "damage_taken_streak_ramp_penalty": reward.damage_taken_streak_ramp_penalty,
        "damage_taken_streak_cap_frames": reward.damage_taken_streak_cap_frames,
        "airborne_landing_reward": reward.airborne_landing_reward,
        "collision_recoil_penalty": reward.collision_recoil_penalty,
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


def effective_train_algorithm(config: ManagedRunConfig) -> str:
    if continuous_action_axes(config):
        return (
            TRAINING_ALGORITHMS.maskable_hybrid_recurrent_ppo
            if config.policy.recurrent_enabled
            else TRAINING_ALGORITHMS.maskable_hybrid_action_ppo
        )
    return (
        TRAINING_ALGORITHMS.maskable_recurrent_ppo
        if config.policy.recurrent_enabled
        else TRAINING_ALGORITHMS.maskable_ppo
    )


def default_core_path() -> Path:
    return (project_root_dir() / "local" / "libretro" / "mupen64plus_next_libretro.so").resolve()


def default_rom_path() -> Path:
    return (project_root_dir() / "local" / "roms" / "F-Zero X (USA).n64").resolve()
