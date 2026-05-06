# src/rl_fzerox/core/manager/training.py
"""Project manager-owned run configs into concrete training configs."""

from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.config.paths import project_root_dir
from rl_fzerox.core.config.schema import TrainAppConfig
from rl_fzerox.core.config.track_registry import expand_track_registry_metadata
from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS
from rl_fzerox.core.manager.config import ManagedRunConfig
from rl_fzerox.core.manager.paths import manager_runs_root
from rl_fzerox.core.manager.training_projection import (
    build_action_data,
    build_observation_data,
    build_policy_data,
    build_state_feature_dropout_groups,
    build_track_sampling_data,
    continuous_action_axes,
    fork_observation_signature,
    fork_policy_signature,
)


def build_managed_train_app_config(
    config: ManagedRunConfig,
    *,
    run_id: str,
    run_dir: Path,
) -> TrainAppConfig:
    """Project one manager-owned config into the current training schema."""

    _validate_launch_support(config)
    train_data = {
        "seed": config.seed,
        "emulator": _emulator_data(config),
        "track": {},
        "env": _env_data(config),
        "reward": _reward_data(config),
        "policy": build_policy_data(config),
        "train": _train_data(config, run_id=run_id, run_dir=run_dir),
    }
    expand_track_registry_metadata(
        train_data,
        config_root=project_root_dir().resolve(),
    )
    return TrainAppConfig.model_validate(train_data)


def build_managed_resume_train_app_config(
    config: ManagedRunConfig,
    *,
    run_id: str,
    run_dir: Path,
    tensorboard_step_offset: int = 0,
) -> TrainAppConfig:
    """Project one manager config into an in-place full-model resume run."""

    train_config = build_managed_train_app_config(
        config,
        run_id=run_id,
        run_dir=run_dir,
    )
    return train_config.model_copy(
        update={
            "train": train_config.train.model_copy(
                update={
                    "continue_run_dir": run_dir,
                    "resume_run_dir": run_dir,
                    "resume_source_algorithm": train_config.train.algorithm,
                    "resume_artifact": "latest",
                    "resume_mode": "full_model",
                    "tensorboard_step_offset": tensorboard_step_offset,
                }
            )
        }
    )


def build_managed_fork_train_app_config(
    config: ManagedRunConfig,
    *,
    run_id: str,
    run_dir: Path,
    source_run_dir: Path,
    source_artifact: str,
    tensorboard_step_offset: int = 0,
) -> TrainAppConfig:
    """Project one manager config into a child run warm-started from another run."""

    train_config = build_managed_train_app_config(
        config,
        run_id=run_id,
        run_dir=run_dir,
    )
    return train_config.model_copy(
        update={
            "train": train_config.train.model_copy(
                update={
                    "resume_run_dir": source_run_dir,
                    "resume_source_algorithm": train_config.train.algorithm,
                    "resume_artifact": source_artifact,
                    "resume_mode": "weights_only",
                    "tensorboard_step_offset": tensorboard_step_offset,
                }
            )
        }
    )


def assert_managed_fork_compatible(
    source_config: ManagedRunConfig,
    candidate_config: ManagedRunConfig,
) -> None:
    """Fail early when one forked child config is incompatible with its source checkpoint."""

    source_train = build_managed_train_app_config(
        source_config,
        run_id="source-compat",
        run_dir=Path("local/manager/source-compat"),
    )
    candidate_train = build_managed_train_app_config(
        candidate_config,
        run_id="candidate-compat",
        run_dir=Path("local/manager/candidate-compat"),
    )

    source_signature = _fork_compatibility_signature(source_train)
    candidate_signature = _fork_compatibility_signature(candidate_train)
    if source_signature == candidate_signature:
        return

    changed = [
        label
        for key, label in (
            ("algorithm", "training algorithm"),
            ("observation", "observation structure"),
            ("action", "action layout"),
            ("policy", "policy architecture"),
        )
        if source_signature[key] != candidate_signature[key]
    ]
    detail = ", ".join(changed) if changed else "checkpoint structure"
    raise ValueError(
        "Cannot fork from this checkpoint after incompatible edits: "
        f"{detail}. Change reward/training knobs only, or start a fresh run."
    )


def _fork_compatibility_signature(train_config: TrainAppConfig) -> dict[str, object]:
    observation = fork_observation_signature(train_config)
    action = {
        "name": train_config.env.action.name,
        "steer_buckets": train_config.env.action.steer_buckets,
        "pitch_buckets": train_config.env.action.pitch_buckets,
        "independent_lean_buttons": train_config.env.action.independent_lean_buttons,
        "layout_continuous_axes": tuple(train_config.env.action.layout_continuous_axes),
        "layout_discrete_axes": tuple(train_config.env.action.layout_discrete_axes),
    }
    policy = fork_policy_signature(train_config)
    return {
        "algorithm": train_config.train.algorithm,
        "observation": observation,
        "action": action,
        "policy": policy,
    }


def _validate_launch_support(config: ManagedRunConfig) -> None:
    unsupported: list[str] = []
    if config.tracks.pool_mode != "built_in":
        unsupported.append("x cup launch is not wired into training")
    if unsupported:
        joined = "; ".join(unsupported)
        raise ValueError(f"Cannot launch this config yet: {joined}")


def _env_data(config: ManagedRunConfig) -> dict[str, object]:
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


def _emulator_data(config: ManagedRunConfig) -> dict[str, object]:
    return {
        "core_path": _default_core_path(),
        "rom_path": _default_rom_path(),
        "renderer": config.environment.renderer,
    }
def _reward_data(config: ManagedRunConfig) -> dict[str, object]:
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
        "airborne_pitch_up_penalty": reward.airborne_pitch_up_penalty,
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


def _train_data(config: ManagedRunConfig, *, run_id: str, run_dir: Path) -> dict[str, object]:
    train = config.train
    return {
        "algorithm": _effective_train_algorithm(config),
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


def _effective_train_algorithm(config: ManagedRunConfig) -> str:
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


def _default_core_path() -> Path:
    return (project_root_dir() / "local" / "libretro" / "mupen64plus_next_libretro.so").resolve()


def _default_rom_path() -> Path:
    return (project_root_dir() / "local" / "roms" / "F-Zero X (USA).n64").resolve()
