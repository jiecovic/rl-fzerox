# src/rl_fzerox/core/manager/training.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from rl_fzerox.core.config import load_train_app_config
from rl_fzerox.core.config.paths import config_root_dir
from rl_fzerox.core.config.schema import TrainAppConfig
from rl_fzerox.core.config.track_registry import expand_track_registry_metadata
from rl_fzerox.core.config.vehicle_catalog import (
    resolve_engine_setting,
    vehicle_by_id,
)
from rl_fzerox.core.domain.action_adapters import ACTION_ADAPTERS
from rl_fzerox.core.domain.courses import built_in_course_ref_by_id
from rl_fzerox.core.domain.training_algorithms import TRAINING_ALGORITHMS
from rl_fzerox.core.manager.config import ManagedRunConfig, ManagedStateComponentConfig


def build_managed_train_app_config(
    config: ManagedRunConfig,
    *,
    run_id: str,
    run_dir: Path,
) -> TrainAppConfig:
    """Project one manager-owned config into the current training schema."""

    _validate_launch_support(config)
    base_data = _base_train_config().model_dump(mode="python")
    base_data["seed"] = config.seed
    base_data["emulator"]["renderer"] = config.environment.renderer
    base_data["track"] = {}
    base_data["env"] = _env_data(config)
    base_data["reward"] = _reward_data(config)
    base_data["policy"] = _policy_data(config)
    base_data["train"] = _train_data(config, run_id=run_id, run_dir=run_dir)
    expand_track_registry_metadata(base_data, config_root=config_root_dir().resolve())
    return TrainAppConfig.model_validate(base_data)


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
    observation = _fork_observation_signature(train_config)
    action = {
        "name": train_config.env.action.name,
        "steer_buckets": train_config.env.action.steer_buckets,
        "pitch_buckets": train_config.env.action.pitch_buckets,
        "independent_lean_buttons": train_config.env.action.independent_lean_buttons,
        "layout_continuous_axes": tuple(train_config.env.action.layout_continuous_axes),
        "layout_discrete_axes": tuple(train_config.env.action.layout_discrete_axes),
    }
    policy = _fork_policy_signature(train_config)
    return {
        "algorithm": train_config.train.algorithm,
        "observation": observation,
        "action": action,
        "policy": policy,
    }


def _fork_observation_signature(train_config: TrainAppConfig) -> dict[str, object]:
    observation = train_config.env.observation
    return {
        "mode": observation.mode,
        "preset": observation.preset,
        "frame_stack": observation.frame_stack,
        "stack_mode": observation.stack_mode,
        "minimap_layer": observation.minimap_layer,
        "state_components": tuple(
            component.model_dump(mode="python") for component in observation.state_components or ()
        ),
        "excluded_state_features": tuple(observation.excluded_state_features),
    }


def _fork_policy_signature(train_config: TrainAppConfig) -> dict[str, object]:
    extractor = train_config.policy.extractor
    recurrent = train_config.policy.recurrent
    net_arch = train_config.policy.net_arch
    return {
        "extractor": {
            "conv_profile": extractor.conv_profile,
            "custom_conv_layers": tuple(
                layer.model_dump(mode="python") for layer in extractor.custom_conv_layers
            ),
            "features_dim": extractor.features_dim,
            "state_net_arch": tuple(extractor.state_net_arch or ()),
            "fusion_features_dim": extractor.fusion_features_dim,
            "layer_norm": extractor.layer_norm,
        },
        "recurrent": {
            "enabled": recurrent.enabled,
            "hidden_size": recurrent.hidden_size,
            "n_lstm_layers": recurrent.n_lstm_layers,
            "shared_lstm": recurrent.shared_lstm,
            "enable_critic_lstm": recurrent.enable_critic_lstm,
        },
        "net_arch": {
            "pi": tuple(net_arch.pi),
            "vf": tuple(net_arch.vf),
        },
    }


@lru_cache(maxsize=1)
def _base_train_config() -> TrainAppConfig:
    return load_train_app_config(config_root_dir() / "train_base.yaml")


def _validate_launch_support(config: ManagedRunConfig) -> None:
    unsupported: list[str] = []
    if config.tracks.pool_mode != "built_in":
        unsupported.append("x cup launch is not wired into training")
    if unsupported:
        joined = "; ".join(unsupported)
        raise ValueError(f"Cannot launch this config yet: {joined}")


def _env_data(config: ManagedRunConfig) -> dict[str, object]:
    env_data = _base_train_config().env.model_dump(mode="python")
    env_data.update(
        {
            "action_repeat": config.action.action_repeat,
            "max_episode_steps": config.environment.max_episode_steps,
            "progress_frontier_stall_limit_frames": (
                config.environment.progress_frontier_stall_limit_frames
            ),
            "progress_frontier_epsilon": config.environment.progress_frontier_epsilon,
            "boost_min_energy_fraction": config.action.boost_min_energy_fraction,
            "track_sampling": _track_sampling_data(config),
            "action": _action_data(config),
            "observation": _observation_data(config),
        }
    )
    return env_data


def _track_sampling_data(config: ManagedRunConfig) -> dict[str, object]:
    return {
        "enabled": True,
        "sampling_mode": "random" if config.tracks.sampling_mode == "equal" else "step_balanced",
        "entries": _track_sampling_entries(config),
    }


def _action_data(config: ManagedRunConfig) -> dict[str, object]:
    continuous_axes = _continuous_action_axes(config)
    discrete_axes = _discrete_action_axes(config)
    return {
        "name": (
            ACTION_ADAPTERS.configured_hybrid
            if continuous_axes
            else ACTION_ADAPTERS.configured_discrete
        ),
        "steer_buckets": config.action.steer_buckets,
        "steer_response_power": 1.0,
        "continuous_drive_deadzone": config.action.continuous_drive_deadzone,
        "continuous_drive_full_threshold": config.action.continuous_drive_full_threshold,
        "continuous_drive_min_thrust": config.action.continuous_drive_min_thrust,
        "continuous_air_brake_deadzone": config.action.continuous_air_brake_deadzone,
        "continuous_air_brake_full_threshold": config.action.continuous_air_brake_full_threshold,
        "continuous_air_brake_min_duty": config.action.continuous_air_brake_min_duty,
        "force_full_throttle": config.action.force_full_throttle,
        "mask_air_brake_on_ground": config.action.mask_air_brake_on_ground,
        "continuous_air_brake_mode": _continuous_air_brake_mode(config),
        "lean_mode": config.action.lean_mode,
        "boost_unmask_max_speed_kph": config.action.boost_unmask_max_speed_kph,
        "lean_unmask_min_speed_kph": config.action.lean_unmask_min_speed_kph,
        "lean_initial_lockout_frames": config.action.lean_initial_lockout_frames,
        "pitch_buckets": config.action.pitch_buckets,
        "independent_lean_buttons": config.action.lean_output_mode == "independent_buttons",
        "layout_continuous_axes": list(continuous_axes),
        "layout_discrete_axes": list(discrete_axes),
        "configured_mask_overrides": _configured_mask_overrides(config),
    }


def _observation_data(config: ManagedRunConfig) -> dict[str, object]:
    state_components: list[dict[str, object]] = []
    zeroed_components: list[str] = []
    active_feature_names: set[str] = set()
    for component in config.observation.state_components:
        if component.mode == "exclude":
            continue
        state_components.append(_state_component_data(component))
        active_feature_names.update(_component_feature_names(component))
        if component.mode == "zero":
            zeroed_components.append(component.name)

    zeroed_features = [
        feature.name
        for feature in config.observation.state_feature_modes
        if feature.mode == "zero" and feature.name in active_feature_names
    ]
    excluded_features = [
        feature.name
        for feature in config.observation.state_feature_modes
        if feature.mode == "exclude" and feature.name in active_feature_names
    ]
    return {
        "mode": "image_state",
        "preset": config.observation.preset,
        "frame_stack": config.observation.frame_stack,
        "stack_mode": config.observation.stack_mode,
        "minimap_layer": config.observation.minimap_layer,
        "resize_filter": config.observation.resize_filter,
        "minimap_resize_filter": config.observation.minimap_resize_filter,
        "state_components": state_components,
        "zeroed_state_components": zeroed_components,
        "zeroed_state_features": zeroed_features,
        "excluded_state_features": excluded_features,
    }


def _state_component_data(component: ManagedStateComponentConfig) -> dict[str, object]:
    data: dict[str, object] = {"name": component.name}
    if component.encoding is not None:
        data["encoding"] = component.encoding
    if component.progress_source is not None:
        data["progress_source"] = component.progress_source
    if component.length is not None:
        data["length"] = component.length
    if component.controls is not None:
        data["controls"] = component.controls
    return data


def _component_feature_names(component: ManagedStateComponentConfig) -> tuple[str, ...]:
    from rl_fzerox.core.envs.observations.state.components import state_component_definition

    settings = component.data()
    return tuple(
        feature.name
        for feature in state_component_definition(settings).features(settings)
    )


def _policy_data(config: ManagedRunConfig) -> dict[str, object]:
    return {
        "activation": config.policy.activation,
        "extractor": {
            "conv_profile": config.policy.conv_profile,
            "custom_conv_layers": [
                layer.model_dump(mode="python") for layer in config.policy.custom_conv_layers
            ],
            "features_dim": config.policy.features_dim,
            "state_net_arch": list(config.policy.state_net_arch),
            "fusion_features_dim": config.policy.fusion_features_dim,
            "layer_norm": config.policy.layer_norm,
        },
        "recurrent": {
            "enabled": config.policy.recurrent_enabled,
            "hidden_size": config.policy.recurrent_hidden_size,
            "n_lstm_layers": config.policy.recurrent_n_lstm_layers,
            "shared_lstm": config.policy.recurrent_shared_lstm,
            "enable_critic_lstm": config.policy.recurrent_enable_critic_lstm,
        },
        "action_bias": {
            "gas_on_logit": (
                config.policy.gas_on_logit if config.action.drive_mode == "on_off" else 0.0
            )
        },
        "net_arch": {
            "pi": list(config.policy.pi_net_arch),
            "vf": list(config.policy.vf_net_arch),
        },
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
        "course_context_dropout_prob": train.course_context_dropout_prob,
        "output_root": run_dir.parent,
        "run_name": run_id,
        "explicit_run_dir": run_dir,
    }


def _effective_train_algorithm(config: ManagedRunConfig) -> str:
    if _continuous_action_axes(config):
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


def _continuous_action_axes(config: ManagedRunConfig) -> tuple[str, ...]:
    axes: list[str] = []
    if config.action.steering_mode == "continuous":
        axes.append("steer")
    if config.action.drive_mode == "pwm":
        axes.append("drive")
    if config.action.include_air_brake and config.action.air_brake_mode == "pwm":
        axes.append("air_brake")
    if config.action.include_pitch and config.action.pitch_mode == "continuous":
        axes.append("pitch")
    return tuple(axes)


def _discrete_action_axes(config: ManagedRunConfig) -> tuple[str, ...]:
    axes: list[str] = []
    if config.action.steering_mode == "discrete":
        axes.append("steer")
    if config.action.drive_mode == "on_off":
        axes.append("gas")
    if config.action.include_air_brake and config.action.air_brake_mode == "on_off":
        axes.append("air_brake")
    if config.action.include_boost:
        axes.append("boost")
    if config.action.include_lean:
        axes.append("lean")
    if config.action.include_pitch and config.action.pitch_mode == "discrete":
        axes.append("pitch")
    return tuple(axes)


def _configured_mask_overrides(config: ManagedRunConfig) -> dict[str, tuple[int, ...]] | None:
    overrides: dict[str, tuple[int, ...]] = {}
    if config.action.drive_mode == "on_off" and config.action.force_full_throttle:
        overrides["gas"] = (1,)
    if (
        config.action.include_air_brake
        and config.action.air_brake_mode == "on_off"
        and not config.action.enable_air_brake
    ):
        overrides["air_brake"] = (0,)
    if config.action.include_boost and not config.action.enable_boost:
        overrides["boost"] = (0,)
    if config.action.include_lean and not config.action.enable_lean:
        overrides["lean"] = (0,)
    if (
        config.action.include_pitch
        and config.action.pitch_mode == "discrete"
        and not config.action.enable_pitch
    ):
        overrides["pitch"] = (config.action.pitch_buckets // 2,)
    return overrides or None


def _continuous_air_brake_mode(config: ManagedRunConfig) -> str:
    if not config.action.include_air_brake:
        return "off"
    if config.action.air_brake_mode != "pwm":
        return "always"
    if not config.action.enable_air_brake:
        return "off"
    if config.action.mask_air_brake_on_ground:
        return "disable_on_ground"
    return "always"


def _track_sampling_entries(config: ManagedRunConfig) -> list[dict[str, object]]:
    source_vehicle_id = config.vehicle.selected_vehicle_ids[0]
    source_engine = _source_engine_setting(config)
    entries: list[dict[str, object]] = []
    for course_id in config.tracks.selected_course_ids:
        course_ref = _course_ref(course_id)
        for vehicle_id in config.vehicle.selected_vehicle_ids:
            entries.append(
                _track_sampling_entry(
                    course_id=course_id,
                    course_ref=course_ref,
                    race_mode=config.tracks.race_mode,
                    target_vehicle_id=vehicle_id,
                    source_vehicle_id=source_vehicle_id,
                    source_engine_setting_id=source_engine.id,
                    source_engine_setting_raw_value=source_engine.raw_value,
                    fixed_engine_setting_raw_value=(
                        config.vehicle.engine_setting_raw_value
                        if config.vehicle.engine_mode == "fixed"
                        else None
                    ),
                    random_engine_min_raw_value=(
                        config.vehicle.engine_setting_min_raw_value
                        if config.vehicle.engine_mode == "random_range"
                        else None
                    ),
                    random_engine_max_raw_value=(
                        config.vehicle.engine_setting_max_raw_value
                        if config.vehicle.engine_mode == "random_range"
                        else None
                    ),
                )
            )
    return entries


def _track_sampling_entry(
    *,
    course_id: str,
    course_ref: str,
    race_mode: str,
    target_vehicle_id: str,
    source_vehicle_id: str,
    source_engine_setting_id: str,
    source_engine_setting_raw_value: int,
    fixed_engine_setting_raw_value: int | None,
    random_engine_min_raw_value: int | None,
    random_engine_max_raw_value: int | None,
) -> dict[str, object]:
    vehicle = vehicle_by_id(target_vehicle_id)
    if fixed_engine_setting_raw_value is not None:
        target_engine = resolve_engine_setting(
            fixed_engine_setting_raw_value,
            context=f"manager track_sampling {course_id}/{target_vehicle_id}",
        )
        engine_id = target_engine.id
        engine_raw = target_engine.raw_value
        engine_suffix = engine_id
    else:
        if random_engine_min_raw_value is None or random_engine_max_raw_value is None:
            raise ValueError("random engine range requires both min and max raw values")
        engine_id = source_engine_setting_id
        engine_raw = source_engine_setting_raw_value
        engine_suffix = f"engine_range_{random_engine_min_raw_value}_{random_engine_max_raw_value}"

    return {
        "id": f"{course_id}_{race_mode}_{target_vehicle_id}_{engine_suffix}",
        "course_ref": course_ref,
        "mode": race_mode,
        "vehicle": target_vehicle_id,
        "vehicle_name": vehicle.display_name,
        "source_vehicle": source_vehicle_id,
        "engine_setting": engine_id,
        "engine_setting_raw_value": engine_raw,
        "source_engine_setting": source_engine_setting_id,
        "source_engine_setting_raw_value": source_engine_setting_raw_value,
        "engine_setting_min_raw_value": random_engine_min_raw_value,
        "engine_setting_max_raw_value": random_engine_max_raw_value,
    }


def _source_engine_setting(config: ManagedRunConfig):
    if config.vehicle.engine_mode == "fixed":
        raw_value = config.vehicle.engine_setting_raw_value
    else:
        raw_value = (
            config.vehicle.engine_setting_min_raw_value
            + config.vehicle.engine_setting_max_raw_value
        ) // 2
    return resolve_engine_setting(
        raw_value,
        context="manager source engine setting",
    )


def _course_ref(course_id: str) -> str:
    matches = built_in_course_ref_by_id(course_id)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one built-in course ref for {course_id!r}")
    return matches[0]
