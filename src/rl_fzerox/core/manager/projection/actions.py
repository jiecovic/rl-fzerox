# src/rl_fzerox/core/manager/projection/actions.py
from __future__ import annotations

from rl_fzerox.core.manager.run_spec import ManagedRunConfig


def build_action_data(config: ManagedRunConfig) -> dict[str, object]:
    continuous_axes = continuous_action_axes(config)
    discrete_axes = discrete_action_axes(config)
    return {
        "adapter_name": "configured_hybrid",
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
        "air_brake_episode_mask_probability": (
            config.action.air_brake_episode_mask_probability
            if config.action.include_air_brake and config.action.air_brake_mode == "on_off"
            else 0.0
        ),
        "continuous_air_brake_mode": _continuous_air_brake_mode(config),
        "lean_mode": config.action.lean_mode,
        "lean_output_mode": config.action.lean_output_mode,
        "mask_boost_when_active": config.action.mask_boost_when_active,
        "mask_boost_when_airborne": config.action.mask_boost_when_airborne,
        "boost_decision_interval_frames": (
            config.action.boost_decision_interval_steps * config.action.action_repeat
        ),
        "boost_request_lockout_frames": config.action.boost_request_lockout_frames,
        "spin_cooldown_frames": config.action.spin_cooldown_frames,
        "spin_episode_mask_probability": (
            config.action.spin_episode_mask_probability if config.action.include_spin else 0.0
        ),
        "boost_unmask_max_speed_kph": config.action.boost_unmask_max_speed_kph,
        "lean_unmask_min_speed_kph": config.action.lean_unmask_min_speed_kph,
        "lean_initial_lockout_frames": config.action.lean_initial_lockout_frames,
        "lean_episode_mask_probability": config.action.lean_episode_mask_probability,
        "mask_pitch_on_ground": _mask_pitch_on_ground(config),
        "pitch_deadzone": config.action.pitch_deadzone,
        "pitch_buckets": config.action.pitch_buckets,
        "layout_continuous_axes": list(continuous_axes),
        "layout_discrete_axes": list(discrete_axes),
        "mask": _action_mask_data(config),
    }


def continuous_action_axes(config: ManagedRunConfig) -> tuple[str, ...]:
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


def discrete_action_axes(config: ManagedRunConfig) -> tuple[str, ...]:
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
        if config.action.lean_output_mode == "independent_buttons":
            axes.extend(("lean_left", "lean_right"))
        else:
            axes.append("lean")
            if config.action.include_spin:
                axes.append("spin")
    if config.action.include_pitch and config.action.pitch_mode == "discrete":
        axes.append("pitch")
    return tuple(axes)


def _action_mask_data(config: ManagedRunConfig) -> dict[str, tuple[int, ...]] | None:
    mask_data: dict[str, tuple[int, ...]] = {}
    if config.action.drive_mode == "on_off" and config.action.force_full_throttle:
        mask_data["gas"] = (1,)
    if (
        config.action.include_air_brake
        and config.action.air_brake_mode == "on_off"
        and not config.action.enable_air_brake
    ):
        mask_data["air_brake"] = (0,)
    if config.action.include_boost and not config.action.enable_boost:
        mask_data["boost"] = (0,)
    if config.action.include_lean and not config.action.enable_lean:
        if config.action.lean_output_mode == "independent_buttons":
            mask_data["lean_left"] = (0,)
            mask_data["lean_right"] = (0,)
        else:
            mask_data["lean"] = (0,)
    if config.action.include_spin and not config.action.enable_spin:
        mask_data["spin"] = (0,)
    if (
        config.action.include_pitch
        and config.action.pitch_mode == "discrete"
        and not config.action.enable_pitch
    ):
        mask_data["pitch"] = (config.action.pitch_buckets // 2,)
    return mask_data or None


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


def _mask_pitch_on_ground(config: ManagedRunConfig) -> bool:
    return bool(
        config.action.include_pitch
        and config.action.pitch_mode == "discrete"
        and config.action.mask_pitch_on_ground
    )
