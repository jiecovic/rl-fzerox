from __future__ import annotations

from rl_fzerox.core.domain.action_adapters import ACTION_ADAPTERS
from rl_fzerox.core.manager.config import ManagedRunConfig


def build_action_data(config: ManagedRunConfig) -> dict[str, object]:
    continuous_axes = continuous_action_axes(config)
    discrete_axes = discrete_action_axes(config)
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
