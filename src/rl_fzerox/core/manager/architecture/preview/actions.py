# src/rl_fzerox/core/manager/architecture/preview/actions.py
from __future__ import annotations

from rl_fzerox.core.manager.architecture.models import ActionBranchPreview
from rl_fzerox.core.manager.run_spec import ManagedRunConfig


def action_branch_previews(config: ManagedRunConfig) -> tuple[ActionBranchPreview, ...]:
    branches: list[ActionBranchPreview] = []
    if config.action.steering_mode == "continuous":
        branches.append(ActionBranchPreview(name="steer", kind="continuous", size=1, enabled=True))
    else:
        branches.append(
            ActionBranchPreview(
                name="steer",
                kind="discrete",
                size=int(config.action.steer_buckets),
                enabled=True,
            )
        )
    if config.action.drive_mode == "pwm":
        branches.append(
            ActionBranchPreview(
                name="throttle",
                kind="continuous",
                size=1,
                enabled=not config.action.force_full_throttle,
                mask_label=None if not config.action.force_full_throttle else "forced full",
            )
        )
    else:
        branches.append(
            ActionBranchPreview(
                name="throttle",
                kind="discrete",
                size=2,
                enabled=not config.action.force_full_throttle,
                mask_label=None if not config.action.force_full_throttle else "forced engaged",
            )
        )
    if config.action.include_air_brake:
        air_brake_mask_label = None
        if not config.action.enable_air_brake:
            air_brake_mask_label = "masked idle"
        elif config.action.mask_air_brake_on_ground:
            air_brake_mask_label = "air-only"
        if config.action.air_brake_mode == "pwm":
            branches.append(
                ActionBranchPreview(
                    name="air_brake",
                    kind="continuous",
                    size=1,
                    enabled=config.action.enable_air_brake,
                    mask_label=air_brake_mask_label,
                )
            )
        else:
            branches.append(
                ActionBranchPreview(
                    name="air_brake",
                    kind="discrete",
                    size=2,
                    enabled=config.action.enable_air_brake,
                    mask_label=air_brake_mask_label,
                )
            )
    if config.action.include_boost:
        boost_mask_label = None
        if not config.action.enable_boost:
            boost_mask_label = "masked idle"
        else:
            boost_guards: list[str] = []
            if config.action.boost_unmask_max_speed_kph is not None:
                boost_guards.insert(0, f"≤ {config.action.boost_unmask_max_speed_kph:g} kph")
            if config.action.boost_min_energy_fraction > 0:
                boost_guards.append(f"≥ {config.action.boost_min_energy_fraction * 100:g}% energy")
            if config.action.mask_boost_when_active:
                boost_guards.append("idle only")
            if config.action.boost_request_lockout_frames > 0:
                boost_guards.append(f"{config.action.boost_request_lockout_frames:d}f cooldown")
            boost_mask_label = ", ".join(boost_guards) if boost_guards else None
        branches.append(
            ActionBranchPreview(
                name="boost",
                kind="discrete",
                size=2,
                enabled=config.action.enable_boost,
                mask_label=boost_mask_label,
            )
        )
    if config.action.include_lean:
        lean_mask_label = None
        if not config.action.enable_lean:
            lean_mask_label = "masked idle"
        elif config.action.lean_unmask_min_speed_kph is not None:
            lean_mask_label = f"≥ {config.action.lean_unmask_min_speed_kph:g} kph"
        if config.action.lean_output_mode == "independent_buttons":
            branches.extend(
                (
                    ActionBranchPreview(
                        name="lean_left",
                        kind="discrete",
                        size=2,
                        enabled=config.action.enable_lean,
                        mask_label=lean_mask_label,
                    ),
                    ActionBranchPreview(
                        name="lean_right",
                        kind="discrete",
                        size=2,
                        enabled=config.action.enable_lean,
                        mask_label=lean_mask_label,
                    ),
                )
            )
        else:
            branches.append(
                ActionBranchPreview(
                    name="lean",
                    kind="discrete",
                    size=4 if config.action.lean_output_mode == "four_way_categorical" else 3,
                    enabled=config.action.enable_lean,
                    mask_label=lean_mask_label,
                )
            )
            if config.action.include_spin:
                branches.append(
                    ActionBranchPreview(
                        name="spin",
                        kind="discrete",
                        size=3,
                        enabled=config.action.enable_spin,
                        mask_label=None if config.action.enable_spin else "masked idle",
                    )
                )
    if config.action.include_pitch:
        if config.action.pitch_mode == "continuous":
            branches.append(
                ActionBranchPreview(
                    name="pitch",
                    kind="continuous",
                    size=1,
                    enabled=True,
                )
            )
        else:
            branches.append(
                ActionBranchPreview(
                    name="pitch",
                    kind="discrete",
                    size=int(config.action.pitch_buckets),
                    enabled=config.action.enable_pitch,
                    mask_label=None if config.action.enable_pitch else "masked neutral",
                )
            )
    return tuple(branches)


def action_net_detail(
    *,
    pi_output_dim: int,
    action_branches: tuple[ActionBranchPreview, ...],
    continuous_action_dims: int,
    discrete_action_logits: int,
) -> str:
    branch_details = ", ".join(action_branch_detail(branch) for branch in action_branches)
    return (
        f"{pi_output_dim} → {continuous_action_dims} continuous, "
        f"{discrete_action_logits} logits ({branch_details})"
    )


def action_branch_detail(branch: ActionBranchPreview) -> str:
    detail = f"{branch.name} {branch.size}"
    if branch.mask_label is not None:
        return f"{detail} {branch.mask_label}"
    return detail
