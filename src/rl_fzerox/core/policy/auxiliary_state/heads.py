# src/rl_fzerox/core/policy/auxiliary_state/heads.py
"""Auxiliary-state prediction heads attached to policy latent features."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from sb3x.common.auxiliary_losses import PolicyAuxiliaryLoss
from torch import nn
from torch.nn import functional as F

from .targets import (
    AuxiliaryStateDecodedValue,
    AuxiliaryStateTargetName,
    resolve_auxiliary_state_target,
)

_SCALAR_TARGET_NAMES: tuple[AuxiliaryStateTargetName, ...] = (
    "vehicle_state.speed_norm",
    "vehicle_state.energy_frac",
    "vehicle_state.lateral_velocity_norm",
    "track_position.lap_progress",
    "track_position.edge_ratio",
    "track_position.height_above_ground_norm",
)
_BINARY_TARGET_NAMES: tuple[AuxiliaryStateTargetName, ...] = (
    "vehicle_state.reverse_active",
    "vehicle_state.airborne",
    "vehicle_state.can_boost",
    "vehicle_state.boost_active",
    "vehicle_state.sliding_active",
    "track_position.outside_track_bounds",
    "surface_state.on_refill_surface",
    "surface_state.on_dirt_surface",
    "surface_state.on_ice_surface",
)
_SCALAR_HEAD_INDEX = {name: index for index, name in enumerate(_SCALAR_TARGET_NAMES)}
_BINARY_HEAD_INDEX = {name: index for index, name in enumerate(_BINARY_TARGET_NAMES)}


@dataclass(frozen=True, slots=True)
class AuxiliaryStateLossTerm:
    name: AuxiliaryStateTargetName
    weight: float
    grounded_only: bool = False


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, bool]:
    if mask is None:
        return values.mean(), values.numel() > 0

    active = mask.to(dtype=values.dtype)
    active_sum = active.sum()
    if float(active_sum.item()) <= 0.0:
        return values.new_zeros(()), False
    return (values * active).sum() / active_sum, True


def _requested_target_names(
    names: Sequence[AuxiliaryStateTargetName] | None,
) -> tuple[AuxiliaryStateTargetName, ...]:
    if names is None:
        return _SCALAR_TARGET_NAMES + _BINARY_TARGET_NAMES + ("course_context.builtin_course_id",)
    return tuple(names)


def _needs_scalar_head(names: Sequence[AuxiliaryStateTargetName]) -> bool:
    return any(name in _SCALAR_HEAD_INDEX for name in names)


def _needs_binary_head(names: Sequence[AuxiliaryStateTargetName]) -> bool:
    return any(name in _BINARY_HEAD_INDEX for name in names)


def _needs_course_head(names: Sequence[AuxiliaryStateTargetName]) -> bool:
    return "course_context.builtin_course_id" in names


class AuxiliaryStateHeadBank(nn.Module):
    """Grouped aux heads over the shared policy latent."""

    def __init__(
        self,
        *,
        input_dim: int,
        head_arch: Sequence[int] = (),
        activation_fn: type[nn.Module] = nn.ReLU,
        builtin_course_count: int = 24,
    ) -> None:
        super().__init__()
        trunk_layers: list[nn.Module] = []
        trunk_input_dim = input_dim
        for hidden_dim in head_arch:
            trunk_layers.append(nn.Linear(trunk_input_dim, int(hidden_dim)))
            trunk_layers.append(activation_fn())
            trunk_input_dim = int(hidden_dim)
        self.trunk = nn.Identity() if not trunk_layers else nn.Sequential(*trunk_layers)
        self.scalar_head = nn.Linear(trunk_input_dim, len(_SCALAR_TARGET_NAMES))
        self.binary_head = nn.Linear(trunk_input_dim, len(_BINARY_TARGET_NAMES))
        self.course_head = nn.Linear(trunk_input_dim, builtin_course_count)

    def loss(
        self,
        latent: torch.Tensor,
        *,
        aux_targets: torch.Tensor,
        losses: Sequence[AuxiliaryStateLossTerm],
        sample_mask: torch.Tensor | None = None,
    ) -> PolicyAuxiliaryLoss | None:
        if not losses:
            return None

        requested_names: tuple[AuxiliaryStateTargetName, ...] = tuple(loss.name for loss in losses)
        trunk_latent = self.trunk(latent)
        scalar_predictions = (
            self.scalar_head(trunk_latent) if _needs_scalar_head(requested_names) else None
        )
        binary_logits = (
            self.binary_head(trunk_latent) if _needs_binary_head(requested_names) else None
        )
        course_logits = (
            self.course_head(trunk_latent) if _needs_course_head(requested_names) else None
        )
        airborne_index = resolve_auxiliary_state_target("vehicle_state.airborne").vector_start
        airborne = aux_targets[:, airborne_index]
        total_loss = latent.new_zeros(())
        metrics: dict[str, float] = {}
        active_losses = 0

        for loss_term in losses:
            target = resolve_auxiliary_state_target(loss_term.name)
            effective_mask = sample_mask
            if loss_term.grounded_only:
                grounded_mask = airborne < 0.5
                effective_mask = (
                    grounded_mask
                    if effective_mask is None
                    else (effective_mask.bool() & grounded_mask)
                )

            loss_value: torch.Tensor
            if target.kind == "scalar":
                if scalar_predictions is None:
                    raise RuntimeError("Scalar auxiliary prediction head is unavailable")
                predicted = scalar_predictions[:, _SCALAR_HEAD_INDEX[target.name]]
                target_values = aux_targets[:, target.vector_start]
                per_sample = F.smooth_l1_loss(
                    predicted,
                    target_values,
                    reduction="none",
                )
                loss_value, has_active_samples = _masked_mean(per_sample, effective_mask)
            elif target.kind == "binary":
                if binary_logits is None:
                    raise RuntimeError("Binary auxiliary prediction head is unavailable")
                logits = binary_logits[:, _BINARY_HEAD_INDEX[target.name]]
                target_values = aux_targets[:, target.vector_start]
                per_sample = F.binary_cross_entropy_with_logits(
                    logits,
                    target_values,
                    reduction="none",
                )
                loss_value, has_active_samples = _masked_mean(per_sample, effective_mask)
            elif target.kind == "categorical":
                if course_logits is None:
                    raise RuntimeError("Categorical auxiliary prediction head is unavailable")
                logits = course_logits
                target_slice = aux_targets[:, target.vector_start : target.vector_stop]
                target_indices = target_slice.argmax(dim=1)
                per_sample = F.cross_entropy(
                    logits,
                    target_indices,
                    reduction="none",
                )
                loss_value, has_active_samples = _masked_mean(per_sample, effective_mask)
            else:  # pragma: no cover - kept for defensive exhaustiveness
                raise ValueError(f"Unsupported auxiliary target kind: {target.kind!r}")

            if not has_active_samples:
                continue

            total_loss = total_loss + float(loss_term.weight) * loss_value
            metrics[target.name] = float(loss_value.detach().cpu().item())
            active_losses += 1

        if active_losses == 0:
            return None
        metrics["__total__"] = float(total_loss.detach().cpu().item())
        return PolicyAuxiliaryLoss(total_loss=total_loss, metrics=metrics)

    def predict_values(
        self,
        latent: torch.Tensor,
        *,
        names: Sequence[AuxiliaryStateTargetName] | None = None,
    ) -> dict[AuxiliaryStateTargetName, AuxiliaryStateDecodedValue]:
        requested_names = _requested_target_names(names)
        if not requested_names:
            return {}

        trunk_latent = self.trunk(latent)
        scalar_predictions = (
            self.scalar_head(trunk_latent) if _needs_scalar_head(requested_names) else None
        )
        binary_logits = (
            self.binary_head(trunk_latent) if _needs_binary_head(requested_names) else None
        )
        course_logits = (
            self.course_head(trunk_latent) if _needs_course_head(requested_names) else None
        )

        sample_index = 0
        values: dict[AuxiliaryStateTargetName, AuxiliaryStateDecodedValue] = {}
        for name in requested_names:
            if name not in _SCALAR_HEAD_INDEX:
                continue
            if scalar_predictions is None:
                raise RuntimeError("Scalar auxiliary prediction head is unavailable")
            values[name] = float(
                scalar_predictions[sample_index, _SCALAR_HEAD_INDEX[name]].detach().cpu().item()
            )
        for name in requested_names:
            if name not in _BINARY_HEAD_INDEX:
                continue
            if binary_logits is None:
                raise RuntimeError("Binary auxiliary prediction head is unavailable")
            probability = binary_logits[sample_index, _BINARY_HEAD_INDEX[name]].sigmoid()
            values[name] = float(probability.detach().cpu().item())

        if "course_context.builtin_course_id" in requested_names:
            if course_logits is None:
                raise RuntimeError("Categorical auxiliary prediction head is unavailable")
            course_probabilities = course_logits[sample_index].softmax(dim=0)
            course_confidence, course_index = course_probabilities.max(dim=0)
            values["course_context.builtin_course_id"] = {
                "index": int(course_index.detach().cpu().item()),
                "confidence": float(course_confidence.detach().cpu().item()),
                "probabilities": [
                    float(probability)
                    for probability in course_probabilities.detach().cpu().tolist()
                ],
            }
        return values
