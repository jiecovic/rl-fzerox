# src/rl_fzerox/ui/watch/view/panels/content/state_vector.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.policy.auxiliary_state.targets import (
    AuxiliaryStateTargetName,
    auxiliary_state_target_bounds,
    resolve_auxiliary_state_target,
)
from rl_fzerox.core.runtime_spec.schema import PolicyConfig
from rl_fzerox.ui.watch.view.panels.core.lines import (
    panel_divider,
    panel_heading,
    panel_line,
)
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelLine, PanelSection, StatusIcon


def policy_state_sections(
    *,
    observation_state: StateVector | None,
    observation_state_reference: StateVector | None = None,
    feature_names: tuple[str, ...],
    policy_config: PolicyConfig | None = None,
    auxiliary_predictions: dict[str, object] | None = None,
    auxiliary_targets: dict[str, object] | None = None,
    zeroed_features: frozenset[str] = frozenset(),
    watch_zeroed_features: frozenset[str] = frozenset(),
) -> list[PanelSection]:
    aux_losses = _auxiliary_loss_specs(policy_config)
    if observation_state is None and not aux_losses:
        return []

    observation_values = (
        None
        if observation_state is None
        else np.asarray(observation_state, dtype=np.float32).reshape(-1)
    )
    reference_values = (
        None
        if observation_state_reference is None
        else np.asarray(observation_state_reference, dtype=np.float32).reshape(-1)
    )
    names = _resolved_state_feature_names(feature_names, observation_values)
    section_lines: list[PanelLine] = []
    if aux_losses:
        section_lines.append(
            panel_line(
                " ",
                _state_vector_header_value(),
                PALETTE.text_muted,
                status_icon="none",
            )
        )
    for group_title, group_prefix, component_name in _state_vector_groups(
        names,
        auxiliary_loss_names=tuple(aux_losses),
    ):
        group_zeroed = component_name is not None and component_name in zeroed_features
        group_lines = _state_vector_group_lines(
            names=names,
            observation_values=observation_values,
            reference_values=reference_values,
            group_prefix=group_prefix,
            auxiliary_losses=aux_losses,
            auxiliary_predictions=auxiliary_predictions,
            auxiliary_targets=auxiliary_targets,
            zeroed=group_zeroed,
            zeroed_features=zeroed_features,
            watch_zeroed_features=watch_zeroed_features,
        )
        if group_lines:
            if section_lines:
                section_lines.append(panel_divider())
            section_lines.append(
                panel_heading(_state_vector_group_title(group_title, group_zeroed))
            )
            section_lines.extend(group_lines)
    if not section_lines:
        return []
    return [PanelSection(title="State Vector", lines=section_lines)]


@dataclass(frozen=True, slots=True)
class _AuxiliaryLossSpec:
    name: AuxiliaryStateTargetName
    grounded_only: bool


@dataclass(frozen=True, slots=True)
class _StateVectorColumnLayout:
    pred_width: int = 6
    ref_width: int = 6
    err_width: int = 5
    obs_width: int = 6


STATE_VECTOR_COLUMNS = _StateVectorColumnLayout()


def _resolved_state_feature_names(
    feature_names: tuple[str, ...],
    observation_values: StateVector | None,
) -> tuple[str, ...]:
    if observation_values is None:
        return feature_names
    return (
        feature_names
        if len(feature_names) == observation_values.size
        else tuple(f"state_{index}" for index in range(observation_values.size))
    )


def _auxiliary_loss_specs(
    policy_config: PolicyConfig | None,
) -> dict[str, _AuxiliaryLossSpec]:
    if policy_config is None:
        return {}
    return {
        loss.name: _AuxiliaryLossSpec(
            name=loss.name,
            grounded_only=bool(loss.grounded_only),
        )
        for loss in policy_config.auxiliary_state.losses
    }


def _state_vector_groups(
    names: tuple[str, ...],
    *,
    auxiliary_loss_names: tuple[str, ...] = (),
) -> tuple[tuple[str, str | None, str | None], ...]:
    component_groups = (
        ("Vehicle", "vehicle_state.", "vehicle_state"),
        ("Machine", "machine_context.", "machine_context"),
        ("Track Position", "track_position.", "track_position"),
        ("Surface", "surface_state.", "surface_state"),
        ("Course", "course_context.", "course_context"),
    )
    used_component_names = {
        name
        for _, prefix, _ in component_groups
        for name in names
        if prefix is not None and name.startswith(prefix)
    }
    groups: tuple[tuple[str, str | None, str | None], ...] = tuple(
        (title, prefix, component_name)
        for title, prefix, component_name in component_groups
        if any(name.startswith(prefix) for name in names)
        or any(_auxiliary_name_matches_group(name, prefix) for name in auxiliary_loss_names)
    )
    if any(name.startswith("control_history.") or name.startswith("prev_") for name in names):
        groups = (*groups, ("Control History", "control_history.", "control_history"))
    legacy_names = tuple(
        name
        for name in names
        if name not in used_component_names
        and not name.startswith("control_history.")
        and not name.startswith("prev_")
    )
    if legacy_names:
        return (*groups, ("State", None, None))
    return groups


def _state_vector_group_lines(
    *,
    names: tuple[str, ...],
    observation_values: StateVector | None,
    reference_values: StateVector | None,
    group_prefix: str | None,
    auxiliary_losses: dict[str, _AuxiliaryLossSpec],
    auxiliary_predictions: dict[str, object] | None,
    auxiliary_targets: dict[str, object] | None,
    zeroed: bool,
    zeroed_features: frozenset[str],
    watch_zeroed_features: frozenset[str],
) -> list[PanelLine]:
    if group_prefix == "course_context.":
        return _course_context_state_lines(
            names=names,
            observation_values=observation_values,
            reference_values=reference_values,
            auxiliary_losses=auxiliary_losses,
            auxiliary_predictions=auxiliary_predictions,
            auxiliary_targets=auxiliary_targets,
            zeroed=zeroed,
            zeroed_features=zeroed_features,
            watch_zeroed_features=watch_zeroed_features,
        )
    if group_prefix == "control_history.":
        if observation_values is None:
            return []
        return _control_history_state_lines(
            names=names,
            observation_values=observation_values,
            reference_values=reference_values,
            show_aux_columns=bool(auxiliary_losses),
            zeroed=zeroed,
            zeroed_features=zeroed_features,
            watch_zeroed_features=watch_zeroed_features,
        )
    lines: list[PanelLine] = []
    matched_aux_names: set[str] = set()
    if observation_values is not None:
        for index, name in enumerate(names):
            if not _state_vector_name_matches_group(name, group_prefix):
                continue
            observation_value = (
                None
                if index >= int(observation_values.size)
                else float(observation_values[index])
            )
            reference_value = (
                observation_value
                if reference_values is None or index >= int(reference_values.size)
                else float(reference_values[index])
            )
            feature_zeroed = _state_vector_entry_zeroed(
                name,
                zeroed=zeroed,
                zeroed_features=zeroed_features,
            )
            aux_loss = auxiliary_losses.get(name)
            if aux_loss is not None:
                matched_aux_names.add(name)
            prediction = (
                None
                if aux_loss is None or auxiliary_predictions is None
                else auxiliary_predictions.get(name)
            )
            target = (
                None
                if aux_loss is None or auxiliary_targets is None
                else auxiliary_targets.get(name)
            )
            lines.append(
                panel_line(
                    _state_vector_line_label(
                        name,
                        group_prefix=group_prefix,
                        zeroed=feature_zeroed,
                        auxiliary_loss=aux_loss,
                    ),
                    _format_state_vector_value(
                        auxiliary_name=None if aux_loss is None else aux_loss.name,
                        show_aux_columns=bool(auxiliary_losses),
                        observation_value=observation_value,
                        reference_value=reference_value,
                        prediction=prediction,
                        target=target,
                    ),
                    _state_vector_line_color(feature_zeroed),
                    status_icon=_state_vector_toggle_icon(
                        name,
                        watch_zeroed_features=watch_zeroed_features,
                    ),
                    click_state_feature_name=name,
                )
            )
    for aux_loss in auxiliary_losses.values():
        if aux_loss.name in matched_aux_names:
            continue
        if not _auxiliary_name_matches_group(aux_loss.name, group_prefix):
            continue
        prediction = (
            None
            if auxiliary_predictions is None
            else auxiliary_predictions.get(aux_loss.name)
        )
        target = None if auxiliary_targets is None else auxiliary_targets.get(aux_loss.name)
        lines.append(
            panel_line(
                _auxiliary_state_line_label(aux_loss),
                _format_state_vector_value(
                    auxiliary_name=aux_loss.name,
                    show_aux_columns=True,
                    observation_value=None,
                    reference_value=None,
                    prediction=prediction,
                    target=target,
                ),
                PALETTE.text_primary if target is not None else PALETTE.text_muted,
                status_icon="none",
            )
        )
    return lines


def _control_history_state_lines(
    *,
    names: tuple[str, ...],
    observation_values: StateVector,
    reference_values: StateVector | None,
    show_aux_columns: bool,
    zeroed: bool,
    zeroed_features: frozenset[str],
    watch_zeroed_features: frozenset[str],
) -> list[PanelLine]:
    return [
        panel_line(
            _state_vector_line_label(
                name,
                group_prefix="control_history.",
                zeroed=_state_vector_entry_zeroed(
                    name,
                    zeroed=zeroed,
                    zeroed_features=zeroed_features,
                ),
            ),
            _format_state_vector_value(
                auxiliary_name=None,
                show_aux_columns=show_aux_columns,
                observation_value=float(observation_values[index]),
                reference_value=(
                    float(reference_values[index])
                    if reference_values is not None and index < int(reference_values.size)
                    else float(observation_values[index])
                ),
                prediction=None,
                target=None,
            ),
            _state_vector_line_color(
                _state_vector_entry_zeroed(
                    name,
                    zeroed=zeroed,
                    zeroed_features=zeroed_features,
                )
            ),
            status_icon=_state_vector_toggle_icon(
                name,
                watch_zeroed_features=watch_zeroed_features,
            ),
            click_state_feature_name=name,
        )
        for index, name in enumerate(names)
        if name.startswith("control_history.") or name.startswith("prev_")
    ]


def _course_context_state_lines(
    *,
    names: tuple[str, ...],
    observation_values: StateVector | None,
    reference_values: StateVector | None,
    auxiliary_losses: dict[str, _AuxiliaryLossSpec],
    auxiliary_predictions: dict[str, object] | None,
    auxiliary_targets: dict[str, object] | None,
    zeroed: bool,
    zeroed_features: frozenset[str],
    watch_zeroed_features: frozenset[str],
) -> list[PanelLine]:
    if observation_values is None:
        course_bits = []
    else:
        course_bits = [
            float(value)
            for name, value in zip(names, observation_values, strict=True)
            if name.startswith("course_context.course_builtin_")
        ]
    lines: list[PanelLine] = []
    if course_bits:
        observation_index = _one_hot_active_index(course_bits)
        encoded_bits = "".join("1" if value >= 0.5 else "0" for value in course_bits)
        feature_zeroed = _state_vector_entry_zeroed(
            "course_context",
            zeroed=zeroed,
            zeroed_features=zeroed_features,
        )
        course_value = (
            f"-- | {encoded_bits}"
            if observation_index is None
            else f"{observation_index} | {encoded_bits}"
        )
        lines.append(
            panel_line(
                _zeroed_label("course", zeroed=feature_zeroed),
                course_value,
                _state_vector_line_color(feature_zeroed),
                status_icon=_state_vector_toggle_icon(
                    "course_context",
                    watch_zeroed_features=watch_zeroed_features,
                ),
                click_state_feature_name="course_context",
            )
        )

    aux_loss = auxiliary_losses.get("course_context.builtin_course_id")
    if aux_loss is None:
        return lines
    prediction = (
        None if auxiliary_predictions is None else auxiliary_predictions.get(aux_loss.name)
    )
    target = None if auxiliary_targets is None else auxiliary_targets.get(aux_loss.name)
    lines.append(
        panel_line(
            _auxiliary_state_line_label(aux_loss),
            _format_state_vector_value(
                auxiliary_name=aux_loss.name,
                show_aux_columns=True,
                observation_value=None,
                reference_value=None,
                prediction=prediction,
                target=target,
            ),
            PALETTE.text_primary if target is not None else PALETTE.text_muted,
            status_icon="none",
        )
    )
    return lines


def _one_hot_active_index(values: list[float]) -> int | None:
    active_indices = [index for index, value in enumerate(values) if value >= 0.5]
    if len(active_indices) == 1:
        return active_indices[0]
    return None


def _state_vector_name_matches_group(name: str, group_prefix: str | None) -> bool:
    if group_prefix is None:
        return "." not in name and not name.startswith("prev_")
    return name.startswith(group_prefix)


def _state_vector_label(name: str, *, group_prefix: str | None) -> str:
    if group_prefix is None:
        return name
    if group_prefix == "control_history." and name.startswith("prev_"):
        return name
    return name.removeprefix(group_prefix)


def _state_vector_group_title(title: str, zeroed: bool) -> str:
    return f"// {title}" if zeroed else title


def _state_vector_entry_zeroed(
    name: str,
    *,
    zeroed: bool,
    zeroed_features: frozenset[str],
) -> bool:
    return zeroed or name in zeroed_features


def _state_vector_line_label(
    name: str,
    *,
    group_prefix: str | None,
    zeroed: bool,
    auxiliary_loss: _AuxiliaryLossSpec | None = None,
) -> str:
    base_label = _state_vector_label(name, group_prefix=group_prefix)
    if auxiliary_loss is not None:
        base_label = _auxiliary_state_display_label(base_label, auxiliary_loss)
    return _zeroed_label(base_label, zeroed=zeroed)


def _auxiliary_state_line_label(auxiliary_loss: _AuxiliaryLossSpec) -> str:
    full_name = auxiliary_loss.name
    if full_name == "course_context.builtin_course_id":
        return "course id"
    if "." in full_name:
        suffix = full_name.split(".", maxsplit=1)[1]
    else:
        suffix = full_name
    return _auxiliary_state_display_label(suffix, auxiliary_loss)


def _auxiliary_state_display_label(base_label: str, auxiliary_loss: _AuxiliaryLossSpec) -> str:
    if auxiliary_loss.grounded_only:
        return f"{base_label} (ground)"
    return base_label


def _auxiliary_name_matches_group(name: str, group_prefix: str | None) -> bool:
    if group_prefix is None:
        return "." not in name
    return name.startswith(group_prefix)


def _format_state_vector_value(
    *,
    auxiliary_name: AuxiliaryStateTargetName | None,
    show_aux_columns: bool,
    observation_value: float | None,
    reference_value: float | None,
    prediction: object,
    target: object,
) -> str:
    if prediction is None and target is None:
        if observation_value is None:
            return "--"
        if not show_aux_columns:
            return f"{observation_value:.3f}"
        return (
            f"{_blank_pred_slot()} | "
            f"{_blank_ref_slot()} | "
            f"{_blank_error_slot()} | "
            f"{_format_obs_scalar_slot(observation_value)}"
        )
    if isinstance(prediction, dict) or isinstance(target, dict):
        return _format_categorical_auxiliary_value(
            prediction,
            target,
            observation_value=observation_value,
        )

    predicted_scalar = _float_or_none(prediction)
    target_scalar = _float_or_none(target if target is not None else reference_value)
    if (
        auxiliary_name is not None
        and resolve_auxiliary_state_target(auxiliary_name).kind == "binary"
    ):
        return _format_binary_auxiliary_value(
            predicted=predicted_scalar,
            target=target_scalar,
            observation_value=observation_value,
        )
    predicted_text = _format_aux_scalar_slot(predicted_scalar)
    reference_text = _format_ref_scalar_slot(target_scalar)
    error_text = _format_aux_percent_slot(
        auxiliary_name=auxiliary_name,
        predicted=predicted_scalar,
        target=target_scalar,
    )
    observation_text = _format_obs_scalar_slot(observation_value)
    return f"{predicted_text} | {reference_text} | {error_text} | {observation_text}"


def _format_categorical_auxiliary_value(
    prediction: object,
    target: object,
    *,
    observation_value: float | None,
) -> str:
    predicted_index = None
    predicted_confidence = None
    if isinstance(prediction, dict):
        raw_index = prediction.get("index")
        raw_confidence = prediction.get("confidence")
        if isinstance(raw_index, int):
            predicted_index = raw_index
        if isinstance(raw_confidence, int | float):
            predicted_confidence = float(raw_confidence)

    target_index = None
    if isinstance(target, dict):
        raw_index = target.get("index")
        if isinstance(raw_index, int):
            target_index = raw_index

    predicted_text = _blank_pred_slot()
    if predicted_index is not None and predicted_confidence is not None:
        predicted_text = f"{predicted_index:>2d}@{predicted_confidence * 100.0:02.0f}%"
    elif predicted_index is not None:
        predicted_text = f"{predicted_index:>{STATE_VECTOR_COLUMNS.pred_width}d}"
    target_text = (
        _blank_ref_slot()
        if target_index is None
        else f"{target_index:>{STATE_VECTOR_COLUMNS.ref_width}d}"
    )
    match_text = _blank_error_slot()
    if predicted_index is not None and target_index is not None:
        match_text = " hit " if predicted_index == target_index else "miss "
    return (
        f"{predicted_text} | "
        f"{target_text} | "
        f"{match_text} | "
        f"{_format_obs_scalar_slot(observation_value)}"
    )


def _format_binary_auxiliary_value(
    *,
    predicted: float | None,
    target: float | None,
    observation_value: float | None,
) -> str:
    predicted_text = _format_aux_scalar_slot(predicted)
    target_text = _format_ref_scalar_slot(target)
    match_text = _blank_error_slot()
    if predicted is not None and target is not None:
        predicted_positive = predicted >= 0.5
        target_positive = target >= 0.5
        match_text = " hit " if predicted_positive == target_positive else "miss "
    return (
        f"{predicted_text} | "
        f"{target_text} | "
        f"{match_text} | "
        f"{_format_obs_scalar_slot(observation_value)}"
    )


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _format_aux_scalar_slot(value: float | None) -> str:
    if value is None:
        return _blank_pred_slot()
    return f"{value:>{STATE_VECTOR_COLUMNS.pred_width}.2f}"


def _format_aux_percent_slot(
    *,
    auxiliary_name: AuxiliaryStateTargetName | None,
    predicted: float | None,
    target: float | None,
) -> str:
    if predicted is None or target is None or auxiliary_name is None:
        return _blank_error_slot()
    low, high = auxiliary_state_target_bounds(auxiliary_name)
    span = max(high - low, 1e-9)
    percent = abs(predicted - target) / span * 100.0
    return f"{percent:>{STATE_VECTOR_COLUMNS.err_width - 1}.0f}%"


def _format_ref_scalar_slot(value: float | None) -> str:
    if value is None:
        return _blank_ref_slot()
    return f"{value:>{STATE_VECTOR_COLUMNS.ref_width}.2f}"


def _format_obs_scalar_slot(value: float | None) -> str:
    if value is None:
        return _blank_obs_slot()
    return f"{value:>{STATE_VECTOR_COLUMNS.obs_width}.2f}"


def _blank_pred_slot() -> str:
    return " " * STATE_VECTOR_COLUMNS.pred_width


def _blank_error_slot() -> str:
    return " " * STATE_VECTOR_COLUMNS.err_width


def _blank_ref_slot() -> str:
    return " " * STATE_VECTOR_COLUMNS.ref_width


def _blank_obs_slot() -> str:
    return " " * STATE_VECTOR_COLUMNS.obs_width


def _state_vector_header_value() -> str:
    return (
        f"{'pred':>{STATE_VECTOR_COLUMNS.pred_width}} | "
        f"{'ref':>{STATE_VECTOR_COLUMNS.ref_width}} | "
        f"{'err':>{STATE_VECTOR_COLUMNS.err_width}} | "
        f"{'obs':>{STATE_VECTOR_COLUMNS.obs_width}}"
    )


def _zeroed_label(label: str, *, zeroed: bool) -> str:
    return f"// {label}" if zeroed else label


def _state_vector_line_color(zeroed: bool):
    return PALETTE.text_muted if zeroed else PALETTE.text_primary


def _state_vector_toggle_icon(
    name: str,
    *,
    watch_zeroed_features: frozenset[str],
) -> StatusIcon:
    return "toggle_off" if name in watch_zeroed_features else "toggle_on"
