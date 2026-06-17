# src/rl_fzerox/ui/watch/view/panels/content/state_vector_panel/sections.py
from __future__ import annotations

from fzerox_emulator.arrays import StateVector
from rl_fzerox.core.runtime_spec.schema import PolicyConfig
from rl_fzerox.ui.watch.view.panels.content.state_vector_panel.formatting import (
    format_state_vector_value,
    state_vector_header_value,
)
from rl_fzerox.ui.watch.view.panels.content.state_vector_panel.model import (
    AuxiliaryLossSpec,
    auxiliary_loss_specs,
    auxiliary_name_matches_group,
    flattened_state_vector,
    is_control_history_feature,
    resolved_state_feature_names,
    state_feature_rows,
    state_vector_groups,
    state_vector_label,
)
from rl_fzerox.ui.watch.view.panels.core.lines import (
    panel_divider,
    panel_heading,
    panel_line,
)
from rl_fzerox.ui.watch.view.screen.theme import PALETTE, Color
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
    aux_losses = auxiliary_loss_specs(policy_config)
    if observation_state is None and not aux_losses:
        return []

    observation_values = flattened_state_vector(observation_state)
    reference_values = flattened_state_vector(observation_state_reference)
    names = resolved_state_feature_names(feature_names, observation_values)
    section_lines: list[PanelLine] = []
    if aux_losses:
        section_lines.append(
            panel_line(
                " ",
                state_vector_header_value(),
                PALETTE.text_muted,
                status_icon="none",
            )
        )
    for group in state_vector_groups(names, auxiliary_loss_names=tuple(aux_losses)):
        group_zeroed = group.component_name is not None and group.component_name in zeroed_features
        group_lines = _state_vector_group_lines(
            names=names,
            observation_values=observation_values,
            reference_values=reference_values,
            group_prefix=group.prefix,
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
                panel_heading(_state_vector_group_title(group.title, group_zeroed))
            )
            section_lines.extend(group_lines)
    if not section_lines:
        return []
    return [PanelSection(title="State Vector", lines=section_lines)]


def _state_vector_group_lines(
    *,
    names: tuple[str, ...],
    observation_values: StateVector | None,
    reference_values: StateVector | None,
    group_prefix: str | None,
    auxiliary_losses: dict[str, AuxiliaryLossSpec],
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
            auxiliary_losses=auxiliary_losses,
            auxiliary_predictions=auxiliary_predictions,
            auxiliary_targets=auxiliary_targets,
            zeroed=zeroed,
            zeroed_features=zeroed_features,
            watch_zeroed_features=watch_zeroed_features,
        )
    if group_prefix == "control_history.":
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
    for row in state_feature_rows(
        names=names,
        observation_values=observation_values,
        reference_values=reference_values,
        group_prefix=group_prefix,
    ):
        feature_zeroed = _state_vector_entry_zeroed(
            row.name,
            zeroed=zeroed,
            zeroed_features=zeroed_features,
        )
        aux_loss = auxiliary_losses.get(row.name)
        if aux_loss is not None:
            matched_aux_names.add(row.name)
        prediction = _auxiliary_value(auxiliary_predictions, aux_loss)
        target = _auxiliary_value(auxiliary_targets, aux_loss)
        lines.append(
            panel_line(
                _state_vector_line_label(
                    row.name,
                    group_prefix=group_prefix,
                    zeroed=feature_zeroed,
                    auxiliary_loss=aux_loss,
                ),
                format_state_vector_value(
                    feature_name=row.name,
                    auxiliary_name=None if aux_loss is None else aux_loss.name,
                    show_aux_columns=bool(auxiliary_losses),
                    observation_value=row.observation_value,
                    reference_value=row.reference_value,
                    prediction=prediction,
                    target=target,
                ),
                _state_vector_line_color(feature_zeroed),
                status_icon=_state_vector_toggle_icon(
                    row.name,
                    watch_zeroed_features=watch_zeroed_features,
                ),
                click_state_feature_name=row.name,
            )
        )
    lines.extend(
        _unmatched_auxiliary_lines(
            group_prefix=group_prefix,
            matched_aux_names=matched_aux_names,
            auxiliary_losses=auxiliary_losses,
            auxiliary_predictions=auxiliary_predictions,
            auxiliary_targets=auxiliary_targets,
        )
    )
    return lines


def _control_history_state_lines(
    *,
    names: tuple[str, ...],
    observation_values: StateVector | None,
    reference_values: StateVector | None,
    show_aux_columns: bool,
    zeroed: bool,
    zeroed_features: frozenset[str],
    watch_zeroed_features: frozenset[str],
) -> list[PanelLine]:
    if observation_values is None:
        return []
    return [
        panel_line(
            _state_vector_line_label(
                row.name,
                group_prefix="control_history.",
                zeroed=_state_vector_entry_zeroed(
                    row.name,
                    zeroed=zeroed,
                    zeroed_features=zeroed_features,
                ),
            ),
            format_state_vector_value(
                feature_name=row.name,
                auxiliary_name=None,
                show_aux_columns=show_aux_columns,
                observation_value=row.observation_value,
                reference_value=row.reference_value,
                prediction=None,
                target=None,
            ),
            _state_vector_line_color(
                _state_vector_entry_zeroed(
                    row.name,
                    zeroed=zeroed,
                    zeroed_features=zeroed_features,
                )
            ),
            status_icon=_state_vector_toggle_icon(
                row.name,
                watch_zeroed_features=watch_zeroed_features,
            ),
            click_state_feature_name=row.name,
        )
        for row in state_feature_rows(
            names=names,
            observation_values=observation_values,
            reference_values=reference_values,
            group_prefix="control_history.",
        )
        if is_control_history_feature(row.name)
    ]


def _course_context_state_lines(
    *,
    names: tuple[str, ...],
    observation_values: StateVector | None,
    auxiliary_losses: dict[str, AuxiliaryLossSpec],
    auxiliary_predictions: dict[str, object] | None,
    auxiliary_targets: dict[str, object] | None,
    zeroed: bool,
    zeroed_features: frozenset[str],
    watch_zeroed_features: frozenset[str],
) -> list[PanelLine]:
    course_bits = _course_one_hot_bits(names=names, observation_values=observation_values)
    lines: list[PanelLine] = []
    if course_bits:
        feature_zeroed = _state_vector_entry_zeroed(
            "course_context",
            zeroed=zeroed,
            zeroed_features=zeroed_features,
        )
        lines.append(
            panel_line(
                zeroed_label("course", zeroed=feature_zeroed),
                _course_one_hot_value(course_bits),
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
    lines.append(
        _auxiliary_line(
            aux_loss,
            prediction=_auxiliary_value(auxiliary_predictions, aux_loss),
            target=_auxiliary_value(auxiliary_targets, aux_loss),
        )
    )
    return lines


def _course_one_hot_bits(
    *,
    names: tuple[str, ...],
    observation_values: StateVector | None,
) -> tuple[float, ...]:
    if observation_values is None:
        return ()
    return tuple(
        float(value)
        for name, value in zip(names, observation_values, strict=True)
        if name.startswith("course_context.course_builtin_")
    )


def _course_one_hot_value(course_bits: tuple[float, ...]) -> str:
    observation_index = _one_hot_active_index(course_bits)
    encoded_bits = "".join("1" if value >= 0.5 else "0" for value in course_bits)
    if observation_index is None:
        return f"-- | {encoded_bits}"
    return f"{observation_index} | {encoded_bits}"


def _one_hot_active_index(values: tuple[float, ...]) -> int | None:
    active_indices = [index for index, value in enumerate(values) if value >= 0.5]
    if len(active_indices) == 1:
        return active_indices[0]
    return None


def _unmatched_auxiliary_lines(
    *,
    group_prefix: str | None,
    matched_aux_names: set[str],
    auxiliary_losses: dict[str, AuxiliaryLossSpec],
    auxiliary_predictions: dict[str, object] | None,
    auxiliary_targets: dict[str, object] | None,
) -> list[PanelLine]:
    lines: list[PanelLine] = []
    for aux_loss in auxiliary_losses.values():
        if aux_loss.name in matched_aux_names:
            continue
        if not auxiliary_name_matches_group(aux_loss.name, group_prefix):
            continue
        lines.append(
            _auxiliary_line(
                aux_loss,
                prediction=_auxiliary_value(auxiliary_predictions, aux_loss),
                target=_auxiliary_value(auxiliary_targets, aux_loss),
            )
        )
    return lines


def _auxiliary_line(
    aux_loss: AuxiliaryLossSpec,
    *,
    prediction: object,
    target: object,
) -> PanelLine:
    return panel_line(
        _auxiliary_state_line_label(aux_loss),
        format_state_vector_value(
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


def _auxiliary_value(
    values: dict[str, object] | None,
    aux_loss: AuxiliaryLossSpec | None,
) -> object:
    if values is None or aux_loss is None:
        return None
    return values.get(aux_loss.name)


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
    auxiliary_loss: AuxiliaryLossSpec | None = None,
) -> str:
    base_label = state_vector_label(name, group_prefix=group_prefix)
    if auxiliary_loss is not None:
        base_label = _auxiliary_state_display_label(base_label, auxiliary_loss)
    return zeroed_label(base_label, zeroed=zeroed)


def _auxiliary_state_line_label(auxiliary_loss: AuxiliaryLossSpec) -> str:
    full_name = auxiliary_loss.name
    if full_name == "course_context.builtin_course_id":
        return "course id"
    suffix = full_name.split(".", maxsplit=1)[1] if "." in full_name else full_name
    return _auxiliary_state_display_label(suffix, auxiliary_loss)


def _auxiliary_state_display_label(base_label: str, auxiliary_loss: AuxiliaryLossSpec) -> str:
    if auxiliary_loss.grounded_only:
        return f"{base_label} (ground)"
    return base_label


def zeroed_label(label: str, *, zeroed: bool) -> str:
    return f"// {label}" if zeroed else label


def _state_vector_line_color(zeroed: bool) -> Color:
    return PALETTE.text_muted if zeroed else PALETTE.text_primary


def _state_vector_toggle_icon(
    name: str,
    *,
    watch_zeroed_features: frozenset[str],
) -> StatusIcon:
    return "toggle_off" if name in watch_zeroed_features else "toggle_on"
