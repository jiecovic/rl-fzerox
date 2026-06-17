# src/rl_fzerox/ui/watch/view/panels/content/state_vector_panel/formatting.py
from __future__ import annotations

from rl_fzerox.core.policy.auxiliary_state.targets import (
    AuxiliaryStateTargetName,
    auxiliary_state_target_bounds,
    resolve_auxiliary_state_target,
)
from rl_fzerox.ui.watch.view.panels.content.state_vector_panel.model import (
    STATE_VECTOR_COLUMNS,
)


def format_state_vector_value(
    *,
    feature_name: str | None = None,
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
            return _format_obs_scalar_value(observation_value, feature_name=feature_name)
        return (
            f"{blank_pred_slot()} | "
            f"{blank_ref_slot()} | "
            f"{blank_error_slot()} | "
            f"{format_obs_scalar_slot(observation_value, feature_name=feature_name)}"
        )
    if isinstance(prediction, dict) or isinstance(target, dict):
        return _format_categorical_auxiliary_value(
            prediction,
            target,
            observation_value=observation_value,
            feature_name=feature_name,
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
            feature_name=feature_name,
        )
    predicted_text = _format_aux_scalar_slot(predicted_scalar)
    reference_text = format_ref_scalar_slot(target_scalar)
    error_text = _format_aux_percent_slot(
        auxiliary_name=auxiliary_name,
        predicted=predicted_scalar,
        target=target_scalar,
    )
    observation_text = format_obs_scalar_slot(observation_value, feature_name=feature_name)
    return f"{predicted_text} | {reference_text} | {error_text} | {observation_text}"


def state_vector_header_value() -> str:
    return (
        f"{'pred':>{STATE_VECTOR_COLUMNS.pred_width}} | "
        f"{'ref':>{STATE_VECTOR_COLUMNS.ref_width}} | "
        f"{'err':>{STATE_VECTOR_COLUMNS.err_width}} | "
        f"{'obs':>{STATE_VECTOR_COLUMNS.obs_width}}"
    )


def format_ref_scalar_slot(value: float | None) -> str:
    if value is None:
        return blank_ref_slot()
    return f"{value:>{STATE_VECTOR_COLUMNS.ref_width}.4f}"


def format_obs_scalar_slot(value: float | None, *, feature_name: str | None = None) -> str:
    if value is None:
        return blank_obs_slot()
    formatted = _format_obs_scalar_value(value, feature_name=feature_name)
    return f"{formatted:>{STATE_VECTOR_COLUMNS.obs_width}}"


def blank_pred_slot() -> str:
    return " " * STATE_VECTOR_COLUMNS.pred_width


def blank_error_slot() -> str:
    return " " * STATE_VECTOR_COLUMNS.err_width


def blank_ref_slot() -> str:
    return " " * STATE_VECTOR_COLUMNS.ref_width


def blank_obs_slot() -> str:
    return " " * STATE_VECTOR_COLUMNS.obs_width


def _format_categorical_auxiliary_value(
    prediction: object,
    target: object,
    *,
    observation_value: float | None,
    feature_name: str | None,
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

    predicted_text = blank_pred_slot()
    if predicted_index is not None and predicted_confidence is not None:
        predicted_text = f"{predicted_index:>2d}@{predicted_confidence * 100.0:02.0f}%"
    elif predicted_index is not None:
        predicted_text = f"{predicted_index:>{STATE_VECTOR_COLUMNS.pred_width}d}"
    target_text = (
        blank_ref_slot()
        if target_index is None
        else f"{target_index:>{STATE_VECTOR_COLUMNS.ref_width}d}"
    )
    match_text = blank_error_slot()
    if predicted_index is not None and target_index is not None:
        match_text = " hit " if predicted_index == target_index else "miss "
    return (
        f"{predicted_text} | "
        f"{target_text} | "
        f"{match_text} | "
        f"{format_obs_scalar_slot(observation_value, feature_name=feature_name)}"
    )


def _format_binary_auxiliary_value(
    *,
    predicted: float | None,
    target: float | None,
    observation_value: float | None,
    feature_name: str | None,
) -> str:
    predicted_text = _format_aux_scalar_slot(predicted)
    target_text = format_ref_scalar_slot(target)
    match_text = blank_error_slot()
    if predicted is not None and target is not None:
        predicted_positive = predicted >= 0.5
        target_positive = target >= 0.5
        match_text = " hit " if predicted_positive == target_positive else "miss "
    return (
        f"{predicted_text} | "
        f"{target_text} | "
        f"{match_text} | "
        f"{format_obs_scalar_slot(observation_value, feature_name=feature_name)}"
    )


def _format_obs_scalar_value(value: float, *, feature_name: str | None) -> str:
    del feature_name
    return f"{value:.4f}"


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _format_aux_scalar_slot(value: float | None) -> str:
    if value is None:
        return blank_pred_slot()
    return f"{value:>{STATE_VECTOR_COLUMNS.pred_width}.4f}"


def _format_aux_percent_slot(
    *,
    auxiliary_name: AuxiliaryStateTargetName | None,
    predicted: float | None,
    target: float | None,
) -> str:
    if predicted is None or target is None or auxiliary_name is None:
        return blank_error_slot()
    low, high = auxiliary_state_target_bounds(auxiliary_name)
    span = max(high - low, 1e-9)
    percent = abs(predicted - target) / span * 100.0
    return f"{percent:>{STATE_VECTOR_COLUMNS.err_width - 1}.0f}%"
