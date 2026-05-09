# src/rl_fzerox/ui/watch/view/panels/content/auxiliary.py
from __future__ import annotations

from rl_fzerox.ui.watch.view.auxiliary_metrics import (
    AuxiliaryEpisodeMetric,
    AuxiliaryEpisodeMetricsSnapshot,
)
from rl_fzerox.ui.watch.view.panels.core.lines import panel_divider, panel_line
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelSection


def auxiliary_episode_sections(
    snapshot: AuxiliaryEpisodeMetricsSnapshot | None,
) -> list[PanelSection]:
    if snapshot is None or not snapshot.metrics:
        return [
            PanelSection(
                title="Auxiliary State",
                lines=[
                    panel_line("Episode", "-", PALETTE.text_muted),
                    panel_line("Metrics", "disabled", PALETTE.text_muted),
                ],
            )
        ]

    lines = [
        panel_line("Episode", str(snapshot.episode), PALETTE.text_primary),
        panel_divider(),
        panel_line(" ", _aux_metric_header_value(), PALETTE.text_muted),
    ]
    grouped_metrics = _grouped_metrics(snapshot.metrics)
    for group_index, (group_name, metrics) in enumerate(grouped_metrics):
        if group_index > 0:
            lines.append(panel_divider())
        lines.append(panel_line(group_name, "", PALETTE.text_primary, heading=True))
        lines.extend(_metric_lines(metrics))
    return [PanelSection(title="Auxiliary State", lines=lines)]


def _grouped_metrics(
    metrics: tuple[AuxiliaryEpisodeMetric, ...],
) -> tuple[tuple[str, tuple[AuxiliaryEpisodeMetric, ...]], ...]:
    groups = (
        ("Vehicle", "vehicle_state."),
        ("Track", "track_position."),
        ("Surface", "surface_state."),
        ("Course", "course_context."),
    )
    return tuple(
        (
            group_name,
            tuple(metric for metric in metrics if metric.name.startswith(prefix)),
        )
        for group_name, prefix in groups
        if any(metric.name.startswith(prefix) for metric in metrics)
    )


def _metric_lines(metrics: tuple[AuxiliaryEpisodeMetric, ...]) -> list:
    return [
        panel_line(
            _metric_label(metric.name),
            _metric_value(metric),
            PALETTE.text_primary if metric.sample_count > 0 else PALETTE.text_muted,
        )
        for metric in metrics
    ]


def _metric_label(name: str) -> str:
    if name == "course_context.builtin_course_id":
        return "course id"
    _, suffix = name.split(".", maxsplit=1)
    return suffix


def _metric_value(metric: AuxiliaryEpisodeMetric) -> str:
    if metric.sample_count <= 0:
        return (
            f"{'-':>5} | {_kind_slot('n/a')} | {_dash_stat_slot()} | "
            f"{_blank_conf_slot()} | {_count_slot(0)}"
        )
    if metric.accuracy is not None:
        return (
            f"{metric.mean_loss:>5.3f} | "
            f"{_kind_slot('acc')} | "
            f"{_percent_slot(metric.accuracy * 100.0)} | "
            f"{_conf_slot(metric.mean_confidence)} | "
            f"{_count_slot(metric.sample_count)}"
        )
    if metric.mean_error_percent is not None:
        return (
            f"{metric.mean_loss:>5.3f} | "
            f"{_kind_slot('err')} | "
            f"{_percent_slot(metric.mean_error_percent)} | "
            f"{_dash_conf_slot()} | "
            f"{_count_slot(metric.sample_count)}"
        )
    return (
        f"{metric.mean_loss:>5.3f} | {'n/a':>4} | {'-':>5} | {'-':>5} | "
        f"{_count_slot(metric.sample_count)}"
    )


def _aux_metric_header_value() -> str:
    return f"{'loss':>5} | {'kind':>4} | {'stat':>5} | {'conf':>5} | {'cnt':>5}"


def _kind_slot(label: str) -> str:
    return f"{label:>4}"


def _percent_slot(value: float) -> str:
    return f"{value:>4.0f}%"


def _dash_stat_slot() -> str:
    return f"{'-':>5}"


def _conf_slot(value: float | None) -> str:
    if value is None:
        return _blank_conf_slot()
    return f"{value * 100.0:>4.0f}%"


def _blank_conf_slot() -> str:
    return " " * 5


def _dash_conf_slot() -> str:
    return f"{'-':>5}"


def _count_slot(value: int) -> str:
    if value < 10_000:
        return f"{value:>5d}"
    if value < 100_000:
        return f"{value / 1_000:>4.1f}k"
    if value < 1_000_000:
        return f"{value // 1_000:>4d}k"
    if value < 10_000_000:
        return f"{value / 1_000_000:>4.1f}m"
    return f"{value // 1_000_000:>4d}m"
