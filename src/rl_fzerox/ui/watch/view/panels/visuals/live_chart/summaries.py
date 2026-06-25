# src/rl_fzerox/ui/watch/view/panels/visuals/live_chart/summaries.py
"""Text summaries and reference-line data for the Watch live chart tab."""

from __future__ import annotations

from rl_fzerox.ui.watch.live_series import EpisodeLiveSeriesSnapshot, KoStarRewardEvent
from rl_fzerox.ui.watch.view.panels.visuals.live_chart.axis import _format_axis_value
from rl_fzerox.ui.watch.view.panels.visuals.live_chart.elements import _PlotReferenceLine
from rl_fzerox.ui.watch.view.panels.visuals.live_chart.geometry import _float_tuple_stats
from rl_fzerox.ui.watch.view.panels.visuals.live_chart.style import LIVE_CHART_STYLE


def _speed_summary(live_series: EpisodeLiveSeriesSnapshot | None) -> str:
    if live_series is None or not live_series.speed_kph:
        return "now - · avg - · progress -"
    current_speed = live_series.speed_kph[-1]
    average_speed = _episode_average_speed(live_series)
    current_progress = live_series.current_progress * 100.0
    return (
        f"now {current_speed:.1f} km/h · avg {average_speed:.1f} · progress {current_progress:.1f}%"
    )


def _ko_star_events_summary(live_series: EpisodeLiveSeriesSnapshot | None) -> str:
    if live_series is None:
        return "KO stars - · reward events waiting"
    count = (
        "-" if live_series.current_ko_star_count is None else str(live_series.current_ko_star_count)
    )
    if not live_series.ko_star_events:
        return f"KO stars {count} · reward events none"
    event_text = "; ".join(_format_ko_star_event(event) for event in live_series.ko_star_events)
    return f"KO stars {count} · reward events {event_text}"


def _format_ko_star_event(event: KoStarRewardEvent) -> str:
    return (
        f"#{event.env_step} +{event.gained} "
        f"{event.previous_count}->{event.current_count} ({event.reward:+.2f})"
    )


def _speed_reference_lines(
    live_series: EpisodeLiveSeriesSnapshot | None,
) -> tuple[_PlotReferenceLine, ...]:
    if live_series is None or not live_series.speed_kph:
        return ()
    average_speed = _episode_average_speed(live_series)
    return (
        _PlotReferenceLine(
            value=average_speed,
            color=LIVE_CHART_STYLE.average_speed_color,
            label=_format_axis_value(average_speed),
        ),
    )


def _episode_average_speed(live_series: EpisodeLiveSeriesSnapshot) -> float:
    return _float_tuple_stats(live_series.speed_kph).average


def _step_reward_summary(live_series: EpisodeLiveSeriesSnapshot | None) -> str:
    if live_series is None or not live_series.step_rewards:
        return "now - · avg - · max progress -"
    current_reward = live_series.step_rewards[-1]
    average_reward = _float_tuple_stats(live_series.step_rewards).average
    max_progress = live_series.max_progress * 100.0
    return (
        f"now {current_reward:+.4f} · avg {average_reward:+.4f} · max progress {max_progress:.1f}%"
    )


def _multiplier_summary(live_series: EpisodeLiveSeriesSnapshot | None) -> str:
    if live_series is None or not live_series.progress_speed_position_multiplier:
        return "speed - · position - · total -"
    speed = live_series.progress_speed_multiplier[-1]
    position = live_series.position_progress_multiplier[-1]
    total = live_series.progress_speed_position_multiplier[-1]
    return f"speed {speed:.2f}x · position {position:.2f}x · total {total:.2f}x"


def _multiplier_range(live_series: EpisodeLiveSeriesSnapshot | None) -> tuple[float, float] | None:
    if live_series is None:
        return None
    if not live_series.progress_speed_multiplier:
        return None
    speed_stats = _float_tuple_stats(live_series.progress_speed_multiplier)
    position_stats = _float_tuple_stats(live_series.position_progress_multiplier)
    total_stats = _float_tuple_stats(live_series.progress_speed_position_multiplier)
    lower = max(
        0.0,
        min(
            speed_stats.minimum,
            position_stats.minimum,
            total_stats.minimum,
            1.0,
        ),
    )
    upper = max(
        speed_stats.maximum,
        position_stats.maximum,
        total_stats.maximum,
        1.0,
    )
    if lower == upper:
        return (max(0.0, lower - 0.1), upper + 0.1)
    padding = max(0.05, (upper - lower) * 0.1)
    return (max(0.0, lower - padding), upper + padding)


def _edge_ratio_summary(live_series: EpisodeLiveSeriesSnapshot | None) -> str:
    if live_series is None or not live_series.edge_ratio:
        return "now - · max |ratio| -"
    current_ratio = live_series.edge_ratio[-1]
    max_absolute_ratio = _float_tuple_stats(live_series.edge_ratio).max_abs
    return f"now {current_ratio:+.3f} · max |ratio| {max_absolute_ratio:.3f}"


def _outside_edge_excess_ratio_summary(
    live_series: EpisodeLiveSeriesSnapshot | None,
) -> str:
    if live_series is None or not live_series.outside_edge_excess_ratio:
        return "now - · max -"
    current_excess = live_series.outside_edge_excess_ratio[-1]
    max_excess = _float_tuple_stats(live_series.outside_edge_excess_ratio).maximum
    return f"now {current_excess:.3f} · max {max_excess:.3f}"


def _height_above_ground_summary(live_series: EpisodeLiveSeriesSnapshot | None) -> str:
    if live_series is None or not live_series.height_above_ground:
        return "now - · min - · max -"
    current_height = live_series.height_above_ground[-1]
    height_stats = _float_tuple_stats(live_series.height_above_ground)
    return (
        f"now {current_height:.1f} · min {height_stats.minimum:.1f} "
        f"· max {height_stats.maximum:.1f}"
    )


def _height_above_ground_reference_lines(
    live_series: EpisodeLiveSeriesSnapshot | None,
) -> tuple[_PlotReferenceLine, ...]:
    if live_series is None or not live_series.height_above_ground:
        return ()
    min_height = _float_tuple_stats(live_series.height_above_ground).minimum
    return (
        _PlotReferenceLine(
            value=min_height,
            color=LIVE_CHART_STYLE.min_height_color,
            label=_format_axis_value(min_height),
        ),
    )
