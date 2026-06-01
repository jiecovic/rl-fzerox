# tests/ui/test_live_episode.py
from __future__ import annotations

from dataclasses import dataclass

import pytest

from rl_fzerox.ui.watch.live_series import LIVE_SERIES_LIMITS, EpisodeLiveSeriesTracker
from rl_fzerox.ui.watch.view.panels.visuals.live import (
    LIVE_CHART_STYLE,
    _ko_star_events_summary,
    _plot_legend_rows,
    _plot_points,
    _PlotSeries,
    _speed_summary,
)
from rl_fzerox.ui.watch.view.screen.types import ViewerFonts


@dataclass(frozen=True)
class _Snapshot:
    episode: int
    policy_decision_frame: bool
    info: dict[str, object]
    episode_reward: float
    telemetry_data: dict[str, object] | None = None


@dataclass(frozen=True)
class _Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class _TextSurface:
    width: int
    height: int = 10

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height


class _RenderFont:
    def render(
        self,
        text: str,
        antialias: bool,
        color: tuple[int, int, int],
    ) -> _TextSurface:
        _ = (antialias, color)
        return _TextSurface(width=len(text) * 7)


@dataclass(frozen=True)
class _Fonts:
    font: _RenderFont

    def viewer_fonts(self) -> ViewerFonts:
        return ViewerFonts(
            title=self.font,
            section=self.font,
            record_header=self.font,
            body=self.font,
            small=self.font,
        )


def test_live_episode_tracker_uses_decision_frames_only() -> None:
    tracker = EpisodeLiveSeriesTracker()

    tracker.observe_snapshot(
        _Snapshot(
            episode=3,
            policy_decision_frame=False,
            info={
                "episode_step": 5,
                "episode_completion_fraction": 0.2,
                "speed_kph": 300.0,
            },
            episode_reward=1.0,
        ),
        action_repeat=2,
    )
    tracker.observe_snapshot(
        _Snapshot(
            episode=3,
            policy_decision_frame=True,
            info={
                "episode_step": 6,
                "episode_completion_fraction": 0.25,
                "speed_kph": 420.0,
                "step_reward": 0.75,
                "progress_speed_multiplier": 0.95,
                "position_progress_multiplier": 1.1,
                "progress_speed_position_multiplier": 1.045,
            },
            episode_reward=1.5,
            telemetry_data={
                "player": {
                    "lateral_distance": 12.5,
                    "signed_lateral_offset": -8.25,
                    "current_radius_left": 20.0,
                    "current_radius_right": 7.0,
                    "height_above_ground": 42.5,
                }
            },
        ),
        action_repeat=2,
    )

    snapshot = tracker.snapshot()
    assert snapshot is not None
    assert snapshot.env_steps == (3,)
    assert snapshot.speed_kph == (420.0,)
    assert snapshot.step_rewards == (0.75,)
    assert snapshot.progress_speed_multiplier == (0.95,)
    assert snapshot.position_progress_multiplier == (1.1,)
    assert snapshot.progress_speed_position_multiplier == (1.045,)
    assert snapshot.edge_ratio == (-1.0,)
    assert snapshot.outside_edge_excess_ratio == pytest.approx((0.1785714286,))
    assert snapshot.height_above_ground == (42.5,)
    assert snapshot.current_return == 1.5
    assert snapshot.current_progress == 0.25
    assert snapshot.max_progress == 0.25


def test_live_episode_tracker_resets_on_new_episode() -> None:
    tracker = EpisodeLiveSeriesTracker()

    tracker.observe_snapshot(
        _Snapshot(
            episode=1,
            policy_decision_frame=True,
            info={
                "episode_step": 4,
                "episode_completion_fraction": 0.1,
                "speed_kph": 250.0,
            },
            episode_reward=0.5,
        ),
        action_repeat=2,
    )
    tracker.observe_snapshot(
        _Snapshot(
            episode=1,
            policy_decision_frame=True,
            info={
                "episode_step": 6,
                "episode_completion_fraction": 0.2,
                "speed_kph": 320.0,
            },
            episode_reward=0.25,
        ),
        action_repeat=2,
    )
    tracker.observe_snapshot(
        _Snapshot(
            episode=2,
            policy_decision_frame=True,
            info={
                "episode_step": 2,
                "episode_completion_fraction": 0.05,
                "speed_kph": 180.0,
            },
            episode_reward=0.1,
        ),
        action_repeat=2,
    )

    snapshot = tracker.snapshot()
    assert snapshot is not None
    assert snapshot.episode == 2
    assert snapshot.env_steps == (1,)
    assert snapshot.speed_kph == (180.0,)
    assert snapshot.step_rewards == (0.0,)
    assert snapshot.progress_speed_multiplier == (1.0,)
    assert snapshot.position_progress_multiplier == (1.0,)
    assert snapshot.progress_speed_position_multiplier == (1.0,)
    assert snapshot.edge_ratio == (0.0,)
    assert snapshot.outside_edge_excess_ratio == (0.0,)
    assert snapshot.height_above_ground == (0.0,)
    assert snapshot.current_return == 0.1
    assert snapshot.current_progress == 0.05
    assert snapshot.max_progress == 0.05


def test_speed_summary_reports_current_episode_average() -> None:
    tracker = EpisodeLiveSeriesTracker()
    for step, speed in ((2, 300.0), (4, 600.0), (6, 900.0)):
        tracker.observe_snapshot(
            _Snapshot(
                episode=1,
                policy_decision_frame=True,
                info={
                    "episode_step": step,
                    "episode_completion_fraction": 0.2,
                    "speed_kph": speed,
                },
                episode_reward=0.0,
            ),
            action_repeat=1,
        )

    summary = _speed_summary(tracker.snapshot())

    assert "now 900.0 km/h" in summary
    assert "avg 600.0" in summary


def test_live_episode_tracker_computes_left_edge_excess_ratio() -> None:
    tracker = EpisodeLiveSeriesTracker()

    tracker.observe_snapshot(
        _Snapshot(
            episode=1,
            policy_decision_frame=True,
            info={
                "episode_step": 2,
                "episode_completion_fraction": 0.1,
                "speed_kph": 250.0,
            },
            episode_reward=0.5,
            telemetry_data={
                "player": {
                    "signed_lateral_offset": 15.0,
                    "current_radius_left": 10.0,
                    "current_radius_right": 30.0,
                }
            },
        ),
        action_repeat=1,
    )

    snapshot = tracker.snapshot()
    assert snapshot is not None
    assert snapshot.outside_edge_excess_ratio == (0.5,)


def test_live_episode_tracker_suppresses_edge_margin_spikes() -> None:
    tracker = EpisodeLiveSeriesTracker()

    tracker.observe_snapshot(
        _Snapshot(
            episode=1,
            policy_decision_frame=True,
            info={
                "episode_step": 2,
                "episode_completion_fraction": 0.1,
                "speed_kph": 250.0,
            },
            episode_reward=0.5,
            telemetry_data={
                "player": {
                    "signed_lateral_offset": 10.8,
                    "current_radius_left": 10.0,
                    "current_radius_right": 30.0,
                }
            },
        ),
        action_repeat=1,
    )

    snapshot = tracker.snapshot()
    assert snapshot is not None
    assert snapshot.outside_edge_excess_ratio == (0.0,)


def test_live_episode_tracker_reports_step_reward() -> None:
    tracker = EpisodeLiveSeriesTracker()

    tracker.observe_snapshot(
        _Snapshot(
            episode=1,
            policy_decision_frame=True,
            info={
                "episode_step": 2,
                "episode_completion_fraction": 0.1,
                "speed_kph": 250.0,
                "step_reward": 0.25,
            },
            episode_reward=0.5,
        ),
        action_repeat=1,
    )
    tracker.observe_snapshot(
        _Snapshot(
            episode=1,
            policy_decision_frame=True,
            info={
                "episode_step": 6,
                "episode_completion_fraction": 0.2,
                "speed_kph": 320.0,
                "step_reward": 0.5,
            },
            episode_reward=2.5,
        ),
        action_repeat=1,
    )
    tracker.observe_snapshot(
        _Snapshot(
            episode=1,
            policy_decision_frame=True,
            info={
                "episode_step": 6,
                "episode_completion_fraction": 0.2,
                "speed_kph": 320.0,
                "step_reward": 0.75,
            },
            episode_reward=3.5,
        ),
        action_repeat=1,
    )

    snapshot = tracker.snapshot()
    assert snapshot is not None
    assert snapshot.env_steps == (2, 6)
    assert snapshot.step_rewards == (0.25, 0.75)


def test_live_episode_tracker_bounds_sample_history() -> None:
    tracker = EpisodeLiveSeriesTracker()

    for index in range(LIVE_SERIES_LIMITS.max_samples + 3):
        tracker.observe_decision(
            episode=1,
            info={
                "episode_step": index + 1,
                "step_reward": float(index),
            },
            episode_reward=float(index),
            telemetry_data=None,
            action_repeat=1,
        )

    snapshot = tracker.snapshot()
    assert snapshot is not None
    assert len(snapshot.env_steps) == LIVE_SERIES_LIMITS.max_samples
    assert len(snapshot.step_rewards) == LIVE_SERIES_LIMITS.max_samples
    assert snapshot.env_steps[0] == 4
    assert snapshot.step_rewards[-1] == pytest.approx(float(LIVE_SERIES_LIMITS.max_samples + 2))


def test_live_episode_tracker_reports_exact_ko_reward_events() -> None:
    tracker = EpisodeLiveSeriesTracker()

    tracker.observe_snapshot(
        _Snapshot(
            episode=1,
            policy_decision_frame=True,
            info={
                "episode_step": 4,
                "ko_star_count": 3,
                "ko_star_reward_event": True,
                "ko_star_reward_previous_count": 1,
                "ko_star_reward_current_count": 3,
                "ko_star_reward_gain": 2,
                "ko_star_reward_value": 5.0,
            },
            episode_reward=5.0,
        ),
        action_repeat=2,
    )

    snapshot = tracker.snapshot()

    assert snapshot is not None
    assert snapshot.current_ko_star_count == 3
    assert len(snapshot.ko_star_events) == 1
    event = snapshot.ko_star_events[0]
    assert event.env_step == 2
    assert event.previous_count == 1
    assert event.current_count == 3
    assert event.gained == 2
    assert event.reward == 5.0
    assert _ko_star_events_summary(snapshot) == "KO stars 3 · reward events #2 +2 1->3 (+5.00)"


def test_plot_legend_rows_include_only_labeled_series() -> None:
    rows = _plot_legend_rows(
        fonts=_Fonts(font=_RenderFont()).viewer_fonts(),
        rect_width=220,
        series=(
            _PlotSeries(y_values=(1.0,), color=(1, 2, 3), label="speed"),
            _PlotSeries(y_values=(1.0,), color=(4, 5, 6)),
            _PlotSeries(y_values=(1.0,), color=(7, 8, 9), label="total"),
        ),
    )

    assert tuple(item.label for row in rows for item in row) == ("speed", "total")


def test_plot_points_buckets_preserve_single_sample_spikes() -> None:
    rect = _Rect(x=10, y=20, width=4, height=101)
    x_values = tuple(range(20))
    y_values = tuple(10.0 if index == 5 else 0.0 for index in range(20))

    points = _plot_points(
        x_values=x_values,
        y_values=y_values,
        rect=rect,
        x_start=0,
        x_end=19,
        y_min=0.0,
        span=10.0,
    )

    assert len(points) <= rect.width * 4
    assert (11, 20) in points


def test_plot_points_caps_wide_plot_buckets() -> None:
    rect = _Rect(x=0, y=0, width=1_200, height=101)
    x_values = tuple(range(8_000))
    y_values = tuple(10.0 if index % 997 == 0 else 0.0 for index in x_values)

    points = _plot_points(
        x_values=x_values,
        y_values=y_values,
        rect=rect,
        x_start=0,
        x_end=7_999,
        y_min=0.0,
        span=10.0,
    )

    assert len(points) <= LIVE_CHART_STYLE.plot_max_buckets * 4
