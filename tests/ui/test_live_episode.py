# tests/ui/test_live_episode.py
from __future__ import annotations

from dataclasses import dataclass

import pytest

from rl_fzerox.ui.watch.view.live_episode import EpisodeLiveSeriesTracker
from rl_fzerox.ui.watch.view.panels.visuals.live import (
    _plot_legend_rows,
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
