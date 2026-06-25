# src/rl_fzerox/ui/watch/view/screen/render.py
"""Pygame entrypoint for drawing one Watch worker snapshot.

`screen.view_model` converts config and worker snapshot data into
`FrameRenderData`. This module keeps the public render call close to pygame:
resolve the current render FPS, build the view model, and hand it to
`screen.frame`.
"""

from __future__ import annotations

from rl_fzerox.core.runtime_spec.schema import WatchAppConfig
from rl_fzerox.ui.watch.live_series import EpisodeLiveSeriesSnapshot
from rl_fzerox.ui.watch.runtime.ipc import WatchSnapshot
from rl_fzerox.ui.watch.runtime.timing import RateMeter
from rl_fzerox.ui.watch.view.auxiliary_metrics import AuxiliaryEpisodeMetricsSnapshot
from rl_fzerox.ui.watch.view.panels.core.tabs import WATCH_PANEL_TABS, PanelTabRegistry
from rl_fzerox.ui.watch.view.screen.frame import _draw_frame
from rl_fzerox.ui.watch.view.screen.types import (
    PygameModule,
    PygameSurface,
    ViewerFonts,
    ViewerHitboxes,
)
from rl_fzerox.ui.watch.view.screen.view_model import build_frame_render_data


def draw_watch_frame(
    *,
    pygame: PygameModule,
    screen: PygameSurface,
    fonts: ViewerFonts,
    config: WatchAppConfig,
    snapshot: WatchSnapshot,
    paused: bool,
    render_rate: RateMeter,
    target_render_fps: float | None,
    track_pool_records: tuple[dict[str, object], ...] | None = None,
    auxiliary_episode_metrics: AuxiliaryEpisodeMetricsSnapshot | None = None,
    live_episode_series: EpisodeLiveSeriesSnapshot | None = None,
    panel_tabs: PanelTabRegistry = WATCH_PANEL_TABS,
    panel_tab_index: int = 0,
    cnn_layer_tab_index: int = 0,
    record_tab_index: int = 0,
    policy_observation_layout_shape: tuple[int, ...] | None = None,
    policy_observation_layout_info: dict[str, object] | None = None,
) -> ViewerHitboxes:
    """Render one worker state packet without leaking env/policy logic into drawing."""

    return _draw_frame(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        data=build_frame_render_data(
            config=config,
            snapshot=snapshot,
            paused=paused,
            current_render_fps=render_rate.rate_hz(),
            target_render_fps=target_render_fps,
            track_pool_records=track_pool_records,
            auxiliary_episode_metrics=auxiliary_episode_metrics,
            live_episode_series=live_episode_series,
            panel_tabs=panel_tabs,
            panel_tab_index=panel_tab_index,
            cnn_layer_tab_index=cnn_layer_tab_index,
            record_tab_index=record_tab_index,
            policy_observation_layout_shape=policy_observation_layout_shape,
            policy_observation_layout_info=policy_observation_layout_info,
        ),
    )
