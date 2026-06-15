# src/rl_fzerox/ui/watch/view/panels/core/tabs.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.runtime_spec.schema import WatchAppConfig


@dataclass(frozen=True, slots=True)
class PanelTab:
    """One watch side-panel tab."""

    key: str
    label: str


@dataclass(frozen=True, slots=True)
class PanelTabRegistry:
    """Ordered tab descriptors plus named lookups for special tabs."""

    tabs: tuple[PanelTab, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(tab.label for tab in self.tabs)

    @property
    def count(self) -> int:
        return len(self.tabs)

    @property
    def cnn_index(self) -> int:
        return self.index("cnn")

    @property
    def live_index(self) -> int:
        return self.index("live")

    @property
    def records_index(self) -> int | None:
        return self.optional_index("records")

    @property
    def career_index(self) -> int | None:
        return self.optional_index("career")

    @property
    def state_index(self) -> int:
        return self.index("state")

    @property
    def aux_index(self) -> int:
        return self.index("aux")

    def index(self, key: str) -> int:
        for index, tab in enumerate(self.tabs):
            if tab.key == key:
                return index
        raise ValueError(f"Unknown panel tab key: {key!r}")

    def optional_index(self, key: str) -> int | None:
        for index, tab in enumerate(self.tabs):
            if tab.key == key:
                return index
        return None

    def key(self, index: int) -> str:
        return self.tabs[self.normalize(index)].key

    def normalize(self, index: int) -> int:
        return index % self.count


WATCH_PANEL_TABS = PanelTabRegistry(
    tabs=(
        PanelTab(key="run", label="Run"),
        PanelTab(key="live", label="Live"),
        PanelTab(key="obs", label="Obs"),
        PanelTab(key="details", label="Details"),
        PanelTab(key="state", label="State"),
        PanelTab(key="aux", label="Aux"),
        PanelTab(key="cnn", label="CNN"),
        PanelTab(key="records", label="Records"),
        PanelTab(key="train", label="Train"),
    )
)

CAREER_PANEL_TABS = PanelTabRegistry(
    tabs=(
        PanelTab(key="run", label="Run"),
        PanelTab(key="live", label="Live"),
        PanelTab(key="obs", label="Obs"),
        PanelTab(key="details", label="Details"),
        PanelTab(key="state", label="State"),
        PanelTab(key="aux", label="Aux"),
        PanelTab(key="cnn", label="CNN"),
        PanelTab(key="records", label="Records"),
        PanelTab(key="career", label="Career"),
        PanelTab(key="train", label="Train"),
    )
)

PANEL_TABS = WATCH_PANEL_TABS


def panel_tabs_for_config(config: WatchAppConfig) -> PanelTabRegistry:
    """Return the tab set for the active viewer session controller."""

    if config.watch.managed_save_game_id is not None:
        return CAREER_PANEL_TABS
    return WATCH_PANEL_TABS
