# src/rl_fzerox/ui/watch/view/panels/core/tabs.py
from __future__ import annotations

from dataclasses import dataclass


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
    def records_index(self) -> int:
        return self.index("records")

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

    def key(self, index: int) -> str:
        return self.tabs[self.normalize(index)].key

    def normalize(self, index: int) -> int:
        return index % self.count


PANEL_TABS = PanelTabRegistry(
    tabs=(
        PanelTab(key="run", label="Run"),
        PanelTab(key="live", label="Live"),
        PanelTab(key="details", label="Details"),
        PanelTab(key="state", label="State"),
        PanelTab(key="aux", label="Aux"),
        PanelTab(key="cnn", label="CNN"),
        PanelTab(key="records", label="Records"),
        PanelTab(key="train", label="Train"),
    )
)
