# src/rl_fzerox/ui/watch/view/screen/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rl_fzerox.ui.watch.view.screen.theme import Color


class TextSurface(Protocol):
    """Minimal surface contract used by the viewer layout helpers."""

    def get_width(self) -> int: ...

    def get_height(self) -> int: ...


class RenderFont(Protocol):
    """Minimal font contract used by the viewer renderer."""

    def render(
        self,
        text: str,
        antialias: bool,
        color: Color,
    ) -> TextSurface: ...


@dataclass(frozen=True)
class PanelLine:
    """One labeled line in the side panel."""

    label: str
    value: str
    color: Color
    wrap: bool = False
    min_value_lines: int = 1


@dataclass(frozen=True)
class ControlViz:
    """Compact control-state visualization for the watch panel."""

    steer_x: float
    gas_level: float
    thrust_warning_threshold: float | None
    air_brake_axis: float | None
    air_brake_disabled: bool
    boost_pressed: bool
    boost_active: bool
    boost_lamp_level: float
    lean_direction: int


@dataclass(frozen=True)
class FlagToken:
    """One compact flag label rendered inside the watch panel."""

    label: str
    active: bool


@dataclass(frozen=True)
class FlagViz:
    """Fixed multi-row racer flag visualization for the watch panel."""

    rows: tuple[tuple[FlagToken, ...], ...]


@dataclass(frozen=True)
class PanelSection:
    """One titled section in the side panel."""

    title: str
    lines: list[PanelLine]
    control_viz: ControlViz | None = None
    flag_viz: FlagViz | None = None


@dataclass(frozen=True)
class PanelColumns:
    """Left and right panel columns rendered next to the game view."""

    left: list[PanelSection]
    right: list[PanelSection]


@dataclass(frozen=True)
class ViewerFonts:
    """Font bundle used by the watch panel renderer."""

    title: RenderFont
    section: RenderFont
    body: RenderFont
    small: RenderFont
