# src/rl_fzerox/ui/watch/view/screen/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

from rl_fzerox.ui.watch.view.screen.theme import Color

PygameModule: TypeAlias = Any
PygameRect: TypeAlias = Any
PygameSurface: TypeAlias = Any
StatusIcon: TypeAlias = Literal["none", "in_range", "outside"]


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
    divider: bool = False
    heading: bool = False
    status_icon: StatusIcon | None = None
    status_text: str = ""


@dataclass(frozen=True)
class ControlViz:
    """Compact control-state visualization for the watch panel."""

    steer_x: float
    pitch_y: float
    gas_level: float
    thrust_warning_threshold: float | None
    thrust_deadzone_threshold: float | None
    thrust_full_threshold: float | None
    engine_setting_level: float | None
    speed_kph: float | None
    energy_fraction: float | None
    air_brake_axis: float | None
    air_brake_disabled: bool
    boost_pressed: bool
    boost_active: bool
    boost_lamp_level: float
    lean_direction: int
    thrust_masked: bool = False
    air_brake_masked: bool = False
    boost_masked: bool = False
    lean_left_masked: bool = False
    lean_right_masked: bool = False
    pitch_masked: bool = False


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
    """Side-panel columns rendered next to the game view."""

    left: list[PanelSection]
    middle: list[PanelSection]
    stats: list[PanelSection]


@dataclass(frozen=True)
class ViewerFonts:
    """Font bundle used by the watch panel renderer."""

    title: RenderFont
    section: RenderFont
    record_header: RenderFont
    body: RenderFont
    small: RenderFont
