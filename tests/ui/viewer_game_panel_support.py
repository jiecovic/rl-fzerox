# tests/ui/viewer_game_panel_support.py
from fzerox_emulator import RaceControlState
from rl_fzerox.ui.watch.view.screen.types import PanelColumns, PanelSection


def race_control_state(
    *,
    control_mask: int = 0,
    stick_x: float = 0.0,
    pitch: float = 0.0,
) -> RaceControlState:
    return RaceControlState.from_mask(
        control_mask,
        stick_x=stick_x,
        pitch=pitch,
    )


def _race_section(columns: PanelColumns) -> PanelSection:
    return next(section for section in columns.left if section.title == "Race State")


def _setup_section(columns: PanelColumns) -> PanelSection:
    return next(section for section in columns.left if section.title == "Race Setup")


def _career_section(columns: PanelColumns, title: str) -> PanelSection:
    return next(section for section in columns.career if section.title == title)
