# tests/ui/test_viewer_macro_legend.py
"""Watch viewer tests for keyboard legend content and wrapping.

The legend replaces older side-panel key lines, so these tests keep shortcut
labels, controller aliases, and wrapping behavior stable.
"""

import numpy as np

from fzerox_emulator import RaceControlState
from rl_fzerox.ui.watch.view.components.macro_legend import (
    MACRO_LEGEND_HINTS,
    _macro_legend_rows,
)
from rl_fzerox.ui.watch.view.panels.core.model import _build_panel_columns
from tests.ui.viewer_support import fake_viewer_fonts
from tests.ui.viewer_support import sample_telemetry as _sample_telemetry


def test_macro_legend_includes_toggle_anchor_hotkey() -> None:
    hints = {hint.keys: hint.action for hint in MACRO_LEGEND_HINTS}

    assert hints["F"] == "toggle anchor"
    assert hints["C"] == "cnn mode"
    assert hints["Tab / 1-9"] == "tabs"


def test_macro_legend_replaces_side_panel_key_lines() -> None:
    columns = _build_panel_columns(
        episode=0,
        info={},
        reset_info={},
        episode_reward=0.0,
        paused=False,
        control_state=RaceControlState(),
        policy_action=np.array([2, 1, 0], dtype=np.int64),
        policy_reload_age_seconds=5.0,
        policy_reload_error=None,
        action_repeat=2,
        stuck_min_speed_kph=50.0,
        game_display_size=(592, 444),
        observation_shape=(84, 116, 12),
        telemetry=_sample_telemetry(),
    )

    runtime_section = next(section for section in columns.left if section.title == "Runtime")
    key_map = {line.label: line.value for line in runtime_section.lines}
    hint_map = {hint.keys: (hint.controller, hint.action) for hint in MACRO_LEGEND_HINTS}

    assert "Keys" not in key_map
    assert "More keys" not in key_map
    assert key_map["Game size"] == "444x592"
    assert key_map["Obs size"] == "84x116x12"
    assert hint_map == {
        "Esc": (None, "close"),
        "P": (None, "pause"),
        "N": (None, "step"),
        "R": (None, "same course"),
        "E": (None, "prev course"),
        "T": (None, "next course"),
        "G": (None, "difficulty"),
        "F": (None, "toggle anchor"),
        "K": (None, "save"),
        "M": (None, "manual"),
        "D": (None, "policy"),
        "Tab / 1-9": (None, "tabs"),
        "C": (None, "cnn mode"),
        "+/-": (None, "speed"),
        "0": (None, "reset speed"),
        "Arrow keys": ("stick X/Y", "steer/pitch"),
        "Z": ("A", "accelerate"),
        "X": ("C-down", "air brake"),
        "Space": ("B", "boost"),
        "A": ("Z", "lean left"),
        "S": ("R", "lean right"),
        "Q": (None, "spin left"),
        "W": (None, "spin right"),
        "Enter": ("Start", "start"),
    }


def test_macro_legend_wraps_inside_preview_column() -> None:
    fonts = fake_viewer_fonts()
    rows = _macro_legend_rows(font=fonts.small, width=568)

    assert len(rows) > 1
    assert tuple(hint for row in rows for hint in row) == MACRO_LEGEND_HINTS
