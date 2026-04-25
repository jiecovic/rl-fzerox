# src/rl_fzerox/ui/watch/view/screen/layout.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewerLayout:
    """Spacing and sizing used by the watch window layout."""

    game_display_size: tuple[int, int] = (592, 444)
    panel_width: int = 400
    panel_min_height: int = 700
    panel_state_min_height: int = 980
    panel_padding: int = 12
    preview_gap: int = 12
    preview_scale: int = 1
    preview_padding: int = 12
    preview_title_gap: int = 6
    column_gap: int = 16
    title_gap: int = 2
    title_section_gap: int = 8
    section_gap: int = 8
    section_title_gap: int = 4
    section_rule_gap: int = 4
    line_gap: int = 2
    inline_value_gap: int = 8
    wrapped_value_indent: int = 10
    control_viz_gap: int = 3
    control_widget_gap: int = 12
    control_track_gap: int = 4
    control_side_pill_gap: int = 8
    control_steer_width: int = 116
    control_steer_height: int = 14
    control_gas_width: int = 16
    control_gas_height: int = 72
    control_gas_pair_gap: int = 48
    control_gas_offset_x: int = 20
    control_marker_radius: int = 6
    control_caption_gap: int = 3
    control_boost_gap: int = 4
    flag_token_gap: int = 4
    flag_token_pad_x: int = 6
    flag_token_pad_y: int = 2


LAYOUT = ViewerLayout()
