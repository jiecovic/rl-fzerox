# src/rl_fzerox/ui/watch/view/components/cockpit/steer.py
"""Compatibility facade for cockpit steering and pitch instruments."""

from rl_fzerox.ui.watch.view.components.cockpit.axis import (
    draw_pitch_instrument as _draw_pitch_instrument,
)
from rl_fzerox.ui.watch.view.components.cockpit.axis import (
    draw_steer_instrument as _draw_steer_instrument,
)

__all__ = ["_draw_pitch_instrument", "_draw_steer_instrument"]
