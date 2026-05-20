# tests/core/game/test_track_bounds.py
from __future__ import annotations

from rl_fzerox.core.envs.track_bounds import track_edge_state
from tests.support.native_objects import make_player_telemetry


def test_track_edge_state_treats_small_center_overhang_as_edge_not_outside() -> None:
    player = make_player_telemetry(
        signed_lateral_offset=104.0,
        current_radius_left=100.0,
    )

    edge_state = track_edge_state(player)

    assert edge_state.near_side == "left"
    assert edge_state.ratio == 1.04
    assert not edge_state.outside_bounds


def test_track_edge_state_marks_clear_overhang_as_outside() -> None:
    player = make_player_telemetry(
        signed_lateral_offset=-128.0,
        current_radius_right=120.0,
    )

    edge_state = track_edge_state(player)

    assert edge_state.near_side == "right"
    assert edge_state.ratio == -128.0 / 120.0
    assert edge_state.outside_bounds
