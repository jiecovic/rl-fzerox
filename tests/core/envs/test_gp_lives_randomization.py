# tests/core/envs/test_gp_lives_randomization.py
from __future__ import annotations

from rl_fzerox.core.envs.engine.reset.lives import (
    GP_LIVES_RAM,
    patch_gp_lives,
    randomize_gp_lives_on_reset,
)
from tests.support.fakes import SyntheticBackend
from tests.support.native_objects import make_telemetry


def test_patch_gp_lives_writes_all_player_slots() -> None:
    backend = SyntheticBackend()

    patch_gp_lives(backend=backend, lives=3)

    assert backend.read_system_ram(GP_LIVES_RAM.player_lives_offset, 8) == bytes(
        [3, 0, 3, 0, 3, 0, 3, 0]
    )


def test_randomize_gp_lives_uses_difficulty_default_plus_jitter() -> None:
    backend = SyntheticBackend()
    info: dict[str, object] = {}

    randomize_gp_lives_on_reset(
        backend=backend,
        telemetry=make_telemetry(difficulty_raw=2, difficulty_name="expert"),
        jitter_min=4,
        jitter_max=4,
        seed=13,
        info=info,
    )

    assert info["gp_lives_randomized"] is True
    assert info["gp_lives_base"] == 3
    assert info["gp_lives_jitter"] == 4
    assert info["gp_lives"] == 7
    assert backend.read_system_ram(GP_LIVES_RAM.player_lives_offset, 8) == bytes(
        [7, 0, 7, 0, 7, 0, 7, 0]
    )


def test_randomize_gp_lives_prefers_selected_track_difficulty() -> None:
    backend = SyntheticBackend()
    info: dict[str, object] = {}

    randomize_gp_lives_on_reset(
        backend=backend,
        telemetry=make_telemetry(difficulty_raw=0, difficulty_name="novice"),
        target_gp_difficulty="master",
        jitter_min=1,
        jitter_max=1,
        seed=13,
        info=info,
    )

    assert info["gp_lives_difficulty_raw"] == 3
    assert info["gp_lives_base"] == 2
    assert info["gp_lives"] == 3
    assert backend.read_system_ram(GP_LIVES_RAM.player_lives_offset, 8) == bytes(
        [3, 0, 3, 0, 3, 0, 3, 0]
    )


def test_randomize_gp_lives_clamps_only_below_zero() -> None:
    backend = SyntheticBackend()
    info: dict[str, object] = {}

    randomize_gp_lives_on_reset(
        backend=backend,
        telemetry=make_telemetry(difficulty_raw=3, difficulty_name="master"),
        jitter_min=-9,
        jitter_max=-9,
        seed=13,
        info=info,
    )

    assert info["gp_lives"] == 0
    assert backend.read_system_ram(GP_LIVES_RAM.player_lives_offset, 8) == bytes(8)


def test_randomize_gp_lives_skips_non_gp_race_modes() -> None:
    backend = SyntheticBackend()
    info: dict[str, object] = {}

    randomize_gp_lives_on_reset(
        backend=backend,
        telemetry=make_telemetry(game_mode_name="time_attack"),
        jitter_min=0,
        jitter_max=4,
        seed=13,
        info=info,
    )

    assert info == {
        "gp_lives_randomized": False,
        "gp_lives_randomization_skip_reason": "not_gp_race",
    }
    assert backend.read_system_ram(GP_LIVES_RAM.player_lives_offset, 8) == bytes(8)
