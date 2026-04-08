# tests/core/test_boot.py
from __future__ import annotations

import pytest

from fzerox_emulator import ControllerState
from rl_fzerox.core.boot import boot_into_first_race
from tests.support.fakes import SyntheticBackend


def test_boot_into_first_race_raises_when_title_mode_never_appears(monkeypatch) -> None:
    backend = SyntheticBackend()

    monkeypatch.setattr(
        "rl_fzerox.core.boot._wait_for_mode",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(RuntimeError, match="title"):
        boot_into_first_race(backend)

    assert backend.last_controller_state == ControllerState()


def test_boot_into_first_race_raises_when_gp_race_mode_never_appears(monkeypatch) -> None:
    backend = SyntheticBackend()

    monkeypatch.setattr(
        "rl_fzerox.core.boot._wait_for_mode",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "rl_fzerox.core.boot._wait_for_gp_race",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(RuntimeError, match="gp_race"):
        boot_into_first_race(backend)

    assert backend.last_controller_state == ControllerState()
