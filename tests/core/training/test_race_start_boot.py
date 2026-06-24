# tests/core/training/test_race_start_boot.py
from __future__ import annotations

from typing import Any

from pytest import MonkeyPatch

from rl_fzerox.core.training.runs.race_start import boot
from rl_fzerox.core.training.runs.race_start.models import RaceStartVariant


class _FakeEmulator:
    def reset(self) -> None:
        pass

    def set_controller_state(self, _: object) -> None:
        pass

    def step_frames(self, _: int, *, capture_video: bool = False) -> None:
        del capture_video


def test_gp_menu_seed_selects_machine_before_exact_setup(monkeypatch: MonkeyPatch) -> None:
    events: list[str] = []

    def fake_wait_until_mode(
        emulator: object,
        *,
        target_mode: str,
        require_race_mode: bool = False,
    ) -> None:
        del emulator, require_race_mode
        events.append(f"wait:{target_mode}")

    def fake_select_machine(emulator: object, variant: RaceStartVariant) -> None:
        del emulator
        events.append(f"select:{variant.character_index}")

    def fake_apply_exact_setup(emulator: object, variant: RaceStartVariant) -> None:
        del emulator
        events.append(f"exact:{variant.character_index}")

    def fake_step_until_ready(emulator: object, variant: RaceStartVariant) -> None:
        del emulator
        events.append(f"ready:{variant.character_index}")

    monkeypatch.setattr(boot, "wait_until_mode", fake_wait_until_mode)
    monkeypatch.setattr(boot, "select_machine", fake_select_machine)
    monkeypatch.setattr(boot, "_apply_exact_race_start_setup", fake_apply_exact_setup)
    monkeypatch.setattr(boot, "step_until_ready_from_boot", fake_step_until_ready)

    emulator: Any = _FakeEmulator()
    boot.materialize_gp_race_start_from_menu_seed(
        emulator=emulator,
        variant=RaceStartVariant(
            course_index=0,
            mode="gp_race",
            gp_difficulty="novice",
            character_index=1,
            machine_select_slot=1,
            engine_setting_raw_value=50,
            race_intro_target_timer=None,
        ),
    )

    assert events == [
        "wait:machine_select",
        "select:1",
        "wait:machine_settings",
        "exact:1",
        "ready:1",
    ]
