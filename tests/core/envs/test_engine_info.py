# tests/core/envs/test_engine_info.py
from __future__ import annotations

import pytest

from rl_fzerox.core.envs.engine.info import backend_step_info


class _Backend:
    name = "fake"
    frame_index = 12
    display_aspect_ratio = 4 / 3
    native_fps = 60.0

    def vehicle_setup_info(self) -> dict[str, object]:
        return {
            "vehicle_character_index_ram": 0,
            "engine_setting_ram": 0.5,
            "engine_setting_percent_ram": 50.0,
            "character_engine_setting_ram": 0.5,
            "racer_engine_curve_ram": 0.371747,
        }


def test_backend_step_info_includes_native_engine_setting() -> None:
    backend = _Backend()

    info = backend_step_info(backend)

    assert info["engine_setting_ram"] == 0.5
    assert info["engine_setting_percent_ram"] == 50.0
    assert info["character_engine_setting_ram"] == 0.5
    assert info["racer_engine_curve_ram"] == pytest.approx(0.371747)
