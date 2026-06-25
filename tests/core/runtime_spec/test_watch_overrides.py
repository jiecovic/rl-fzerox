# tests/core/runtime_spec/test_watch_overrides.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.runtime_spec.schema import EmulatorConfig, WatchAppConfig, WatchConfig
from rl_fzerox.core.runtime_spec.watch_overrides import (
    apply_watch_config_delta,
    direct_dotlist_override,
    watch_config_delta_from_dotlist,
)


def test_watch_config_delta_from_dotlist_accepts_plain_key_value_overrides() -> None:
    delta = watch_config_delta_from_dotlist(
        (
            "+watch.render_fps=unlimited",
            "watch.deterministic_policy=false",
            "watch.recording.render_input_hud=true",
        )
    )

    assert delta == {
        "watch": {
            "render_fps": "unlimited",
            "deterministic_policy": False,
            "recording": {"render_input_hud": True},
        }
    }


def test_apply_watch_config_delta_merges_nested_watch_sections(tmp_path: Path) -> None:
    config = _watch_app_config(tmp_path)

    updated = apply_watch_config_delta(
        config,
        {
            "watch": {
                "render_fps": "unlimited",
                "recording": {"render_input_hud": True},
            }
        },
    )

    assert updated.watch.render_fps == "unlimited"
    assert updated.watch.recording.render_input_hud is True
    assert updated.watch.recording.session_mp4_enabled is True
    assert config.watch.render_fps == 60.0
    assert config.watch.recording.render_input_hud is False


def test_watch_config_normalizes_null_fps_fields() -> None:
    config = WatchConfig.model_validate({"control_fps": None, "render_fps": None})

    assert config.control_fps == "auto"
    assert config.render_fps == 60.0
    assert config.model_dump(mode="json", exclude_none=False)["control_fps"] == "auto"
    assert config.model_dump(mode="json", exclude_none=False)["render_fps"] == 60.0


def test_watch_config_preserves_explicit_fps_fields() -> None:
    config = WatchConfig(control_fps="unlimited", render_fps=30.0)

    assert config.control_fps == "unlimited"
    assert config.render_fps == 30.0


def test_direct_dotlist_override_strips_leading_plus() -> None:
    assert direct_dotlist_override("+watch.render_fps=60") == "watch.render_fps=60"


@pytest.mark.parametrize(
    "override",
    (
        "watch.render_fps",
        "hydra.run.dir=local/tmp",
        "~watch.render_fps=60",
        "/watch=debug",
        "watch@debug=true",
    ),
)
def test_direct_dotlist_override_rejects_unsupported_syntax(override: str) -> None:
    with pytest.raises(ValueError):
        direct_dotlist_override(override)


def _watch_app_config(tmp_path: Path) -> WatchAppConfig:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    return WatchAppConfig(
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        )
    )
