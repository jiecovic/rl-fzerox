# tests/core/runtime_spec/test_track_registry.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.runtime_spec.track_registry import registry as registry_module
from rl_fzerox.core.runtime_spec.track_registry.expand import expand_track_sampling_section


def test_track_sampling_entry_enrichment_reuses_track_registry_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracks_root = tmp_path / "tracks"
    tracks_root.mkdir()
    (tracks_root / "mute-city-blue-falcon.yaml").write_text(
        """
track:
  id: mute_city_blue_falcon
  display_name: Mute City Blue Falcon
  course_ref: jack/mute_city
  mode: gp_race
  gp_difficulty: master
  vehicle: blue_falcon
  engine_setting_raw_value: 10
""".strip(),
        encoding="utf-8",
    )
    section: dict[str, object] = {
        "entries": [
            {"id": "mute_city_blue_falcon", "weight": 1.0},
            {"id": "mute_city_blue_falcon", "weight": 2.0},
        ],
    }
    original_iter_track_configs = registry_module.iter_track_configs
    calls = 0

    def counted_iter_track_configs(
        *,
        config_root: Path,
    ) -> tuple[tuple[str, dict[str, object]], ...]:
        nonlocal calls
        calls += 1
        return original_iter_track_configs(config_root=config_root)

    monkeypatch.setattr(registry_module, "iter_track_configs", counted_iter_track_configs)

    expand_track_sampling_section(section, tmp_path)

    assert calls == 1
    entries = section["entries"]
    assert isinstance(entries, list)
    assert [entry["weight"] for entry in entries if isinstance(entry, dict)] == [1.0, 2.0]
    assert all(
        isinstance(entry, dict)
        and entry["display_name"] == "Mute City Blue Falcon"
        and entry["vehicle_name"] == "Blue Falcon"
        and entry["course_id"] == "mute_city"
        for entry in entries
    )
