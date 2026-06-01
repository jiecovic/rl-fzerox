# tests/core/manager/test_web_fixtures.py
from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from rl_fzerox.core.manager.fixtures import web


def test_web_fixture_check_accepts_different_json_formatting(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "manager-fixtures.json"
    fixture_path.write_text(
        json.dumps(web.web_test_fixture_payload(), separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.setattr(web, "web_test_fixture_output_path", lambda: fixture_path)

    assert web.check_web_test_fixtures()


def test_web_fixture_check_rejects_stale_payload(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "manager-fixtures.json"
    fixture_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(web, "web_test_fixture_output_path", lambda: fixture_path)

    assert not web.check_web_test_fixtures()
