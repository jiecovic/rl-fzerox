# tests/core/runtime_spec/test_rom_resolution.py
from __future__ import annotations

from pathlib import Path

import pytest

from rl_fzerox.core.runtime_spec import roms


def test_resolve_fzerox_rom_path_accepts_arbitrary_compatible_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rom_dir = tmp_path / "local" / "roms"
    rom_dir.mkdir(parents=True)
    invalid = rom_dir / "other_game.n64"
    valid = rom_dir / "F-Zero X (USA).n64"
    invalid.write_bytes(b"invalid")
    valid.write_bytes(b"valid")

    def validate(path: Path) -> None:
        if path != valid.resolve():
            raise ValueError("unsupported rom")

    monkeypatch.setattr(roms, "_validate_supported_rom_path", validate)

    assert roms.resolve_fzerox_rom_path(tmp_path) == valid.resolve()
    assert valid.exists()
    assert invalid.exists()


def test_resolve_fzerox_rom_path_prefers_documented_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rom_dir = tmp_path / "local" / "roms"
    rom_dir.mkdir(parents=True)
    alternate = rom_dir / "a-compatible-copy.n64"
    documented = rom_dir / "fzerox_usa.n64"
    alternate.write_bytes(b"valid")
    documented.write_bytes(b"valid")
    calls: list[Path] = []

    def validate(path: Path) -> None:
        calls.append(path)

    monkeypatch.setattr(roms, "_validate_supported_rom_path", validate)

    assert roms.resolve_fzerox_rom_path(tmp_path) == documented.resolve()
    assert calls[0] == documented.resolve()


def test_resolve_fzerox_rom_path_reports_rejected_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rom_dir = tmp_path / "local" / "roms"
    rom_dir.mkdir(parents=True)
    (rom_dir / "pal.n64").write_bytes(b"pal")

    def validate(path: Path) -> None:
        raise ValueError(f"unsupported {path.name}")

    monkeypatch.setattr(roms, "_validate_supported_rom_path", validate)

    with pytest.raises(roms.FZeroXRomResolutionError, match="Rejected candidates: pal.n64"):
        roms.resolve_fzerox_rom_path(tmp_path)
