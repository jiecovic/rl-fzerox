# tests/apps/test_save_snapshot.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_fzerox.apps.dev import save_snapshot


class _FakeEmulator:
    instances: list[_FakeEmulator] = []

    def __init__(
        self,
        *,
        core_path: Path,
        rom_path: Path,
        runtime_dir: Path | None,
        renderer: str,
    ) -> None:
        self.core_path = core_path
        self.rom_path = rom_path
        self.runtime_dir = runtime_dir
        self.renderer = renderer
        self.frame_index = 12
        self.injected_save_ram: bytes | None = None
        self.step_calls: list[tuple[int, bool]] = []
        self.closed = False
        _FakeEmulator.instances.append(self)

    def write_save_ram(self, data: bytes) -> None:
        self.injected_save_ram = data

    def step_frames(self, count: int, *, capture_video: bool = True) -> None:
        self.step_calls.append((count, capture_video))

    def read_save_ram(self) -> bytes:
        return b"\x00\x01\x00\xff"

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _fake_emulator(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeEmulator.instances.clear()
    monkeypatch.setattr(save_snapshot, "Emulator", _FakeEmulator)


def test_save_snapshot_captures_emulator_save_ram(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "fzerox.n64"
    runtime_dir = tmp_path / "runtime"
    input_path = tmp_path / "before.srm"
    output_path = tmp_path / "after.srm"
    input_path.write_bytes(b"\x10\x20")

    result = save_snapshot.main(
        [
            "--core",
            str(core_path),
            "--rom",
            str(rom_path),
            "--runtime-dir",
            str(runtime_dir),
            "--renderer",
            "angrylion",
            "--input",
            str(input_path),
            "--frames",
            "7",
            str(output_path),
        ]
    )

    assert result == 0
    assert output_path.read_bytes() == b"\x00\x01\x00\xff"
    emulator = _FakeEmulator.instances[0]
    assert emulator.core_path == core_path.resolve()
    assert emulator.rom_path == rom_path.resolve()
    assert emulator.runtime_dir == runtime_dir.resolve()
    assert emulator.renderer == "angrylion"
    assert emulator.injected_save_ram == b"\x10\x20"
    assert emulator.step_calls == [(7, False)]
    assert emulator.closed

    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output_path.resolve())
    assert payload["renderer"] == "angrylion"
    assert payload["frame_index"] == 12
    assert payload["save_ram"]["size"] == 4
    assert payload["save_ram"]["nonzero_bytes"] == 2


def test_save_snapshot_rejects_existing_output_without_force(tmp_path: Path) -> None:
    output_path = tmp_path / "existing.srm"
    output_path.write_bytes(b"old")

    with pytest.raises(SystemExit, match="--force"):
        save_snapshot.main([str(output_path)])

    assert _FakeEmulator.instances == []
    assert output_path.read_bytes() == b"old"


def test_save_snapshot_rejects_negative_frames(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--frames must be >= 0"):
        save_snapshot.main(["--frames", "-1", str(tmp_path / "out.srm")])


def test_save_snapshot_allows_force_overwrite(tmp_path: Path) -> None:
    output_path = tmp_path / "existing.srm"
    output_path.write_bytes(b"old")

    result = save_snapshot.main(["--force", str(output_path)])

    assert result == 0
    assert output_path.read_bytes() == b"\x00\x01\x00\xff"
