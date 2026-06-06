# tests/apps/test_memory_probe.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_fzerox.apps.dev import memory_probe


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
        self.frame_index = 42
        self.injected_save_ram: bytes | None = None
        self.step_calls: list[tuple[int, bool]] = []
        self.closed = False
        _FakeEmulator.instances.append(self)

    def write_save_ram(self, data: bytes) -> None:
        self.injected_save_ram = data

    def step_frames(self, count: int, *, capture_video: bool = True) -> None:
        self.step_calls.append((count, capture_video))

    def read_save_ram(self) -> bytes:
        return b"\x01\x02\x03\x04"

    def read_system_ram(self, offset: int, length: int) -> bytes:
        return bytes(range(offset, offset + length))

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _fake_emulator(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeEmulator.instances.clear()
    monkeypatch.setattr(memory_probe, "Emulator", _FakeEmulator)


def test_memory_probe_reports_save_and_system_memory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "fzerox.n64"
    runtime_dir = tmp_path / "runtime"
    input_path = tmp_path / "before.srm"
    input_path.write_bytes(b"\x10\x20")

    result = memory_probe.main(
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
            "9",
            "--probe",
            "save_word=save_ram:0x1:2:u16_be:Save word",
            "--probe",
            "mode=system_ram:0x4:4:u32_le:Mode probe",
        ]
    )

    assert result == 0
    emulator = _FakeEmulator.instances[0]
    assert emulator.core_path == core_path.resolve()
    assert emulator.rom_path == rom_path.resolve()
    assert emulator.runtime_dir == runtime_dir.resolve()
    assert emulator.renderer == "angrylion"
    assert emulator.injected_save_ram == b"\x10\x20"
    assert emulator.step_calls == [(9, False)]
    assert emulator.closed

    payload = json.loads(capsys.readouterr().out)
    assert payload["renderer"] == "angrylion"
    assert payload["frame_index"] == 42
    assert payload["probe_count"] == 2
    assert payload["memory"]["save_ram"]["size"] == 4
    assert payload["memory"]["readings"] == [
        {
            "key": "save_word",
            "label": "Save word",
            "length": 2,
            "offset": 1,
            "raw_hex": "0203",
            "region": "save_ram",
            "value": 515,
            "value_format": "u16_be",
        },
        {
            "key": "mode",
            "label": "Mode probe",
            "length": 4,
            "offset": 4,
            "raw_hex": "04050607",
            "region": "system_ram",
            "value": 117835012,
            "value_format": "u32_le",
        },
    ]


def test_memory_probe_rejects_negative_frames(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--frames must be >= 0"):
        memory_probe.main(["--frames", "-1", "--rom", str(tmp_path / "fzerox.n64")])
