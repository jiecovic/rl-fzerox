# tests/core/training/training_artifacts_support.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pytest import MonkeyPatch

from rl_fzerox.core.runtime_spec.schema import (
    TrackSamplingEntryConfig,
)
from rl_fzerox.core.training.runs import (
    baseline_materializer,
)
from rl_fzerox.core.training.runs.race_start import RaceStartVariant


class _FakeSaveable:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def save(self, path: str) -> None:
        Path(path).write_bytes(self._payload)


class _FakeModel:
    def __init__(self) -> None:
        self.policy = _FakeSaveable(b"policy")

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"model")


@dataclass
class _FakeMaterializerCapture:
    variants: list[RaceStartVariant]
    generic_modes: list[str]
    baseline_state_paths: list[Path | None]


def _patch_fake_boot_materializer(
    monkeypatch: MonkeyPatch,
    *,
    payload: bytes = b"generated",
) -> _FakeMaterializerCapture:
    capture = _FakeMaterializerCapture(variants=[], generic_modes=[], baseline_state_paths=[])

    class FakeEmulator:
        def __init__(self, **kwargs: object) -> None:
            raw_baseline_state_path = kwargs.get("baseline_state_path")
            baseline_state_path = (
                raw_baseline_state_path if isinstance(raw_baseline_state_path, Path) else None
            )
            capture.baseline_state_paths.append(baseline_state_path)

        def reset(self) -> None:
            pass

        def set_controller_state(self, _: object) -> None:
            pass

        def step_frames(self, _: int, *, capture_video: bool = False) -> None:
            del capture_video

        def try_read_telemetry(self):
            return None

        def save_state(self, path: Path) -> None:
            path.write_bytes(payload)

        def close(self) -> None:
            pass

    def fake_materialize_generic_mode_seed(
        *,
        emulator: object,
        mode: str,
    ) -> None:
        del emulator
        capture.generic_modes.append(mode)

    def fake_materialize_race_start_from_menu_seed(
        *,
        emulator: object,
        variant: RaceStartVariant,
    ) -> None:
        del emulator
        capture.variants.append(variant)

    monkeypatch.setattr(baseline_materializer, "Emulator", FakeEmulator)
    monkeypatch.setattr(
        baseline_materializer,
        "materialize_generic_mode_seed",
        fake_materialize_generic_mode_seed,
    )
    monkeypatch.setattr(
        baseline_materializer,
        "materialize_race_start_from_menu_seed",
        fake_materialize_race_start_from_menu_seed,
    )
    return capture


def _required_baseline_path(entry: TrackSamplingEntryConfig) -> Path:
    assert entry.baseline_state_path is not None
    return entry.baseline_state_path
