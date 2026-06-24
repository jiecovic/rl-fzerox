# src/rl_fzerox/ui/watch/runtime/courses/baseline.py
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from fzerox_emulator import Emulator
from rl_fzerox.core.domain.courses import X_CUP_COURSE
from rl_fzerox.core.manager import ManagerStore
from rl_fzerox.core.training.runs import RUN_LAYOUT
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingAltBaseline,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.alt_baselines import (
    alt_baseline_reset_variant_key,
)


@dataclass(frozen=True, slots=True)
class AltBaselineSaveResult:
    """Outcome of trying to save a managed alt baseline from watch."""

    handled: bool
    saved: bool
    baseline_id: str | None = None
    state_path: Path | None = None


class _StateSavingEmulator(Protocol):
    def save_state(self, path: Path) -> None: ...


def _save_baseline_state(*, emulator: Emulator, baseline_state_path: Path | None) -> None:
    emulator.capture_current_as_baseline(baseline_state_path)


def _save_managed_alt_baseline(
    *,
    emulator: _StateSavingEmulator,
    manager_db_path: Path | None,
    run_id: str | None,
    info: Mapping[str, object],
) -> AltBaselineSaveResult:
    if manager_db_path is None or not run_id:
        return AltBaselineSaveResult(handled=False, saved=False)
    if info.get("track_generated_course_kind") == X_CUP_COURSE.generated_kind:
        return AltBaselineSaveResult(handled=True, saved=False)

    course_key = _string_info(info, "track_course_key")
    source_entry_id = _string_info(info, "track_alt_baseline_source_entry_id") or _string_info(
        info,
        "track_entry_id",
    )
    if course_key is None or source_entry_id is None:
        return AltBaselineSaveResult(handled=False, saved=False)

    store = ManagerStore(manager_db_path)
    try:
        run = store.get_run(run_id)
        if run is None:
            return AltBaselineSaveResult(handled=True, saved=False)

        now = datetime.now(UTC).isoformat(timespec="seconds")
        baseline_id = _new_alt_baseline_id()
        state_path = run.run_dir / RUN_LAYOUT.baselines_dirname / "alt" / f"{baseline_id}.state"
        _save_state_atomically(emulator, state_path)
        label = _alt_baseline_label(info)
        store.upsert_run_alt_baseline(
            baseline=TrackSamplingAltBaseline(
                id=baseline_id,
                run_id=run_id,
                course_key=course_key,
                reset_variant_key=alt_baseline_reset_variant_key(
                    mode=_string_info(info, "track_mode"),
                    gp_difficulty=_string_info(info, "track_gp_difficulty"),
                    vehicle=_string_info(info, "track_vehicle"),
                ),
                source_entry_id=source_entry_id,
                label=label,
                state_path=state_path,
                weight=1.0,
                enabled=True,
                created_at=now,
                updated_at=now,
            )
        )
    finally:
        store.close()
    return AltBaselineSaveResult(
        handled=True,
        saved=True,
        baseline_id=baseline_id,
        state_path=state_path,
    )


def _save_state_atomically(emulator: _StateSavingEmulator, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}")
    try:
        emulator.save_state(tmp_path)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _new_alt_baseline_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"alt-{stamp}-{uuid4().hex[:8]}"


def _alt_baseline_label(info: Mapping[str, object]) -> str:
    frame = info.get("frame_index")
    if isinstance(frame, int | float) and not isinstance(frame, bool):
        return f"frame {int(frame)}"
    return "watch snapshot"


def _string_info(info: Mapping[str, object], key: str) -> str | None:
    value = info.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
