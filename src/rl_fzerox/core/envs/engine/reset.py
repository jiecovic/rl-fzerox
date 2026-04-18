# src/rl_fzerox/core/envs/engine/reset.py
from __future__ import annotations

from fzerox_emulator import EmulatorBackend, FZeroXTelemetry
from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.config.schema import EnvConfig

from .info import backend_step_info, has_custom_baseline, read_live_telemetry
from .tracks import SelectedTrack, TrackBaselineCache


def load_track_baseline(
    *,
    backend: EmulatorBackend,
    cache: TrackBaselineCache,
    selected_track: SelectedTrack,
    cache_enabled: bool,
) -> None:
    if cache_enabled:
        cache.load_into_backend(backend, selected_track.baseline_state_path)
        return
    backend.load_baseline(selected_track.baseline_state_path)


def reset_race_state(
    *,
    backend: EmulatorBackend,
    config: EnvConfig,
    sampled_track_baseline: bool,
) -> tuple[RgbFrame, dict[str, object], FZeroXTelemetry | None]:
    reset_state = backend.reset()
    info = dict(reset_state.info)
    frame = reset_state.frame
    if config.reset_to_race and not sampled_track_baseline and not has_custom_baseline(info):
        raise RuntimeError(
            "env.reset_to_race requires a custom baseline state. "
            "Configure emulator.baseline_state_path or env.track_sampling.entries instead."
        )
    return frame, info, read_live_telemetry(backend)


def benchmark_noop_reset_state(
    backend: EmulatorBackend,
) -> tuple[dict[str, object], FZeroXTelemetry | None]:
    """Reset Python episode bookkeeping without restoring emulator state."""

    info = backend_step_info(backend)
    info["reset_mode"] = "benchmark_noop_reset"
    info["benchmark_noop_reset"] = True
    return info, read_live_telemetry(backend)
