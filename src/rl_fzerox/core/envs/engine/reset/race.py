# src/rl_fzerox/core/envs/engine/reset/race.py
from __future__ import annotations

from fzerox_emulator import EmulatorBackend, FZeroXTelemetry
from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.config.schema import EnvConfig

from ..info import has_custom_baseline, read_live_telemetry
from .gp_race import retarget_gp_race_baseline
from .time_attack import retarget_time_attack_baseline
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
    selected_track: SelectedTrack | None = None,
) -> tuple[RgbFrame, dict[str, object], FZeroXTelemetry | None]:
    reset_state = backend.reset()
    info = dict(reset_state.info)
    frame = reset_state.frame
    telemetry = read_live_telemetry(backend)
    if selected_track is not None:
        telemetry = _retarget_race_baseline(
            backend=backend,
            selected_track=selected_track,
            telemetry=telemetry,
            info=info,
        )
    if config.reset_to_race and not sampled_track_baseline and not has_custom_baseline(info):
        raise RuntimeError(
            "env.reset_to_race requires a custom baseline state. "
            "Configure emulator.baseline_state_path or env.track_sampling.entries instead."
        )
    return frame, info, telemetry


def _retarget_race_baseline(
    *,
    backend: EmulatorBackend,
    selected_track: SelectedTrack,
    telemetry: FZeroXTelemetry | None,
    info: dict[str, object],
) -> FZeroXTelemetry | None:
    if selected_track.mode == "time_attack":
        return retarget_time_attack_baseline(
            backend=backend,
            selected_track=selected_track,
            telemetry=telemetry,
            info=info,
        )
    if selected_track.mode == "gp_race":
        return retarget_gp_race_baseline(
            backend=backend,
            selected_track=selected_track,
            telemetry=telemetry,
            info=info,
        )
    return telemetry
