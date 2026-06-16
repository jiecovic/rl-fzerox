# src/rl_fzerox/ui/watch/runtime/snapshots/frames.py
from __future__ import annotations

from fzerox_emulator import RaceControlState
from fzerox_emulator.arrays import (
    AudioFrameCounts,
    ControllerMaskBatch,
    DisplayFrames,
    Pcm16Samples,
    RgbFrame,
)


def _display_frames_or_fallback(
    display_frames: DisplayFrames,
    *,
    fallback: RgbFrame,
) -> DisplayFrames:
    if isinstance(display_frames, tuple):
        return display_frames if display_frames else (fallback,)
    return display_frames if len(display_frames) > 0 else (fallback,)


def _display_controller_states(
    display_controller_masks: ControllerMaskBatch,
    *,
    frames: DisplayFrames,
    fallback_previous: RaceControlState,
    fallback_final: RaceControlState,
) -> tuple[tuple[RaceControlState, ...], bool]:
    frame_count = len(frames)
    masks = tuple(int(mask) for mask in display_controller_masks)
    if len(masks) != frame_count:
        return (
            tuple(
                fallback_final if index == frame_count - 1 else fallback_previous
                for index in range(frame_count)
            ),
            False,
        )
    return (
        tuple(
            RaceControlState.from_mask(
                mask,
                stick_x=fallback_final.stick_x,
                pitch=fallback_final.pitch,
            )
            for mask in masks
        ),
        True,
    )


def _recording_frame_info(
    info: dict[str, object],
    *,
    control_state: RaceControlState,
    render_input_hud: bool,
    policy_active: bool,
) -> dict[str, object]:
    if not render_input_hud or not policy_active:
        return info
    return {
        **info,
        "watch_recording_input_hud": True,
        "watch_recording_input_gas": control_state.gas,
        "watch_recording_input_boost": control_state.boost,
        "watch_recording_input_air_brake": control_state.air_brake,
        "watch_recording_input_lean_left": control_state.lean_left,
        "watch_recording_input_lean_right": control_state.lean_right,
        "watch_recording_input_stick_x": control_state.stick_x,
        "watch_recording_input_pitch": control_state.pitch,
    }


def _audio_chunks_for_frames(
    audio_samples: Pcm16Samples,
    audio_frame_counts: AudioFrameCounts,
    *,
    frame_count: int,
) -> tuple[Pcm16Samples, ...]:
    empty = tuple(() for _ in range(frame_count))
    if len(audio_samples) == 0:
        return empty
    counts = tuple(int(count) for count in audio_frame_counts)
    if len(counts) != frame_count or any(count < 0 for count in counts):
        return empty
    chunks: list[Pcm16Samples] = []
    offset = 0
    for count in counts:
        sample_count = count * 2
        chunks.append(audio_samples[offset : offset + sample_count])
        offset += sample_count
    if offset != len(audio_samples):
        return empty
    return tuple(chunks)
