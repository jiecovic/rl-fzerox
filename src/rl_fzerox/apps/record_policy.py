# src/rl_fzerox/apps/record_policy.py
from __future__ import annotations

from rl_fzerox.apps.recording.cli import (
    main,
    parse_args,
)
from rl_fzerox.apps.recording.cli import (
    with_deterministic_policy as _with_deterministic_policy,
)
from rl_fzerox.apps.recording.models import (
    AttemptRunResult,
    RecordAttemptResult,
    RecordingSession,
    RecordMode,
)
from rl_fzerox.apps.recording.runner import (
    _attempt_seed,
    _finished_rank,
    _move_result_to_output,
    _print_attempt_result,
    _record_attempts,
    _record_matched_attempt,
    _run_attempt,
    record_policy_episode,
)
from rl_fzerox.apps.recording.session import open_recording_session as _open_recording_session
from rl_fzerox.apps.recording.video import FfmpegRgbWriter

__all__ = [
    "AttemptRunResult",
    "FfmpegRgbWriter",
    "RecordAttemptResult",
    "RecordMode",
    "RecordingSession",
    "_attempt_seed",
    "_finished_rank",
    "_move_result_to_output",
    "_open_recording_session",
    "_print_attempt_result",
    "_record_attempts",
    "_record_matched_attempt",
    "_run_attempt",
    "_with_deterministic_policy",
    "main",
    "parse_args",
    "record_policy_episode",
]


if __name__ == "__main__":
    main()
