# src/rl_fzerox/ui/watch/runtime/career_mode/recording/summary/__init__.py
"""Career Mode recording summary facade.

Only extraction helpers and file-writing entrypoints are re-exported here.
Builder, snapshot, and session-writer internals live in their concrete modules.
"""

from __future__ import annotations

from rl_fzerox.ui.watch.runtime.career_mode.recording.summary.extract import (
    attempt_id,
    continuing_race_result,
    last_finished_attempt_id,
    last_finished_attempt_status,
    post_gp_exit_frame,
    segment_label,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording.summary.values import utc_timestamp
from rl_fzerox.ui.watch.runtime.career_mode.recording.summary.writer import (
    write_segment_summary_files,
)

__all__ = (
    "attempt_id",
    "continuing_race_result",
    "last_finished_attempt_id",
    "last_finished_attempt_status",
    "post_gp_exit_frame",
    "segment_label",
    "utc_timestamp",
    "write_segment_summary_files",
)
