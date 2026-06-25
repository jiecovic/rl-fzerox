# src/rl_fzerox/ui/watch/records/__init__.py
"""Public Watch record API.

Runtime workers and side panels import this package for per-track record books,
record entries, attempt stats, and record identity helpers. Implementation is
split by responsibility so record-key rules, finish extraction, and book
mutation can be reviewed independently.
"""

from __future__ import annotations

from rl_fzerox.ui.watch.records.attempts import TrackAttemptStats
from rl_fzerox.ui.watch.records.book import TrackRecordBook
from rl_fzerox.ui.watch.records.entry import TrackRecordEntry
from rl_fzerox.ui.watch.records.identity import (
    base_track_record_key,
    record_difficulty,
    track_record_key,
    track_record_lookup_keys,
)
from rl_fzerox.ui.watch.records.types import TrackFinishSetup, TrackRecordKey

__all__ = (
    "TrackAttemptStats",
    "TrackFinishSetup",
    "TrackRecordBook",
    "TrackRecordEntry",
    "TrackRecordKey",
    "base_track_record_key",
    "record_difficulty",
    "track_record_key",
    "track_record_lookup_keys",
)
