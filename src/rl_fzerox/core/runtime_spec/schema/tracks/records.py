# src/rl_fzerox/core/runtime_spec/schema/tracks/records.py
"""Reference record schemas attached to runtime track metadata.

The record models intentionally return flat dictionaries for env/watch info
payloads because those payloads cross process boundaries and must remain
pickle-safe and easy to display.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt


class TrackRecordEntryConfig(BaseModel):
    """One human reference time for a track."""

    model_config = ConfigDict(extra="forbid")

    time_ms: PositiveInt
    player: str | None = None
    date: str | None = None
    mode: Literal["NTSC", "PAL"] | None = None

    def info(self, prefix: str) -> dict[str, object]:
        """Return flat, pickle-safe info fields for HUD/runtime payloads."""

        info: dict[str, object] = {f"{prefix}_time_ms": int(self.time_ms)}
        if self.player is not None:
            info[f"{prefix}_player"] = self.player
        if self.date is not None:
            info[f"{prefix}_date"] = self.date
        if self.mode is not None:
            info[f"{prefix}_mode"] = self.mode
        return info


class TrackRecordsConfig(BaseModel):
    """External reference records for a track."""

    model_config = ConfigDict(extra="forbid")

    source_label: str = "F-Zero X WR History"
    source_url: str | None = None
    non_agg_best: TrackRecordEntryConfig | None = None
    non_agg_worst: TrackRecordEntryConfig | None = None

    def info(self) -> dict[str, object]:
        """Return flat, pickle-safe info fields for HUD/runtime payloads."""

        info: dict[str, object] = {"track_record_source_label": self.source_label}
        if self.source_url is not None:
            info["track_record_source_url"] = self.source_url
        if self.non_agg_best is not None:
            info.update(self.non_agg_best.info("track_non_agg_best"))
        if self.non_agg_worst is not None:
            info.update(self.non_agg_worst.info("track_non_agg_worst"))
        return info
