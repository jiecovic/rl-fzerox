# src/rl_fzerox/core/domain/courses/model.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CourseRecord:
    """One reference time entry for a course."""

    time_ms: int
    player: str
    date: str
    mode: str

    def as_config(self) -> dict[str, object]:
        return {
            "time_ms": self.time_ms,
            "player": self.player,
            "date": self.date,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class CourseRecords:
    """Reference record metadata shown in watch HUD."""

    source_label: str
    source_url: str
    non_agg_best: CourseRecord
    non_agg_worst: CourseRecord

    def as_config(self) -> dict[str, object]:
        return {
            "source_label": self.source_label,
            "source_url": self.source_url,
            "non_agg_best": self.non_agg_best.as_config(),
            "non_agg_worst": self.non_agg_worst.as_config(),
        }


@dataclass(frozen=True, slots=True)
class CourseInfo:
    """Built-in F-Zero X course metadata keyed by stable config id."""

    id: str
    display_name: str
    cup: str
    course_index: int
    records: CourseRecords | None = None

    @property
    def ref(self) -> str:
        return f"{self.cup}/{self.id}"

    def as_config(self) -> dict[str, object]:
        config: dict[str, object] = {
            "id": self.id,
            "display_name": self.display_name,
            "cup": self.cup,
            "course_index": self.course_index,
        }
        if self.records is not None:
            config["records"] = self.records.as_config()
        return config


def course_record(
    track: str,
    *,
    best_time_ms: int,
    best_player: str,
    best_date: str,
    best_mode: str,
    worst_time_ms: int,
    worst_player: str,
    worst_date: str,
    worst_mode: str,
) -> CourseRecords:
    return CourseRecords(
        source_label="F-Zero X WR History",
        source_url=f"https://www.fzerowrs.com/x/display.php?track={track}",
        non_agg_best=CourseRecord(
            time_ms=best_time_ms,
            player=best_player,
            date=best_date,
            mode=best_mode,
        ),
        non_agg_worst=CourseRecord(
            time_ms=worst_time_ms,
            player=worst_player,
            date=worst_date,
            mode=worst_mode,
        ),
    )
