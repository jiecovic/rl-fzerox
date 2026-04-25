# src/rl_fzerox/core/domain/courses.py
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


def built_in_course_configs() -> tuple[tuple[str, dict[str, object]], ...]:
    """Return source-owned built-in course metadata in registry format."""

    return tuple((course.ref, course.as_config()) for course in BUILT_IN_COURSES)


def built_in_course_by_ref(ref: str) -> dict[str, object] | None:
    """Return a built-in course by registry ref, if it exists."""

    course = _COURSES_BY_REF.get(ref)
    return None if course is None else course.as_config()


def built_in_course_ref_by_id(course_id: str, *, cup: str | None = None) -> tuple[str, ...]:
    """Return refs for built-in courses matching id and optional cup."""

    return tuple(
        course.ref
        for course in BUILT_IN_COURSES
        if course.id == course_id and (cup is None or course.cup == cup)
    )


def built_in_course_refs_by_cup(cup: str) -> tuple[str, ...]:
    """Return refs for all built-in courses in one cup, preserving game order."""

    return tuple(course.ref for course in BUILT_IN_COURSES if course.cup == cup)


def _jack_record(
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


BUILT_IN_COURSES: tuple[CourseInfo, ...] = (
    CourseInfo(
        id="mute_city",
        display_name="Mute City",
        cup="jack",
        course_index=0,
        records=_jack_record(
            "Mute+City",
            best_time_ms=67953,
            best_player="Daniel",
            best_date="2009-07-14",
            best_mode="PAL",
            worst_time_ms=69492,
            worst_player="POP",
            worst_date="1998-11-xx",
            worst_mode="NTSC",
        ),
    ),
    CourseInfo(
        id="silence",
        display_name="Silence",
        cup="jack",
        course_index=1,
        records=_jack_record(
            "Silence",
            best_time_ms=60638,
            best_player="Daniel",
            best_date="2012-04-10",
            best_mode="PAL",
            worst_time_ms=63279,
            worst_player="\u3053\u306b\u305f\u307e",
            worst_date="1999-01-25",
            worst_mode="NTSC",
        ),
    ),
    CourseInfo(
        id="sand_ocean",
        display_name="Sand Ocean",
        cup="jack",
        course_index=2,
        records=_jack_record(
            "Sand+Ocean",
            best_time_ms=56171,
            best_player="Linner",
            best_date="2009-07-23",
            best_mode="NTSC",
            worst_time_ms=63185,
            worst_player="\u3053\u306b\u305f\u307e",
            worst_date="1998-12-04",
            worst_mode="NTSC",
        ),
    ),
    CourseInfo(
        id="devils_forest",
        display_name="Devil's Forest",
        cup="jack",
        course_index=3,
        records=_jack_record(
            "Devil%27s+Forest",
            best_time_ms=63216,
            best_player="Linner",
            best_date="2013-07-15",
            best_mode="NTSC",
            worst_time_ms=65845,
            worst_player="KUN",
            worst_date="1999-02-xx",
            worst_mode="NTSC",
        ),
    ),
    CourseInfo(
        id="big_blue",
        display_name="Big Blue",
        cup="jack",
        course_index=4,
        records=_jack_record(
            "Big+Blue",
            best_time_ms=48035,
            best_player="Daniel",
            best_date="2015-08-26",
            best_mode="PAL",
            worst_time_ms=75846,
            worst_player="FTQ",
            worst_date="1998-11-xx",
            worst_mode="NTSC",
        ),
    ),
    CourseInfo(
        id="port_town",
        display_name="Port Town",
        cup="jack",
        course_index=5,
        records=_jack_record(
            "Port+Town",
            best_time_ms=59952,
            best_player="Daniel",
            best_date="2014-02-20",
            best_mode="PAL",
            worst_time_ms=71703,
            worst_player="\u3053\u306b\u305f\u307e",
            worst_date="1998-11-xx",
            worst_mode="NTSC",
        ),
    ),
    CourseInfo("sector_alpha", "Sector Alpha", "queen", 6),
    CourseInfo("red_canyon", "Red Canyon", "queen", 7),
    CourseInfo("devils_forest_2", "Devil's Forest 2", "queen", 8),
    CourseInfo("mute_city_2", "Mute City 2", "queen", 9),
    CourseInfo("big_blue_2", "Big Blue 2", "queen", 10),
    CourseInfo("white_land", "White Land", "queen", 11),
    CourseInfo("fire_field", "Fire Field", "king", 12),
    CourseInfo("silence_2", "Silence 2", "king", 13),
    CourseInfo("sector_beta", "Sector Beta", "king", 14),
    CourseInfo("red_canyon_2", "Red Canyon 2", "king", 15),
    CourseInfo("white_land_2", "White Land 2", "king", 16),
    CourseInfo("mute_city_3", "Mute City 3", "king", 17),
    CourseInfo("rainbow_road", "Rainbow Road", "joker", 18),
    CourseInfo("devils_forest_3", "Devil's Forest 3", "joker", 19),
    CourseInfo("space_plant", "Space Plant", "joker", 20),
    CourseInfo("sand_ocean_2", "Sand Ocean 2", "joker", 21),
    CourseInfo("port_town_2", "Port Town 2", "joker", 22),
    CourseInfo("big_hand", "Big Hand", "joker", 23),
)

_COURSES_BY_REF = {course.ref: course for course in BUILT_IN_COURSES}
