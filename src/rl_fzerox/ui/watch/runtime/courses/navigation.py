# src/rl_fzerox/ui/watch/runtime/courses/navigation.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.domain.race_difficulty import (
    RaceDifficultyName,
    is_race_difficulty_name,
    race_difficulty_names,
)
from rl_fzerox.core.runtime_spec.schema import TrackSamplingEntryConfig
from rl_fzerox.core.runtime_spec.track_sampling_identity import (
    track_sampling_course_key,
    track_sampling_reset_target_key,
)
from rl_fzerox.ui.watch.records import record_difficulty


@dataclass(frozen=True, slots=True)
class WatchCourseTarget:
    """One watch-reset target, scoped to a course/slot and optional difficulty."""

    key: str
    course_key: str
    difficulty: RaceDifficultyName | None


@dataclass(frozen=True, slots=True)
class WatchCourseRotation:
    """Difficulty-major watch reset order for the active track-sampling config."""

    targets: tuple[WatchCourseTarget, ...]
    difficulties: tuple[RaceDifficultyName | None, ...]

    @classmethod
    def from_entries(
        cls,
        entries: tuple[TrackSamplingEntryConfig, ...],
    ) -> WatchCourseRotation:
        course_order = _ordered_course_keys(entries)
        difficulties = _ordered_difficulties(entries)
        targets: list[WatchCourseTarget] = []
        seen_targets: set[str] = set()
        for difficulty in difficulties:
            for course_key in course_order:
                target = _first_target(entries, course_key=course_key, difficulty=difficulty)
                if target is None or target.key in seen_targets:
                    continue
                seen_targets.add(target.key)
                targets.append(target)
        return cls(targets=tuple(targets), difficulties=difficulties)

    def first_target(self) -> WatchCourseTarget | None:
        return self.targets[0] if self.targets else None

    def target_by_key(self, key: str | None) -> WatchCourseTarget | None:
        if key is None:
            return None
        for target in self.targets:
            if target.key == key:
                return target
        return None

    def target_for_info(self, info: dict[str, object]) -> WatchCourseTarget | None:
        target = self.target_by_key(_target_key_from_info(info))
        if target is not None:
            return target

        course_key = _course_key_from_info(info)
        if course_key is None:
            return None
        difficulty = _record_race_difficulty(info)
        return self.target_for_course_difficulty(course_key, difficulty)

    def target_for_course_difficulty(
        self,
        course_key: str,
        difficulty: RaceDifficultyName | None,
    ) -> WatchCourseTarget | None:
        for target in self.targets:
            if target.course_key == course_key and target.difficulty == difficulty:
                return target
        return None

    def normalized_key(self, key: str | None) -> str | None:
        target = self.target_by_key(key)
        if target is not None:
            return target.key
        first = self.first_target()
        return None if first is None else first.key

    def adjacent_target(
        self,
        current_key: str | None,
        *,
        offset: int,
    ) -> WatchCourseTarget | None:
        if not self.targets:
            return None
        current = self.target_by_key(current_key)
        if current is None:
            return self.first_target()
        current_index = self.targets.index(current)
        return self.targets[(current_index + offset) % len(self.targets)]

    def difficulty_target(
        self,
        current_key: str | None,
        *,
        offset: int,
    ) -> WatchCourseTarget | None:
        if not self.targets or not self.difficulties:
            return None
        current = self.target_by_key(current_key) or self.first_target()
        if current is None:
            return None
        try:
            current_difficulty_index = self.difficulties.index(current.difficulty)
        except ValueError:
            current_difficulty_index = 0
        next_difficulty = self.difficulties[
            (current_difficulty_index + offset) % len(self.difficulties)
        ]
        same_course = self.target_for_course_difficulty(current.course_key, next_difficulty)
        if same_course is not None:
            return same_course
        return next(
            (target for target in self.targets if target.difficulty == next_difficulty),
            self.first_target(),
        )


def sync_watch_rotation_info(
    *,
    info: dict[str, object],
    reset_info: dict[str, object],
    rotation: WatchCourseRotation,
    selected_reset_target_key: str | None,
    locked_reset_target_key: str | None,
) -> None:
    selected = rotation.target_by_key(selected_reset_target_key)
    if selected is None:
        info.pop("watch_selected_reset_target_key", None)
        reset_info.pop("watch_selected_reset_target_key", None)
        info.pop("watch_selected_gp_difficulty", None)
        reset_info.pop("watch_selected_gp_difficulty", None)
    else:
        info["watch_selected_reset_target_key"] = selected.key
        reset_info["watch_selected_reset_target_key"] = selected.key
        _set_optional_difficulty(
            info,
            reset_info,
            key="watch_selected_gp_difficulty",
            difficulty=selected.difficulty,
        )

    if locked_reset_target_key is None:
        info.pop("track_sampling_locked_reset_target_key", None)
        reset_info.pop("track_sampling_locked_reset_target_key", None)
        info.pop("track_sampling_locked_course_id", None)
        reset_info.pop("track_sampling_locked_course_id", None)
        return
    info["track_sampling_locked_reset_target_key"] = locked_reset_target_key
    reset_info["track_sampling_locked_reset_target_key"] = locked_reset_target_key
    # The older field name is still read by existing presentation code and means
    # "locked reset target" for watch, not a course-only identity.
    info["track_sampling_locked_course_id"] = locked_reset_target_key
    reset_info["track_sampling_locked_course_id"] = locked_reset_target_key


def _ordered_course_keys(entries: tuple[TrackSamplingEntryConfig, ...]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        course_key = _entry_course_key(entry)
        if course_key in seen:
            continue
        seen.add(course_key)
        ordered.append(course_key)
    return tuple(ordered)


def _ordered_difficulties(
    entries: tuple[TrackSamplingEntryConfig, ...],
) -> tuple[RaceDifficultyName | None, ...]:
    present = {entry.gp_difficulty for entry in entries}
    ordered: list[RaceDifficultyName | None] = [
        difficulty for difficulty in race_difficulty_names() if difficulty in present
    ]
    if None in present:
        ordered.append(None)
    return tuple(ordered)


def _first_target(
    entries: tuple[TrackSamplingEntryConfig, ...],
    *,
    course_key: str,
    difficulty: RaceDifficultyName | None,
) -> WatchCourseTarget | None:
    for entry in entries:
        if _entry_course_key(entry) != course_key or entry.gp_difficulty != difficulty:
            continue
        return WatchCourseTarget(
            key=_entry_reset_target_key(entry),
            course_key=course_key,
            difficulty=difficulty,
        )
    return None


def _entry_course_key(entry: TrackSamplingEntryConfig) -> str:
    return track_sampling_course_key(
        entry_id=entry.id,
        course_id=entry.course_id,
        runtime_course_key=entry.runtime_course_key,
        course_ref=entry.course_ref,
        course_index=entry.course_index,
    )


def _entry_reset_target_key(entry: TrackSamplingEntryConfig) -> str:
    return track_sampling_reset_target_key(
        entry_id=entry.id,
        course_id=entry.course_id,
        runtime_course_key=entry.runtime_course_key,
        course_ref=entry.course_ref,
        course_index=entry.course_index,
        gp_difficulty=entry.gp_difficulty,
    )


def _target_key_from_info(info: dict[str, object]) -> str | None:
    value = info.get("track_reset_target_key")
    return value if isinstance(value, str) and value else None


def _record_race_difficulty(info: dict[str, object]) -> RaceDifficultyName | None:
    difficulty = record_difficulty(info)
    if difficulty is None:
        return None
    if is_race_difficulty_name(difficulty):
        return difficulty
    return None


def _course_key_from_info(info: dict[str, object]) -> str | None:
    for key in (
        "track_reset_course_key",
        "track_course_key",
        "track_runtime_course_key",
        "track_course_id",
    ):
        value = info.get(key)
        if isinstance(value, str) and value:
            return value
    course_index = info.get("track_course_index", info.get("course_index"))
    if isinstance(course_index, int) and not isinstance(course_index, bool):
        return f"course:{course_index}"
    track_id = info.get("track_id")
    return track_id if isinstance(track_id, str) and track_id else None


def _set_optional_difficulty(
    info: dict[str, object],
    reset_info: dict[str, object],
    *,
    key: str,
    difficulty: RaceDifficultyName | None,
) -> None:
    if difficulty is None:
        info.pop(key, None)
        reset_info.pop(key, None)
        return
    info[key] = difficulty
    reset_info[key] = difficulty
