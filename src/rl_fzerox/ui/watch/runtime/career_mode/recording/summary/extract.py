# src/rl_fzerox/ui/watch/runtime/career_mode/recording/summary/extract.py
from __future__ import annotations

from collections.abc import Mapping

from rl_fzerox.core.career_mode.navigation import BUILT_IN_COURSES_BY_INDEX
from rl_fzerox.ui.watch.runtime.career_mode.recording.summary.values import (
    _int_mapping,
    _str_info,
    _summary_json_value,
)


def segment_label(info: Mapping[str, object]) -> str | None:
    label = info.get("career_mode_target_label")
    if not isinstance(label, str):
        return None
    stripped = label.strip()
    return stripped or None


def attempt_id(info: Mapping[str, object]) -> str | None:
    value = info.get("career_mode_attempt_id")
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def last_finished_attempt_id(info: Mapping[str, object]) -> str | None:
    attempt = info.get("career_mode_last_finished_attempt_id")
    return attempt if isinstance(attempt, str) and attempt else None


def last_finished_attempt_status(info: Mapping[str, object]) -> str | None:
    status = info.get("career_mode_last_finished_attempt_status")
    if isinstance(status, str) and status in {"succeeded", "failed"}:
        return status
    return None


def continuing_race_result(info: Mapping[str, object]) -> bool:
    return info.get("career_mode_fsm_continuing_result") is True


def post_gp_exit_frame(info: Mapping[str, object]) -> bool:
    return (
        info.get("career_mode_fsm_observed_screen") in _POST_GP_EXIT_SCREENS
        or info.get("game_mode") in _POST_GP_EXIT_MODES
        or info.get("game_mode_name") in _POST_GP_EXIT_MODES
    )


def _selected_summary_info(info: Mapping[str, object]) -> dict[str, object]:
    selected: dict[str, object] = {}
    for key in _SUMMARY_INFO_FIELDS:
        if key not in info:
            continue
        value = _summary_json_value(info[key])
        if not _valid_summary_info_value(key, value):
            continue
        selected[key] = value
    return selected


def _valid_summary_info_value(key: str, value: object) -> bool:
    if key in {"career_mode_gp_final_rank", "gp_final_rank"}:
        return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 30
    if key in {"career_mode_gp_points", "gp_points"}:
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 999
    return True


def _merge_selected_summary_info(
    current: Mapping[str, object] | None,
    selected: Mapping[str, object],
) -> dict[str, object]:
    if current is None:
        return dict(selected)
    merged = dict(selected)
    for key in _STICKY_SUMMARY_INFO_FIELDS:
        if merged.get(key) is not None:
            continue
        value = current.get(key)
        if value is not None:
            merged[key] = value
    return merged


def _course_result(info: Mapping[str, object]) -> dict[str, object] | None:
    reason = _str_info(info, "termination_reason")
    if reason not in {"finished", "retired", "crashed"}:
        return None
    result: dict[str, object] = {
        "termination_reason": reason,
    }
    for key, output_key in (
        ("track_id", "track_id"),
        ("track_course_id", "course_id"),
        ("track_course_name", "course_name"),
        ("track_course_index", "course_index"),
        ("track_gp_difficulty", "difficulty"),
        ("race_time_ms", "race_time_ms"),
        ("position", "position"),
        ("ko_star_count", "ko_star_count"),
        ("race_laps_completed", "laps_completed"),
        ("total_lap_count", "total_laps"),
        ("track_vehicle_name", "vehicle_name"),
        ("track_engine_setting_raw_value", "engine_setting_raw_value"),
    ):
        if key not in info:
            continue
        value = _summary_json_value(info[key])
        if value is not None:
            result[output_key] = value
    _add_course_result_fallbacks(result, info)
    if reason == "finished" and _course_result_identity(result) is None:
        return None
    return result


def _course_result_signature(result: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        result.get(key)
        for key in (
            "track_id",
            "course_id",
            "course_index",
            "termination_reason",
            "race_time_ms",
            "position",
        )
    )


def _course_result_identity(result: Mapping[str, object]) -> tuple[str, object] | None:
    for key in ("course_id", "track_id", "course_index"):
        value = result.get(key)
        if _valid_course_identity_value(key, value):
            return (key, value)
    return None


def _valid_course_identity_value(key: str, value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        stripped = value.strip()
        return bool(stripped) and stripped != "-"
    if key == "course_index" and isinstance(value, int) and not isinstance(value, bool):
        return value in BUILT_IN_COURSES_BY_INDEX
    return True


def _course_engine_observation(
    info: Mapping[str, object],
) -> tuple[tuple[str, object], int] | None:
    if info.get("game_mode") != "gp_race" and info.get("game_mode_name") != "gp_race":
        return None
    if _str_info(info, "termination_reason") in {"finished", "retired", "crashed"}:
        return None
    identity = _course_identity_from_info(info)
    if identity is None:
        return None
    engine_raw = _int_mapping(info, "engine_setting_raw_value_ram")
    if engine_raw is None:
        engine_raw = _int_mapping(info, "track_engine_setting_raw_value")
    if engine_raw is None:
        engine_raw = _int_mapping(info, "engine_setting_raw_value")
    if engine_raw is None:
        return None
    return identity, engine_raw


def _course_identity_from_info(info: Mapping[str, object]) -> tuple[str, object] | None:
    for key in ("track_course_id", "course_id", "track_id", "career_mode_policy_course_id"):
        value = info.get(key)
        if value is None:
            continue
        if key in {"track_course_id", "career_mode_policy_course_id"}:
            return ("course_id", value)
        return (key, value)
    for key in ("track_course_index", "course_index"):
        value = info.get(key)
        if value is not None:
            if isinstance(value, int) and not isinstance(value, bool):
                course = BUILT_IN_COURSES_BY_INDEX.get(value)
                if course is not None:
                    return ("course_id", course.id)
            return ("course_index", value)
    return None


def _existing_course_result_index(
    results: list[dict[str, object]],
    candidate: Mapping[str, object],
) -> int | None:
    for index, result in enumerate(results):
        if _course_results_match(result, candidate):
            return index
    return None


def _course_results_match(
    current: Mapping[str, object],
    candidate: Mapping[str, object],
) -> bool:
    current_identity = _course_result_identity(current)
    candidate_identity = _course_result_identity(candidate)
    if (
        current_identity is not None
        and candidate_identity is not None
        and current_identity != candidate_identity
    ):
        return False

    # Live Career Mode can observe the same terminal result twice: once before
    # course metadata has been enriched and once after. Match that ordering on
    # the immutable terminal fields so the later event fills the missing course
    # name/index/engine instead of appending a duplicate row. Do not merge only
    # by course identity: a GP attempt can spend lives retrying the same course,
    # and those failed intermediate tries must remain visible in the summary.
    current_fingerprint = _course_terminal_fingerprint(current)
    candidate_fingerprint = _course_terminal_fingerprint(candidate)
    return (
        current_fingerprint is not None
        and candidate_fingerprint is not None
        and current_fingerprint == candidate_fingerprint
    )


def _merge_missing_course_result_fields(
    current: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    merged = dict(current)
    for key, value in candidate.items():
        if key not in merged or merged[key] is None:
            merged[key] = value
    return merged


def _add_course_result_fallbacks(
    result: dict[str, object],
    info: Mapping[str, object],
) -> None:
    course_index = _int_mapping(result, "course_index")
    if course_index is None:
        course_index = _int_mapping(info, "course_index")
    if course_index is not None:
        result.setdefault("course_index", course_index)
        course = BUILT_IN_COURSES_BY_INDEX.get(course_index)
        if course is not None:
            result.setdefault("track_id", course.id)
            result.setdefault("course_id", course.id)
            result.setdefault("course_name", course.display_name)
            difficulty = _summary_json_value(info.get("difficulty"))
            if difficulty is not None:
                result.setdefault("difficulty", difficulty)


def _course_terminal_fingerprint(result: Mapping[str, object]) -> tuple[object, ...] | None:
    race_time_ms = result.get("race_time_ms")
    if race_time_ms is None:
        return None
    return (
        result.get("termination_reason"),
        race_time_ms,
        result.get("position"),
        result.get("ko_star_count"),
        result.get("laps_completed"),
        result.get("total_laps"),
    )


def _policy_checkpoint_summary(info: Mapping[str, object]) -> dict[str, object] | None:
    path = _str_info(info, "career_mode_policy_checkpoint_path")
    if path is None:
        return None
    summary: dict[str, object] = {"path": path}
    for key, output_key in (
        ("career_mode_policy_run_id", "run_id"),
        ("career_mode_policy_run_name", "run_name"),
        ("career_mode_policy_artifact", "artifact"),
        ("career_mode_policy_course_id", "course_id"),
        ("career_mode_policy_checkpoint_num_timesteps", "num_timesteps"),
        ("career_mode_policy_checkpoint_local_num_timesteps", "local_num_timesteps"),
        ("career_mode_policy_checkpoint_mtime_utc", "mtime_utc"),
        ("career_mode_policy_checkpoint_mtime_ns", "mtime_ns"),
        ("career_mode_policy_checkpoint_stage", "stage"),
        ("career_mode_policy_checkpoint_stage_index", "stage_index"),
    ):
        if key not in info:
            continue
        summary[output_key] = _summary_json_value(info[key])
    return summary


def _policy_checkpoint_signature(checkpoint: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        checkpoint.get(key)
        for key in (
            "run_id",
            "artifact",
            "course_id",
            "path",
            "mtime_ns",
            "num_timesteps",
            "local_num_timesteps",
        )
    )


_SUMMARY_INFO_FIELDS = (
    "career_mode_target_label",
    "career_mode_attempt_id",
    "career_mode_last_finished_attempt_id",
    "career_mode_last_finished_attempt_status",
    "career_mode_last_finished_attempt_failure_reason",
    "career_mode_fsm_observed_screen",
    "career_mode_fsm_terminal_reason",
    "career_mode_gp_final_rank",
    "career_mode_gp_points",
    "career_mode_policy_artifact",
    "career_mode_policy_checkpoint_local_num_timesteps",
    "career_mode_policy_checkpoint_mtime_ns",
    "career_mode_policy_checkpoint_mtime_utc",
    "career_mode_policy_checkpoint_num_timesteps",
    "career_mode_policy_checkpoint_path",
    "career_mode_policy_checkpoint_stage",
    "career_mode_policy_checkpoint_stage_index",
    "career_mode_policy_course_id",
    "career_mode_policy_run_id",
    "career_mode_policy_run_name",
    "game_mode",
    "game_mode_name",
    "gp_final_rank",
    "gp_points",
    "track_id",
    "track_course_id",
    "track_course_name",
    "track_course_index",
    "track_gp_difficulty",
    "track_vehicle_name",
    "track_engine_setting_raw_value",
    "termination_reason",
    "race_time_ms",
    "position",
    "race_laps_completed",
    "total_lap_count",
)

_STICKY_SUMMARY_INFO_FIELDS = frozenset(
    {
        "career_mode_gp_final_rank",
        "career_mode_gp_points",
        "gp_final_rank",
        "gp_points",
    }
)

_POST_GP_EXIT_SCREENS = frozenset(
    {
        "title",
        "main_menu_gp",
        "main_menu_other",
        "course_select",
    }
)

_POST_GP_EXIT_MODES = frozenset(
    {
        "title",
        "main_menu",
        "course_select",
        "game_over",
    }
)
