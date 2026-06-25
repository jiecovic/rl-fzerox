# tests/ui/test_watch_runtime_course_rotation.py
"""Watch runtime tests for course rotation and reset target selection.

These cases cover retry-vs-advance behavior after episodes and the stable reset
keys used for GP difficulty and generated X-Cup slots.
"""

from rl_fzerox.core.runtime_spec.schema import TrackSamplingEntryConfig
from rl_fzerox.ui.watch.runtime.courses.navigation import WatchCourseRotation
from rl_fzerox.ui.watch.runtime.live.worker import _sync_next_watch_reset_after_episode
from tests.ui.watch_runtime_ipc_support import _sample_rotation, _SequentialResetEnv


def test_watch_course_rotation_retries_same_course_after_crash() -> None:
    env = _SequentialResetEnv()
    rotation = _sample_rotation()

    selected = _sync_next_watch_reset_after_episode(
        env=env,
        rotation=rotation,
        info={
            "termination_reason": "crashed",
            "track_reset_target_key": "mute_city#difficulty=novice",
            "race_laps_completed": 1,
            "total_lap_count": 3,
        },
        episode_done=True,
        selected_reset_target_key="mute_city#difficulty=novice",
        locked_reset_target_key=None,
    )

    assert selected == "mute_city#difficulty=novice"
    assert env.next_courses == ["mute_city#difficulty=novice"]


def test_watch_course_rotation_retries_same_course_after_timeout() -> None:
    env = _SequentialResetEnv()
    rotation = WatchCourseRotation.from_entries(
        (
            TrackSamplingEntryConfig(
                id="x_cup_a",
                course_id="x_cup_hash_a",
                runtime_course_key="x_cup_slot_1",
                gp_difficulty="novice",
            ),
        )
    )

    selected = _sync_next_watch_reset_after_episode(
        env=env,
        rotation=rotation,
        info={
            "termination_reason": "timeout",
            "track_runtime_course_key": "x_cup_slot_1",
            "track_gp_difficulty": "novice",
            "race_laps_completed": 2,
            "total_lap_count": 3,
        },
        episode_done=True,
        selected_reset_target_key="x_cup_slot_1#difficulty=novice",
        locked_reset_target_key=None,
    )

    assert selected == "x_cup_slot_1#difficulty=novice"
    assert env.next_courses == ["x_cup_slot_1#difficulty=novice"]


def test_watch_course_rotation_advances_after_completed_finish() -> None:
    env = _SequentialResetEnv()
    rotation = _sample_rotation()

    selected = _sync_next_watch_reset_after_episode(
        env=env,
        rotation=rotation,
        info={
            "termination_reason": "finished",
            "track_reset_target_key": "mute_city#difficulty=novice",
            "race_laps_completed": 3,
            "total_lap_count": 3,
        },
        episode_done=True,
        selected_reset_target_key="mute_city#difficulty=novice",
        locked_reset_target_key=None,
    )

    assert selected == "silence#difficulty=novice"
    assert env.next_courses == ["silence#difficulty=novice"]


def test_watch_course_rotation_keeps_manual_and_locked_resets_unchanged() -> None:
    env = _SequentialResetEnv()
    rotation = _sample_rotation()
    info = {
        "termination_reason": "crashed",
        "track_reset_target_key": "mute_city#difficulty=novice",
        "race_laps_completed": 1,
        "total_lap_count": 3,
    }

    _sync_next_watch_reset_after_episode(
        env=env,
        rotation=rotation,
        info=info,
        episode_done=False,
        selected_reset_target_key="mute_city#difficulty=novice",
        locked_reset_target_key=None,
    )
    _sync_next_watch_reset_after_episode(
        env=env,
        rotation=rotation,
        info=info,
        episode_done=True,
        selected_reset_target_key="mute_city#difficulty=novice",
        locked_reset_target_key="mute_city#difficulty=novice",
    )

    assert env.next_courses == []


def test_watch_course_rotation_is_difficulty_major() -> None:
    entries = (
        TrackSamplingEntryConfig(id="mute_novice", course_id="mute_city", gp_difficulty="novice"),
        TrackSamplingEntryConfig(id="mute_expert", course_id="mute_city", gp_difficulty="expert"),
        TrackSamplingEntryConfig(id="silence_novice", course_id="silence", gp_difficulty="novice"),
        TrackSamplingEntryConfig(id="silence_expert", course_id="silence", gp_difficulty="expert"),
    )

    rotation = WatchCourseRotation.from_entries(entries)

    assert tuple(target.key for target in rotation.targets) == (
        "mute_city#difficulty=novice",
        "silence#difficulty=novice",
        "mute_city#difficulty=expert",
        "silence#difficulty=expert",
    )


def test_watch_course_rotation_uses_stable_runtime_course_keys() -> None:
    entries = (
        TrackSamplingEntryConfig(
            id="generated_old",
            course_id="x_cup_aaa111",
            runtime_course_key="x_cup_slot_1",
            gp_difficulty="novice",
        ),
        TrackSamplingEntryConfig(
            id="generated_new",
            course_id="x_cup_bbb222",
            runtime_course_key="x_cup_slot_1",
            gp_difficulty="expert",
        ),
        TrackSamplingEntryConfig(
            id="generated_other",
            course_id="x_cup_ccc333",
            runtime_course_key="x_cup_slot_2",
            gp_difficulty="novice",
        ),
    )

    rotation = WatchCourseRotation.from_entries(entries)

    assert tuple(target.key for target in rotation.targets) == (
        "x_cup_slot_1#difficulty=novice",
        "x_cup_slot_2#difficulty=novice",
        "x_cup_slot_1#difficulty=expert",
    )


def test_watch_course_rotation_switches_difficulty_on_same_course() -> None:
    rotation = _sample_rotation()

    target = rotation.difficulty_target("silence#difficulty=novice", offset=1)

    assert target is not None
    assert target.key == "silence#difficulty=expert"
