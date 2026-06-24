# tests/core/training/test_track_sampling_generation_state.py

import pytest

from rl_fzerox.apps.run_manager.api.payloads.track_sampling import (
    track_sampling_state_payload,
)
from rl_fzerox.core.domain.courses import X_CUP_COURSE, generated_x_cup_slot_key
from rl_fzerox.core.runtime_spec.schema import (
    EnvConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    XCupRotationConfig,
)
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    StepBalancedTrackSamplingController,
    TrackSamplingRuntimeEntry,
    TrackSamplingRuntimeState,
    replace_runtime_generation,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.deficit import (
    DeficitBudgetTrackSamplingController,
)


def test_step_balance_runtime_state_exposes_generated_x_cup_generation() -> None:
    slot_key = generated_x_cup_slot_key(2)
    entry = TrackSamplingEntryConfig(
        id="x_cup_slot_blue_falcon",
        course_id="x_cup_abcd1234",
        runtime_course_key=slot_key,
        course_name="X Cup abcd1234",
        mode=X_CUP_COURSE.race_mode,
        course_index=X_CUP_COURSE.course_index,
        generated_course_kind=X_CUP_COURSE.generated_kind,
        generated_course_seed=123,
        generated_course_hash="abcd1234",
        generated_course_slot=2,
        generated_course_generation=3,
    )
    controller = StepBalancedTrackSamplingController.from_configs(
        env_config=EnvConfig(
            action_repeat=2,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="step_balanced",
                entries=(
                    entry,
                    TrackSamplingEntryConfig(
                        id="mute_city_blue_falcon",
                        course_id="mute_city",
                        course_name="Mute City",
                    ),
                ),
            ),
        ),
    )

    assert controller is not None
    x_cup_runtime = next(
        entry for entry in controller.runtime_state().entries if entry.course_key == slot_key
    )

    assert x_cup_runtime.track_id == slot_key
    assert x_cup_runtime.generated_course_slot == 2
    assert x_cup_runtime.generated_course_generation == 3


def test_deficit_budget_runtime_state_tracks_generated_x_cup_generation_stats() -> None:
    slot_key = generated_x_cup_slot_key(0)
    controller = DeficitBudgetTrackSamplingController.from_configs(
        env_config=EnvConfig(
            action_repeat=2,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="deficit_budget",
                deficit_budget_ema_alpha=1.0,
                x_cup_rotation=XCupRotationConfig(enabled=True, ema_alpha=1.0),
                entries=(
                    TrackSamplingEntryConfig(
                        id="x_cup_slot_master",
                        course_id="x_cup_abcd1234",
                        runtime_course_key=slot_key,
                        course_name="X Cup abcd1234",
                        mode=X_CUP_COURSE.race_mode,
                        course_index=X_CUP_COURSE.course_index,
                        generated_course_kind=X_CUP_COURSE.generated_kind,
                        generated_course_seed=123,
                        generated_course_hash="abcd1234",
                        generated_course_slot=0,
                        generated_course_generation=1,
                    ),
                    TrackSamplingEntryConfig(
                        id="mute_city_master",
                        course_id="mute_city",
                        course_name="Mute City",
                    ),
                ),
            ),
        ),
    )

    assert controller is not None
    controller.record_episodes(
        (
            {
                "track_id": "x_cup_slot_master",
                "episode_step": 100,
                "episode_completion_fraction": 0.75,
                "termination_reason": "finished",
            },
        )
    )
    x_cup_runtime = next(
        entry for entry in controller.runtime_state().entries if entry.course_key == slot_key
    )

    assert x_cup_runtime.episode_count == 1
    assert x_cup_runtime.generation_episode_count == 1
    assert x_cup_runtime.generation_finished_episode_count == 1
    assert x_cup_runtime.generation_success_sample_count == 1
    assert x_cup_runtime.generation_ema_completion_fraction == pytest.approx(0.75)
    assert x_cup_runtime.generated_course_slot == 0
    assert x_cup_runtime.generated_course_generation == 1


def test_track_sampling_payload_uses_runtime_generated_x_cup_metadata() -> None:
    slot_key = generated_x_cup_slot_key(1)
    state = TrackSamplingRuntimeState(
        sampling_mode="step_balanced",
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
        adaptive_min_confidence_episodes=24,
        adaptive_confidence_scale=4.0,
        update_count=1,
        episodes_since_update=0,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id=slot_key,
                course_key=slot_key,
                label="X Cup abcd1234",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=100,
                episode_count=1,
                finished_episode_count=0,
                success_sample_count=1,
                ema_episode_frames=100.0,
                ema_completion_fraction=0.2,
                generated_course_slot=1,
                generated_course_generation=4,
            ),
        ),
    )

    payload = track_sampling_state_payload(state)

    entries_payload = payload["entries"]
    assert isinstance(entries_payload, list)
    entry_payload = entries_payload[0]
    assert isinstance(entry_payload, dict)
    assert entry_payload["generated_course_slot"] == 1
    assert entry_payload["generated_course_generation"] == 4


def test_replace_runtime_generation_keeps_slot_history_and_resets_current_generation() -> None:
    slot_key = generated_x_cup_slot_key(0)
    state = TrackSamplingRuntimeState(
        sampling_mode="step_balanced",
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
        adaptive_min_confidence_episodes=24,
        adaptive_confidence_scale=4.0,
        update_count=1,
        episodes_since_update=0,
        entries=(
            TrackSamplingRuntimeEntry(
                track_id=slot_key,
                course_key=slot_key,
                label="X Cup old",
                base_weight=1.0,
                current_weight=1.0,
                completed_frames=5000,
                episode_count=20,
                finished_episode_count=18,
                success_sample_count=20,
                ema_episode_frames=250.0,
                ema_completion_fraction=0.95,
                generation_episode_count=20,
                generation_finished_episode_count=18,
                generation_success_sample_count=20,
                generation_ema_completion_fraction=0.95,
                generated_course_slot=0,
                generated_course_generation=1,
            ),
        ),
    )

    replaced = replace_runtime_generation(
        state,
        course_key=slot_key,
        replacement_label="X Cup new",
        generated_course_slot=0,
        generated_course_generation=2,
        generated_course_id="x_cup_new",
        generated_course_name="X Cup new",
        generated_course_hash="new",
        generated_course_seed=123,
        generated_course_segment_count=256,
        generated_course_length=12345.0,
    )
    entry = replaced.entries[0]

    assert entry.course_key == slot_key
    assert entry.completed_frames == 5000
    assert entry.ema_episode_frames == 250.0
    assert entry.episode_count == 20
    assert entry.success_sample_count == 20
    assert entry.ema_completion_fraction == 0.95
    assert entry.generation_episode_count == 0
    assert entry.generation_success_sample_count == 0
    assert entry.generation_ema_completion_fraction is None
    assert entry.generated_course_generation == 2
