# tests/core/training/test_track_sampling_step_balance_adaptive.py


from rl_fzerox.core.training.session.callbacks.track_sampling import (
    StepBalancedTrackSamplingController,
)


def test_adaptive_step_balance_controller_tilts_weight_toward_lower_completion() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"mute": 1.0, "silence": 1.0},
        sampling_mode="adaptive_step_balanced",
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.5,
        adaptive_target_completion=0.9,
    )

    weights = controller.record_episodes(
        (
            {"track_id": "mute", "episode_step": 200, "episode_completion_fraction": 0.2},
            {"track_id": "silence", "episode_step": 200, "episode_completion_fraction": 0.8},
        )
    )

    assert weights is not None
    assert weights["mute"] > weights["silence"]


def test_adaptive_step_balance_controller_uses_finish_rate_when_completion_is_similar() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"sector_alpha": 1.0, "white_land": 1.0},
        sampling_mode="adaptive_step_balanced",
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
    )

    weights = controller.record_episodes(
        (
            {
                "track_id": "sector_alpha",
                "episode_step": 200,
                "episode_completion_fraction": 0.95,
                "termination_reason": "finished",
            },
            {
                "track_id": "white_land",
                "episode_step": 200,
                "episode_completion_fraction": 0.85,
                "termination_reason": "stalled",
            },
        )
    )

    assert weights is not None
    assert weights["white_land"] > weights["sector_alpha"]


def test_adaptive_step_balance_controller_can_create_large_target_spread() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"easy": 1.0, "hard": 1.0},
        sampling_mode="adaptive_step_balanced",
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.35,
        adaptive_target_completion=0.9,
    )

    controller.record_episodes(
        (
            {
                "track_id": "easy",
                "episode_step": 200,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "hard",
                "episode_step": 200,
                "episode_completion_fraction": 0.0,
                "termination_reason": "stalled",
            },
        )
    )

    runtime = controller.runtime_state()
    weights_by_track = {entry.track_id: entry.current_weight for entry in runtime.entries}

    assert weights_by_track["hard"] > 1.5 * weights_by_track["easy"]


def test_adaptive_step_balance_prioritizes_low_confidence_courses() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"known": 1.0, "unknown": 1.0},
        sampling_mode="adaptive_step_balanced",
        action_repeat=1,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=5.0,
        adaptive_completion_weight=0.0,
        adaptive_target_completion=0.9,
        adaptive_min_confidence_episodes=4,
        adaptive_confidence_scale=5.0,
    )

    weights = controller.record_episodes(
        (
            {
                "track_id": "known",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "unknown",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "known",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "known",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "known",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
        )
    )

    assert weights is not None
    assert weights["unknown"] > weights["known"]


def test_adaptive_step_balance_converts_frame_target_to_reset_weight() -> None:
    controller = StepBalancedTrackSamplingController(
        track_base_weights={"easy_short": 1.0, "hard_long": 1.0},
        sampling_mode="adaptive_step_balanced",
        action_repeat=2,
        update_episodes=2,
        ema_alpha=1.0,
        max_weight_scale=20.0,
        adaptive_completion_weight=1.0,
        adaptive_target_completion=0.9,
    )

    controller.record_episodes(
        (
            {
                "track_id": "easy_short",
                "episode_step": 100,
                "episode_completion_fraction": 1.0,
                "termination_reason": "finished",
            },
            {
                "track_id": "hard_long",
                "episode_step": 400,
                "episode_completion_fraction": 0.0,
                "termination_reason": "stalled",
            },
        )
    )

    runtime = controller.runtime_state()
    entries = {entry.track_id: entry for entry in runtime.entries}
    easy = entries["easy_short"]
    hard = entries["hard_long"]

    assert hard.current_weight > easy.current_weight
    assert hard.current_weight * 400 > easy.current_weight * 100 * 10.0
