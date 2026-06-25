# tests/ui/test_viewer_runtime_observation.py
"""Watch runtime tests for observation feature names and zeroing.

These cases cover state-feature masking, dropout-derived Watch zeroing, policy
race inheritance, and UI fallback naming for image-state observations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rl_fzerox.core.career_mode.policy import CareerModePolicyControl
from rl_fzerox.core.envs.observations import (
    ImageStateObservation,
    observation_state,
    state_feature_names,
)
from rl_fzerox.core.manager import default_managed_run_config
from rl_fzerox.core.manager.models import ManagedRun, ManagedSaveCourseSetup
from rl_fzerox.core.runtime_spec.schema import (
    ActionConfig,
    EmulatorConfig,
    EnvConfig,
    ObservationConfig,
    ObservationStateComponentConfig,
    StateFeatureDropoutGroupConfig,
    TrainConfig,
    WatchAppConfig,
)
from rl_fzerox.core.training.inference import LoadedPolicy, PolicyRunner
from rl_fzerox.ui.watch.runtime.career_mode.session import CareerModeRuntimeSession
from rl_fzerox.ui.watch.runtime.observation import (
    apply_watch_state_feature_zeroing,
    configured_watch_zeroed_features,
)
from rl_fzerox.ui.watch.view.screen.render import _observation_state_feature_names
from tests.support.fakes import SyntheticBackend
from tests.ui.viewer_runtime_support import PolicyStub, sample_image, sample_state


def test_watch_state_feature_zeroing_masks_selected_features_without_mutating_source() -> None:
    observation: ImageStateObservation = {
        "image": sample_image(),
        "state": sample_state([0.0, 2.0, 3.0]),
    }
    info: dict[str, object] = {
        "observation_state_features": (
            "vehicle_state.speed_kph_norm",
            "machine_context.energy_norm",
            "course_context.course_builtin_0",
        ),
        "observation_zeroed_state_features": ("vehicle_state.speed_kph_norm",),
    }

    masked_observation, masked_info = apply_watch_state_feature_zeroing(
        observation,
        info,
        watch_zeroed_features=frozenset({"machine_context.energy_norm"}),
    )

    assert observation["state"][1] == 2.0
    masked_state = observation_state(masked_observation)
    assert masked_state is not None
    assert masked_state[1] == 0.0
    assert masked_state[2] == 3.0
    assert masked_info["watch_zeroed_state_features"] == ("machine_context.energy_norm",)
    assert masked_info["observation_zeroed_state_features"] == (
        "machine_context.energy_norm",
        "vehicle_state.speed_kph_norm",
    )


def test_watch_state_feature_zeroing_supports_component_level_course_toggle() -> None:
    observation: ImageStateObservation = {
        "image": sample_image(),
        "state": sample_state([1.0, 0.0, 2.0]),
    }
    info = {
        "observation_state_features": (
            "course_context.course_builtin_0",
            "course_context.course_builtin_1",
            "vehicle_state.speed_kph_norm",
        ),
        "observation_zeroed_state_features": (),
    }

    masked_observation, masked_info = apply_watch_state_feature_zeroing(
        observation,
        info,
        watch_zeroed_features=frozenset({"course_context"}),
    )

    masked_state = observation_state(masked_observation)
    assert masked_state is not None
    assert list(masked_state) == [0.0, 0.0, 2.0]
    assert masked_info["watch_zeroed_state_features"] == ("course_context",)


def test_configured_watch_zeroed_features_inherits_dropout_one_groups(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    state_components = (
        ObservationStateComponentConfig(name="track_position", progress_source="segment_progress"),
        ObservationStateComponentConfig(name="course_context", encoding="one_hot_builtin"),
    )
    feature_names = state_feature_names(
        state_components=tuple(component.data() for component in state_components),
    )
    course_feature_names = tuple(
        feature_name for feature_name in feature_names if feature_name.startswith("course_context.")
    )
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            action=ActionConfig(
                layout_continuous_axes=("steer",),
                layout_discrete_axes=("gas", "boost", "lean"),
            ),
            observation=ObservationConfig(mode="image_state", state_components=state_components),
        ),
        train=TrainConfig(
            state_feature_dropout_groups=(
                StateFeatureDropoutGroupConfig(
                    feature_names=("track_position.edge_ratio",),
                    dropout_prob=1.0,
                ),
                StateFeatureDropoutGroupConfig(
                    feature_names=course_feature_names,
                    dropout_prob=1.0,
                ),
                StateFeatureDropoutGroupConfig(
                    feature_names=("track_position.lap_progress",),
                    dropout_prob=0.6,
                ),
            )
        ),
    )

    assert configured_watch_zeroed_features(config) == frozenset(
        {"track_position.edge_ratio", "course_context"}
    )


def test_career_session_inherits_policy_dropout_one_zeroed_features(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = default_managed_run_config().model_copy(deep=True)
    config.observation.state_feature_dropouts = (
        config.observation.state_feature_dropouts[0].model_copy(update={"dropout_prob": 1.0}),
        config.observation.state_feature_dropouts[1].model_copy(update={"dropout_prob": 0.5}),
    )
    run = ManagedRun(
        id="policy-run",
        name="Policy Run",
        status="finished",
        config=config,
        config_hash="hash",
        run_dir=tmp_path / "policy-run",
        created_at="2026-01-01T00:00:00Z",
        lineage_id="policy-run",
    )
    policy_path = run.run_dir / "checkpoints" / "latest.zip"
    policy_path.parent.mkdir(parents=True)
    policy_path.touch()
    policy_control = CareerModePolicyControl(
        course_setup=ManagedSaveCourseSetup(
            id="course-setup",
            save_game_id="save",
            policy_run_id=run.id,
            policy_artifact="latest",
            engine_setting_raw_value=50,
            difficulty="novice",
            cup_id="jack",
            course_id="mute_city",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        ),
        policy_run=run,
        runner=PolicyRunner(
            LoadedPolicy(run_dir=run.run_dir, policy_path=policy_path, artifact="latest"),
            policy=PolicyStub(),
        ),
    )
    emulator: Any = SyntheticBackend()
    session = CareerModeRuntimeSession(
        config=WatchAppConfig(emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path)),
        emulator=emulator,
        native_fps=60.0,
        native_sample_rate=48_000.0,
        native_control_fps=60.0,
        target_control_fps=None,
        target_control_seconds=None,
        watch_zeroed_state_features=frozenset(),
        auxiliary_target_names=(),
    )

    session.begin_policy_race(policy_control=policy_control, seed=7, course_id="mute_city")

    expected_zeroed_features = configured_watch_zeroed_features(
        session.snapshot_config(session.config)
    )
    assert session.watch_zeroed_state_features == expected_zeroed_features
    assert "track_position.edge_ratio" in expected_zeroed_features
    assert "track_position.outside_track_bounds" not in expected_zeroed_features


def test_viewer_state_feature_names_fall_back_to_image_state_config(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    state_components = (
        ObservationStateComponentConfig(name="vehicle_state"),
        ObservationStateComponentConfig(name="control_history", controls=("boost",)),
    )
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            action=ActionConfig(
                layout_continuous_axes=("steer",),
                layout_discrete_axes=("gas", "boost", "lean"),
            ),
            observation=ObservationConfig(
                mode="image_state",
                state_components=state_components,
            ),
        ),
    )

    assert _observation_state_feature_names(config, {}) == state_feature_names(
        state_components=tuple(component.data() for component in state_components),
        split_lean_history=False,
    )
