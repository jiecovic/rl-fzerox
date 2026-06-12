# tests/core/training/test_training_checkpoint_artifacts.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.engine_tuning import (
    ENGINE_TUNING_STATE_VERSION,
    EngineTuningCandidateState,
    EngineTuningRuntimeState,
)
from rl_fzerox.core.training.runs import (
    build_run_paths,
    ensure_run_dirs,
)
from rl_fzerox.core.training.session.artifacts import (
    PolicyArtifactMetadata,
    list_recent_checkpoint_dirs,
    load_engine_tuning_checkpoint_state,
    load_policy_artifact_metadata,
    save_artifacts_atomically,
    save_recent_checkpoint_artifacts,
    trim_recent_checkpoint_artifacts,
)
from tests.core.training.training_artifacts_support import (
    _FakeModel,
)


def test_recent_checkpoint_artifacts_use_timestep_directories_and_trim(
    tmp_path: Path,
) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    model = _FakeModel()
    metadata = PolicyArtifactMetadata(
        curriculum_stage_index=2,
        curriculum_stage_name="all_open",
        num_timesteps=61_440,
    )

    save_recent_checkpoint_artifacts(
        model,
        run_paths,
        num_timesteps=20_480,
        policy_metadata=metadata,
    )
    save_recent_checkpoint_artifacts(
        model,
        run_paths,
        num_timesteps=40_960,
        policy_metadata=metadata,
    )
    save_recent_checkpoint_artifacts(
        model,
        run_paths,
        num_timesteps=61_440,
        policy_metadata=metadata,
    )
    trim_recent_checkpoint_artifacts(run_paths, keep_last=2)

    checkpoint_dirs = list_recent_checkpoint_dirs(run_paths)
    assert [path.name for path in checkpoint_dirs] == [
        "000000040960",
        "000000061440",
    ]
    assert (checkpoint_dirs[-1] / "model.zip").is_file()
    assert (checkpoint_dirs[-1] / "policy.zip").is_file()
    assert load_policy_artifact_metadata(checkpoint_dirs[-1] / "policy.zip") == metadata


def test_save_artifacts_atomically_persists_policy_stage_metadata(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)

    save_artifacts_atomically(
        model=_FakeModel(),
        model_path=run_paths.latest_model_path,
        policy_path=run_paths.latest_policy_path,
        policy_metadata=PolicyArtifactMetadata(
            curriculum_stage_index=1,
            curriculum_stage_name="lean_enabled",
            num_timesteps=123_456,
        ),
    )

    metadata = load_policy_artifact_metadata(run_paths.latest_policy_path)

    assert metadata == PolicyArtifactMetadata(
        curriculum_stage_index=1,
        curriculum_stage_name="lean_enabled",
        num_timesteps=123_456,
    )


def test_save_artifacts_atomically_persists_engine_tuning_checkpoint(
    tmp_path: Path,
) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    state = EngineTuningRuntimeState(
        version=ENGINE_TUNING_STATE_VERSION,
        update_count=7,
        candidates=(
            EngineTuningCandidateState(
                context_key="mute_city|blue_falcon",
                course_key="mute_city",
                vehicle_id="blue_falcon",
                engine_setting_raw_value=65,
                finish_count=2,
                decayed_count=2.5,
                decayed_score_total=3.75,
                score_total=4.5,
                best_score=2.0,
                best_time_ms=90_000,
            ),
        ),
    )

    save_artifacts_atomically(
        model=_FakeModel(),
        model_path=run_paths.latest_model_path,
        policy_path=run_paths.latest_policy_path,
        engine_tuning_state=state,
    )

    assert load_engine_tuning_checkpoint_state(run_paths.latest_policy_path) == state
