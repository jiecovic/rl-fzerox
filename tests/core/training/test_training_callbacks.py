# tests/core/training/test_training_callbacks.py
from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_fzerox.core.domain.x_cup import X_CUP_COURSE
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.runtime_spec.schema import (
    ActionConfig,
    ActionMaskConfig,
    CurriculumConfig,
    EnvConfig,
    PolicyConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainConfig,
)
from rl_fzerox.core.training.runs import build_run_paths, ensure_run_dirs
from rl_fzerox.core.training.session.artifacts import list_recent_checkpoint_dirs
from rl_fzerox.core.training.session.callbacks import build_callbacks
from rl_fzerox.core.training.session.callbacks.checkpoints import resolve_checkpoint_policy
from rl_fzerox.core.training.session.callbacks.track_sampling import (
    TrackSamplingAltBaseline,
    TrackSamplingRuntimePersistence,
)
from rl_fzerox.core.training.session.callbacks.track_sampling.alt_baselines import (
    alt_baseline_reset_variant_key,
)
from rl_fzerox.core.training.session.model import build_ppo_model
from tests.support.fakes import SyntheticBackend, vec_env_fns


def test_resolve_checkpoint_policy_prefers_rollout_interval_for_ppo() -> None:
    policy = resolve_checkpoint_policy(
        TrainConfig(
            algorithm="maskable_hybrid_recurrent_ppo",
            checkpoint_every_rollouts=4,
            num_envs=10,
            save_freq=1_000,
            save_recent_checkpoints=True,
            recent_checkpoint_limit=3,
        )
    )

    assert policy.rollout_interval == 4
    assert policy.step_interval is None
    assert policy.save_recent is True
    assert policy.recent_limit == 3


def test_rollout_checkpoint_callback_saves_and_trims_recent_snapshots(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    callbacks = build_callbacks(
        train_config=TrainConfig(
            algorithm="maskable_hybrid_action_ppo",
            num_envs=1,
            n_steps=4,
            batch_size=4,
            checkpoint_every_rollouts=1,
            save_latest_checkpoint=True,
            save_best_checkpoint=False,
            save_recent_checkpoints=True,
            recent_checkpoint_limit=1,
        ),
        curriculum_config=CurriculumConfig(),
        run_paths=run_paths,
    )
    env = DummyVecEnv(
        vec_env_fns(
            lambda: FZeroXEnv(
                backend=SyntheticBackend(),
                config=EnvConfig(action=ActionConfig(mask=ActionMaskConfig(lean=(0,)))),
            )
        )
    )

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_action_ppo",
                num_envs=1,
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
        )
        model.set_logger(configure(folder=None, format_strings=[]))
        callbacks.init_callback(model)
        callbacks.on_training_start({}, {})
        model.num_timesteps = 4
        callbacks.on_rollout_end()
        model.num_timesteps = 8
        callbacks.on_rollout_end()
    finally:
        env.close()

    checkpoint_dirs = list_recent_checkpoint_dirs(run_paths)
    assert [path.name for path in checkpoint_dirs] == ["000000000008"]
    assert run_paths.latest_model_path.is_file()
    assert _checkpoint_files(checkpoint_dirs[-1]) == {
        "model.zip",
        "policy.metadata.json",
        "policy.zip",
    }


def test_alt_baseline_callback_syncs_stable_entries_in_mixed_x_cup_run(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    stable_state_path = _state_file(tmp_path / "mute.state")
    alt_state_path = _state_file(tmp_path / "mute-alt.state")
    x_cup_state_path = _state_file(tmp_path / "x-cup.state")
    env_config = EnvConfig(
        action=ActionConfig(mask=ActionMaskConfig(lean=(0,))),
        track_sampling=TrackSamplingConfig(
            enabled=True,
            entries=(
                TrackSamplingEntryConfig(
                    id="mute_city_gp_race_novice_blue_falcon",
                    course_id="mute_city",
                    mode="gp_race",
                    gp_difficulty="novice",
                    vehicle="blue_falcon",
                    baseline_state_path=stable_state_path,
                ),
                TrackSamplingEntryConfig(
                    id="x_cup_slot_1_gp_race_novice_blue_falcon",
                    runtime_course_key="x_cup_slot_1",
                    course_id="x_cup_abcd1234",
                    mode=X_CUP_COURSE.race_mode,
                    course_index=X_CUP_COURSE.course_index,
                    gp_difficulty="novice",
                    vehicle="blue_falcon",
                    baseline_state_path=x_cup_state_path,
                    generated_course_kind=X_CUP_COURSE.generated_kind,
                    generated_course_seed=1234,
                    generated_course_hash="abcd1234",
                    generated_course_slot=0,
                    generated_course_generation=1,
                ),
            ),
        ),
    )
    persistence = TrackSamplingRuntimePersistence(
        load=lambda: None,
        save=lambda _: None,
        load_alt_baselines=lambda: (
            TrackSamplingAltBaseline(
                id="alt-a",
                run_id="run",
                course_key="mute_city",
                reset_variant_key=alt_baseline_reset_variant_key(
                    mode="gp_race",
                    gp_difficulty="novice",
                    vehicle="blue_falcon",
                ),
                source_entry_id="mute_city_gp_race_novice_blue_falcon",
                label="chicane approach",
                state_path=alt_state_path,
                weight=1.0,
                enabled=True,
                created_at="2026-06-13T10:00:00+00:00",
                updated_at="2026-06-13T10:00:00+00:00",
            ),
        ),
    )
    callbacks = build_callbacks(
        env_config=env_config,
        train_config=TrainConfig(
            algorithm="maskable_hybrid_action_ppo",
            num_envs=1,
            n_steps=4,
            batch_size=4,
            device="cpu",
        ),
        curriculum_config=CurriculumConfig(),
        run_paths=run_paths,
        track_sampling_runtime_persistence=persistence,
    )
    source_env = FZeroXEnv(backend=SyntheticBackend(), config=env_config)
    env = DummyVecEnv(vec_env_fns(lambda: source_env))
    captured: list[TrackSamplingConfig] = []
    original_setter = source_env.set_track_sampling_config

    def capture_track_sampling_config(config: TrackSamplingConfig) -> None:
        captured.append(config)
        original_setter(config)

    monkeypatch.setattr(source_env, "set_track_sampling_config", capture_track_sampling_config)

    try:
        model = build_ppo_model(
            train_env=env,
            train_config=TrainConfig(
                algorithm="maskable_hybrid_action_ppo",
                n_steps=4,
                batch_size=4,
                device="cpu",
            ),
            policy_config=PolicyConfig(),
            tensorboard_log=None,
        )
        model.set_logger(configure(folder=None, format_strings=[]))
        callbacks.init_callback(model)
        callbacks.on_training_start({}, {})
    finally:
        env.close()

    assert captured
    projected = captured[-1]
    assert [entry.id for entry in projected.entries] == [
        "mute_city_gp_race_novice_blue_falcon",
        "mute_city_gp_race_novice_blue_falcon__alt_alt-a",
        "x_cup_slot_1_gp_race_novice_blue_falcon",
    ]
    assert projected.entries[1].alt_baseline_id == "alt-a"
    assert projected.entries[2].generated_course_kind == X_CUP_COURSE.generated_kind


def _checkpoint_files(path: Path) -> set[str]:
    return {child.name for child in path.iterdir()}


def _state_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"state")
    return path.resolve()
