# tests/core/training/test_training_artifacts.py
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf
from pytest import MonkeyPatch, raises

from rl_fzerox.core.config.schema import (
    ActionConfig,
    CurriculumConfig,
    CurriculumStageConfig,
    EmulatorConfig,
    EnvConfig,
    PolicyConfig,
    RewardConfig,
    TrackConfig,
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
    TrainAppConfig,
    TrainConfig,
    WatchAppConfig,
    WatchConfig,
)
from rl_fzerox.core.training.runs import (
    apply_train_run_to_watch_config,
    baseline_materializer,
    build_run_paths,
    build_watch_session_paths,
    continue_run_paths,
    ensure_run_dirs,
    ensure_watch_session_dirs,
    load_train_run_config,
    materialize_train_run_config,
    materialize_watch_session_config,
    reserve_run_paths,
    resolve_latest_model_path,
    resolve_latest_policy_path,
    resolve_model_artifact_path,
    resolve_policy_artifact_path,
    save_train_run_config,
)
from rl_fzerox.core.training.runs.baseline_race_start import RaceStartVariant
from rl_fzerox.core.training.session.artifacts import (
    PolicyArtifactMetadata,
    load_policy_artifact_metadata,
    save_artifacts_atomically,
)


class _FakeSaveable:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def save(self, path: str) -> None:
        Path(path).write_bytes(self._payload)


class _FakeModel:
    def __init__(self) -> None:
        self.policy = _FakeSaveable(b"policy")

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"model")


def _patch_fake_boot_materializer(
    monkeypatch: MonkeyPatch,
    *,
    payload: bytes = b"generated",
) -> list[RaceStartVariant]:
    captured_variants: list[RaceStartVariant] = []

    class FakeEmulator:
        def __init__(self, **_: object) -> None:
            pass

        def save_state(self, path: Path) -> None:
            path.write_bytes(payload)

        def close(self) -> None:
            pass

    def fake_materialize_time_attack_race_start_from_boot(
        *,
        emulator: object,
        variant: RaceStartVariant,
    ) -> None:
        del emulator
        captured_variants.append(variant)

    monkeypatch.setattr(baseline_materializer, "Emulator", FakeEmulator)
    monkeypatch.setattr(
        baseline_materializer,
        "materialize_time_attack_race_start_from_boot",
        fake_materialize_time_attack_race_start_from_boot,
    )
    return captured_variants


def _required_baseline_path(entry: TrackSamplingEntryConfig) -> Path:
    assert entry.baseline_state_path is not None
    return entry.baseline_state_path


def test_train_run_config_round_trip_and_watch_inheritance(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    train_runtime_dir = tmp_path / "runtime"
    watch_runtime_dir = tmp_path / "watch-runtime"
    core_path.touch()
    rom_path.touch()
    train_runtime_dir.mkdir()
    watch_runtime_dir.mkdir()

    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            runtime_dir=train_runtime_dir,
        ),
        env=EnvConfig(action_repeat=3),
        reward=RewardConfig(progress_bucket_reward=9.0),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=train_config.train.output_root,
        run_name=train_config.train.run_name,
    )
    ensure_run_dirs(run_paths)

    save_train_run_config(config=train_config, run_dir=run_paths.run_dir)
    loaded_train_config = load_train_run_config(run_paths.run_dir)

    watch_config = WatchAppConfig(
        seed=999,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            runtime_dir=watch_runtime_dir,
        ),
        env=EnvConfig(action_repeat=1),
        reward=RewardConfig(progress_bucket_reward=1.0),
        watch=WatchConfig(control_fps=30.0, render_fps=30.0),
    )
    merged_watch_config = apply_train_run_to_watch_config(
        watch_config,
        run_dir=run_paths.run_dir,
        train_config=loaded_train_config,
    )

    assert loaded_train_config.train.output_root == train_config.train.output_root.resolve()
    assert merged_watch_config.seed == 123
    assert loaded_train_config.emulator.runtime_dir == train_runtime_dir.resolve()
    assert merged_watch_config.emulator.runtime_dir == watch_runtime_dir.resolve()
    assert merged_watch_config.env.action_repeat == 3
    assert merged_watch_config.reward.progress_bucket_reward == 9.0
    assert merged_watch_config.watch.policy_run_dir == run_paths.run_dir
    assert merged_watch_config.watch.control_fps == 30.0
    assert merged_watch_config.watch.render_fps == 30.0


def test_save_train_run_config_persists_action_branches_without_adapter_fields(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(
            action=ActionConfig.model_validate(
                {
                    "branches": {
                        "steer": {
                            "type": "continuous",
                            "response_power": 1.0,
                        },
                        "gas": {
                            "type": "discrete",
                            "mask": ("idle", "engaged"),
                        },
                        "boost": {
                            "type": "discrete",
                            "mask": ("idle",),
                        },
                        "lean": {
                            "type": "discrete",
                            "mask": ("idle", "left", "right"),
                            "mode": "release_cooldown",
                        },
                    }
                }
            ),
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="exp_v3_cnn"),
    )
    run_paths = build_run_paths(
        output_root=train_config.train.output_root,
        run_name=train_config.train.run_name,
    )
    ensure_run_dirs(run_paths)

    config_path = save_train_run_config(config=train_config, run_dir=run_paths.run_dir)

    saved = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert isinstance(saved, dict)
    env_data = saved["env"]
    assert isinstance(env_data, dict)
    action_data = env_data["action"]
    assert isinstance(action_data, dict)
    assert set(action_data) == {"branches"}
    branches_data = action_data["branches"]
    assert isinstance(branches_data, dict)
    boost_data = branches_data["boost"]
    assert isinstance(boost_data, dict)
    assert "decision_interval_frames" not in boost_data
    assert "request_lockout_frames" not in boost_data

    loaded_config = load_train_run_config(run_paths.run_dir)
    action_config = loaded_config.env.action.runtime()

    assert action_config.name == "hybrid_steer_gas_boost_lean"
    assert action_config.boost_decision_interval_frames == 1
    assert action_config.boost_request_lockout_frames == 5

def test_watch_inheritance_preserves_local_baseline_when_run_snapshot_lacks_it(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    baseline_state_path = tmp_path / "first-race.state"
    core_path.touch()
    rom_path.touch()
    baseline_state_path.write_bytes(b"baseline")

    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(action_repeat=3),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=train_config.train.output_root,
        run_name=train_config.train.run_name,
    )
    ensure_run_dirs(run_paths)

    watch_config = WatchAppConfig(
        seed=999,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            baseline_state_path=baseline_state_path,
        ),
        env=EnvConfig(action_repeat=1),
        watch=WatchConfig(control_fps=30.0, render_fps=30.0),
    )

    merged_watch_config = apply_train_run_to_watch_config(
        watch_config,
        run_dir=run_paths.run_dir,
        train_config=train_config,
    )

    assert merged_watch_config.emulator.baseline_state_path == baseline_state_path.resolve()


def test_watch_inheritance_uses_train_camera_setting(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()

    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(action_repeat=3, camera_setting="regular"),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    watch_config = WatchAppConfig(
        seed=999,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(action_repeat=1, camera_setting="close_behind"),
        watch=WatchConfig(control_fps=30.0, render_fps=30.0),
    )

    merged_watch_config = apply_train_run_to_watch_config(
        watch_config,
        run_dir=tmp_path / "runs" / "ppo_cnn_0001",
        train_config=train_config,
    )

    assert merged_watch_config.env.action_repeat == 3
    assert merged_watch_config.env.camera_setting == "regular"

def test_materialize_train_run_config_preserves_resume_artifact_source(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    init_run_dir = tmp_path / "runs" / "ppo_cnn_0042"
    core_path.touch()
    rom_path.touch()
    init_run_dir.mkdir(parents=True)

    train_config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(
            output_root=tmp_path / "runs",
            run_name="ppo_cnn",
            resume_run_dir=init_run_dir,
            resume_artifact="latest",
        ),
    )
    run_paths = build_run_paths(
        output_root=train_config.train.output_root,
        run_name=train_config.train.run_name,
    )

    materialized = materialize_train_run_config(train_config, run_paths=run_paths)

    assert materialized.train.resume_run_dir == init_run_dir.resolve()
    assert materialized.train.resume_artifact == "latest"


def test_materialize_train_run_config_does_not_copy_init_run_baseline(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    init_baseline_path = tmp_path / "runs" / "source_0001" / "baselines" / "baseline.state"
    core_path.touch()
    rom_path.touch()
    init_baseline_path.parent.mkdir(parents=True)
    init_baseline_path.write_bytes(b"init-run-baseline")
    (init_baseline_path.parent.parent / "train_config.yaml").write_text(
        "\n".join(
            [
                "seed: 7",
                "emulator:",
                f"  core_path: {core_path}",
                f"  rom_path: {rom_path}",
                f"  baseline_state_path: {init_baseline_path}",
                "env: {}",
                "reward: {}",
                "policy: {}",
                "curriculum: {}",
                "train:",
                "  algorithm: maskable_ppo",
            ]
        ),
        encoding="utf-8",
    )

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        ),
        env=EnvConfig(),
        track=TrackConfig(
            id="mute_city_time_attack_blue_falcon_balanced",
            course_index=0,
            mode="time_attack",
            vehicle="blue_falcon",
            engine_setting="balanced",
            engine_setting_raw_value=50,
        ),
        policy=PolicyConfig(),
        train=TrainConfig(
            output_root=tmp_path / "runs",
            run_name="target",
            resume_run_dir=init_baseline_path.parent.parent,
            resume_artifact="latest",
        ),
    )
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    ensure_run_dirs(run_paths)
    _patch_fake_boot_materializer(monkeypatch, payload=b"current-baseline")

    materialized = materialize_train_run_config(
        config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
    )

    assert materialized.emulator.baseline_state_path is not None
    assert materialized.emulator.baseline_state_path.parent == run_paths.baselines_dir
    assert materialized.emulator.baseline_state_path.read_bytes() == b"current-baseline"
    assert materialized.train.resume_run_dir == init_baseline_path.parent.parent.resolve()


def test_resolve_latest_model_path_prefers_latest_over_best_and_final(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    run_paths.final_model_path.write_bytes(b"final")
    run_paths.best_model_path.write_bytes(b"best")
    run_paths.latest_model_path.write_bytes(b"latest")

    resolved_model_path = resolve_latest_model_path(run_paths.run_dir)

    assert resolved_model_path == run_paths.latest_model_path


def test_resolve_latest_policy_path_prefers_latest_over_best_and_final(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    run_paths.final_policy_path.write_bytes(b"final-policy")
    run_paths.best_policy_path.write_bytes(b"best-policy")
    run_paths.latest_policy_path.write_bytes(b"latest-policy")

    resolved_policy_path = resolve_latest_policy_path(run_paths.run_dir)

    assert resolved_policy_path == run_paths.latest_policy_path


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


def test_resolve_best_policy_path_requires_best_artifact(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    run_paths.final_policy_path.write_bytes(b"final-policy")
    run_paths.best_policy_path.write_bytes(b"best-policy")
    run_paths.latest_policy_path.write_bytes(b"latest-policy")

    resolved_policy_path = resolve_policy_artifact_path(
        run_paths.run_dir,
        artifact="best",
    )

    assert resolved_policy_path == run_paths.best_policy_path


def test_resolve_final_model_path_requires_final_artifact(tmp_path: Path) -> None:
    run_paths = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(run_paths)
    run_paths.final_model_path.write_bytes(b"final")
    run_paths.best_model_path.write_bytes(b"best")
    run_paths.latest_model_path.write_bytes(b"latest")

    resolved_model_path = resolve_model_artifact_path(
        run_paths.run_dir,
        artifact="final",
    )

    assert resolved_model_path == run_paths.final_model_path


def test_build_run_paths_allocates_numbered_run_directories(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    first = build_run_paths(output_root=output_root, run_name="ppo_cnn")
    ensure_run_dirs(first)
    second = build_run_paths(output_root=output_root, run_name="ppo_cnn")

    assert first.run_dir.name == "ppo_cnn_0001"
    assert second.run_dir.name == "ppo_cnn_0002"
    assert first.runtime_root == first.run_dir / "runtime"
    assert first.env_runtime_dir(0) == first.run_dir / "runtime" / "env_000"
    assert first.baselines_dir == first.run_dir / "baselines"
    assert first.baseline_state_path == first.run_dir / "baselines" / "baseline.state"


def test_reserve_run_paths_never_reuses_existing_run_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"
    (output_root / "ppo_cnn_0001").mkdir(parents=True)

    reserved = reserve_run_paths(output_root=output_root, run_name="ppo_cnn")

    assert reserved.run_dir.name == "ppo_cnn_0002"
    assert reserved.run_dir.is_dir()


def test_materialize_train_run_config_rejects_source_state_copy(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    source_baseline_path = tmp_path / "shared.state"
    core_path.touch()
    rom_path.touch()
    source_baseline_path.write_bytes(b"baseline")

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            baseline_state_path=source_baseline_path,
        ),
        env=EnvConfig(),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )

    ensure_run_dirs(run_paths)
    with raises(ValueError, match="must be generated by the materializer"):
        materialize_train_run_config(
            config,
            run_paths=run_paths,
            baseline_cache_root=tmp_path / "cache",
        )


def test_materialize_train_run_config_reuses_baseline_materializer_cache(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    cache_root = tmp_path / "cache"
    core_path.touch()
    rom_path.touch()
    captured_variants = _patch_fake_boot_materializer(monkeypatch)

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(camera_setting="close_behind"),
        track=TrackConfig(
            id="mute_city_time_attack_blue_falcon_balanced",
            course_index=0,
            mode="time_attack",
            vehicle="blue_falcon",
            engine_setting="balanced",
            engine_setting_raw_value=50,
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    first_run = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    ensure_run_dirs(first_run)
    first_materialized = materialize_train_run_config(
        config,
        run_paths=first_run,
        baseline_cache_root=cache_root,
    )
    second_run = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    ensure_run_dirs(second_run)
    second_materialized = materialize_train_run_config(
        config,
        run_paths=second_run,
        baseline_cache_root=cache_root,
    )

    assert first_materialized.emulator.baseline_state_path is not None
    assert second_materialized.emulator.baseline_state_path is not None
    first_metadata = json.loads(
        first_materialized.emulator.baseline_state_path.with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    second_metadata = json.loads(
        second_materialized.emulator.baseline_state_path.with_suffix(".json").read_text(
            encoding="utf-8"
        )
    )
    assert first_metadata["cache_key"] == second_metadata["cache_key"]
    assert first_metadata["materializer_mode"] == "boot_menu_time_attack"
    assert len(captured_variants) == 1
    assert len(list(cache_root.glob("*.state"))) == 1


def test_materialize_train_run_config_reuses_existing_run_local_baseline_for_in_place_continue(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    captured_variants = _patch_fake_boot_materializer(monkeypatch)

    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    run_dir.mkdir(parents=True)
    run_paths = continue_run_paths(run_dir)
    ensure_run_dirs(run_paths)

    baseline_path = run_paths.baselines_dir / "mute_city__5775c35f3d88.state"
    baseline_path.write_bytes(b"baseline")
    baseline_path.with_suffix(".json").write_text(
        json.dumps({"cache_key": "5775c35f3d88f00dbabe"}, indent=2) + "\n",
        encoding="utf-8",
    )

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            baseline_state_path=baseline_path,
        ),
        env=EnvConfig(camera_setting="close_behind"),
        track=TrackConfig(
            id="mute_city_time_attack_blue_falcon_balanced",
            baseline_state_path=baseline_path,
            course_index=0,
            mode="time_attack",
            vehicle="blue_falcon",
            engine_setting="balanced",
            engine_setting_raw_value=50,
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )

    materialized = materialize_train_run_config(
        config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
    )

    assert materialized.emulator.baseline_state_path == baseline_path
    assert materialized.track.baseline_state_path == baseline_path
    assert captured_variants == []


def test_materialize_train_run_config_rewrites_track_sampling_baselines(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    captured_variants = _patch_fake_boot_materializer(monkeypatch)

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="mute_city_time_attack_blue_falcon_balanced",
                        course_id="mute_city",
                        course_index=0,
                        mode="time_attack",
                        vehicle="blue_falcon",
                        engine_setting="balanced",
                        engine_setting_raw_value=50,
                    ),
                    TrackSamplingEntryConfig(
                        id="silence_time_attack_blue_falcon_balanced",
                        course_id="silence",
                        course_index=1,
                        mode="time_attack",
                        vehicle="blue_falcon",
                        engine_setting="balanced",
                        engine_setting_raw_value=50,
                    ),
                ),
            ),
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    ensure_run_dirs(run_paths)

    materialized = materialize_train_run_config(
        config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
    )

    entries = materialized.env.track_sampling.entries
    first_baseline_path = _required_baseline_path(entries[0])
    second_baseline_path = _required_baseline_path(entries[1])
    assert first_baseline_path.parent == run_paths.baselines_dir
    assert second_baseline_path.parent == run_paths.baselines_dir
    assert first_baseline_path.read_bytes() == b"generated"
    assert second_baseline_path.read_bytes() == b"generated"
    assert first_baseline_path.with_suffix(".json").is_file()
    assert second_baseline_path.with_suffix(".json").is_file()
    assert [variant.course_index for variant in captured_variants] == [0, 1]


def test_materialize_train_run_config_generates_race_start_engine_variant(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    captured_variants = _patch_fake_boot_materializer(monkeypatch)

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            race_intro_target_timer=38,
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="silence_time_attack_blue_falcon_max_speed",
                        course_index=1,
                        mode="time_attack",
                        vehicle="blue_falcon",
                        engine_setting="max_speed",
                        engine_setting_raw_value=100,
                    ),
                ),
            ),
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    ensure_run_dirs(run_paths)

    materialized = materialize_train_run_config(
        config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
    )

    entry = materialized.env.track_sampling.entries[0]
    baseline_path = _required_baseline_path(entry)
    assert baseline_path.read_bytes() == b"generated"
    metadata = json.loads(baseline_path.with_suffix(".json").read_text())
    assert metadata["materializer_mode"] == "boot_menu_time_attack"
    assert len(captured_variants) == 1
    variant = captured_variants[0]
    assert variant.course_index == 1
    assert variant.mode == "time_attack"
    assert variant.character_index == 0
    assert variant.engine_setting_raw_value == 100
    assert variant.race_intro_target_timer == 38


def test_materialize_train_run_config_reuses_target_variant_cache_without_source(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    cache_root = tmp_path / "cache"
    core_path.touch()
    rom_path.touch()
    captured_variants = _patch_fake_boot_materializer(monkeypatch, payload=b"generated-target")

    def build_config() -> TrainAppConfig:
        return TrainAppConfig(
            seed=123,
            emulator=EmulatorConfig(
                core_path=core_path,
                rom_path=rom_path,
                renderer="angrylion",
            ),
            env=EnvConfig(
                race_intro_target_timer=38,
                track_sampling=TrackSamplingConfig(
                    enabled=True,
                    sampling_mode="balanced",
                    entries=(
                        TrackSamplingEntryConfig(
                            id="silence_time_attack_blue_falcon_max_speed",
                            course_id="silence",
                            course_index=1,
                            mode="time_attack",
                            vehicle="blue_falcon",
                            engine_setting="max_speed",
                            engine_setting_raw_value=100,
                        ),
                    ),
                ),
            ),
            policy=PolicyConfig(),
            train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
        )

    first_run = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(first_run)
    first_materialized = materialize_train_run_config(
        build_config(),
        run_paths=first_run,
        baseline_cache_root=cache_root,
    )
    second_run = build_run_paths(output_root=tmp_path / "runs", run_name="ppo_cnn")
    ensure_run_dirs(second_run)
    second_materialized = materialize_train_run_config(
        build_config(),
        run_paths=second_run,
        baseline_cache_root=cache_root,
    )

    first_entry = first_materialized.env.track_sampling.entries[0]
    second_entry = second_materialized.env.track_sampling.entries[0]
    first_baseline_path = _required_baseline_path(first_entry)
    second_baseline_path = _required_baseline_path(second_entry)
    assert first_baseline_path.read_bytes() == b"generated-target"
    assert second_baseline_path.read_bytes() == b"generated-target"
    assert captured_variants == [
        RaceStartVariant(
            course_index=1,
            mode="time_attack",
            character_index=0,
            machine_select_slot=0,
            engine_setting_raw_value=100,
            race_intro_target_timer=38,
        )
    ]
    assert first_baseline_path.with_suffix(".json").is_file()
    assert second_baseline_path.with_suffix(".json").is_file()


def test_materialize_train_run_config_generates_vehicle_variant(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    captured_variants = _patch_fake_boot_materializer(monkeypatch)

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(
            track_sampling=TrackSamplingConfig(
                enabled=True,
                sampling_mode="balanced",
                entries=(
                    TrackSamplingEntryConfig(
                        id="silence_time_attack_white_cat_engine_70",
                        course_index=1,
                        mode="time_attack",
                        vehicle="white_cat",
                        engine_setting="engine_70",
                        engine_setting_raw_value=70,
                    ),
                ),
            ),
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    ensure_run_dirs(run_paths)

    materialized = materialize_train_run_config(
        config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
    )

    entry = materialized.env.track_sampling.entries[0]
    assert entry.baseline_state_path is not None
    assert entry.baseline_state_path.read_bytes() == b"generated"
    assert captured_variants[0].character_index == 4
    assert captured_variants[0].machine_select_slot == 4
    assert captured_variants[0].engine_setting_raw_value == 70


def test_materialize_train_run_config_rewrites_curriculum_track_sampling(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    core_path.touch()
    rom_path.touch()
    _patch_fake_boot_materializer(monkeypatch, payload=b"stage")

    config = TrainAppConfig(
        seed=123,
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
        env=EnvConfig(),
        curriculum=CurriculumConfig(
            enabled=True,
            stages=(
                CurriculumStageConfig(
                    name="stage",
                    track_sampling=TrackSamplingConfig(
                        enabled=True,
                        entries=(
                            TrackSamplingEntryConfig(
                                id="stage_track",
                                course_index=0,
                                mode="time_attack",
                                vehicle="blue_falcon",
                                engine_setting="balanced",
                                engine_setting_raw_value=50,
                            ),
                        ),
                    ),
                ),
            ),
        ),
        policy=PolicyConfig(),
        train=TrainConfig(output_root=tmp_path / "runs", run_name="ppo_cnn"),
    )
    run_paths = build_run_paths(
        output_root=config.train.output_root,
        run_name=config.train.run_name,
    )
    ensure_run_dirs(run_paths)

    materialized = materialize_train_run_config(
        config,
        run_paths=run_paths,
        baseline_cache_root=tmp_path / "cache",
    )

    stage = materialized.curriculum.stages[0]
    assert stage.track_sampling is not None
    entry = stage.track_sampling.entries[0]
    baseline_path = _required_baseline_path(entry)
    assert baseline_path.parent == run_paths.baselines_dir
    assert baseline_path.read_bytes() == b"stage"


def test_build_watch_session_paths_uses_run_local_watch_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "ppo_cnn_0001"
    baseline_state_path = tmp_path / "baseline.state"

    session_paths = build_watch_session_paths(
        run_dir=run_dir,
        runtime_dir=tmp_path / "runtime",
        baseline_state_path=baseline_state_path,
        session_name="session-001",
    )

    assert session_paths.session_dir == run_dir / "watch" / "session-001"
    assert session_paths.runtime_dir == run_dir / "watch" / "session-001" / "runtime"
    assert session_paths.baseline_state_path == (
        run_dir / "watch" / "session-001" / "baseline.state"
    )


def test_materialize_watch_session_config_isolates_runtime_and_baseline(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    baseline_state_path = tmp_path / "shared.state"
    core_path.touch()
    rom_path.touch()
    baseline_state_path.write_bytes(b"baseline")

    watch_config = WatchAppConfig(
        seed=7,
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
            runtime_dir=tmp_path / "runtime",
            baseline_state_path=baseline_state_path,
        ),
        watch=WatchConfig(),
    )

    materialized = materialize_watch_session_config(
        watch_config,
        run_dir=tmp_path / "runs" / "ppo_cnn_0001",
        session_name="session-001",
    )

    assert materialized.emulator.runtime_dir == (
        tmp_path / "runs" / "ppo_cnn_0001" / "watch" / "session-001" / "runtime"
    )
    assert materialized.emulator.baseline_state_path == (
        tmp_path / "runs" / "ppo_cnn_0001" / "watch" / "session-001" / "baseline.state"
    )
    assert materialized.emulator.baseline_state_path is not None
    assert materialized.emulator.baseline_state_path.read_bytes() == b"baseline"


def test_ensure_watch_session_dirs_creates_runtime_root(tmp_path: Path) -> None:
    paths = build_watch_session_paths(
        run_dir=None,
        runtime_dir=tmp_path / "runtime",
        baseline_state_path=None,
        session_name="session-001",
    )

    ensure_watch_session_dirs(paths)

    assert paths.session_dir.is_dir()
    assert paths.runtime_dir.is_dir()
