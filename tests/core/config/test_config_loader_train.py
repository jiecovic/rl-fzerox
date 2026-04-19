# tests/core/config/test_config_loader_train.py
from __future__ import annotations

from pathlib import Path

from rl_fzerox.core.config import load_train_app_config
from rl_fzerox.core.config.schema import ObservationConfig


def _write_yaml(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def test_observation_state_components_parse_ordered_lego_list() -> None:
    config = ObservationConfig.model_validate(
        {
            "mode": "image_state",
            "state_components": [
                "vehicle_state",
                "track_position",
                "surface_state",
                {"course_context": {"encoding": "one_hot_builtin"}},
                {"control_history": {"length": 2, "controls": ["steer", "thrust"]}},
            ],
        }
    )

    assert config.state_components_data() == (
        {"name": "vehicle_state"},
        {"name": "track_position"},
        {"name": "surface_state"},
        {"name": "course_context", "encoding": "one_hot_builtin"},
        {"name": "control_history", "length": 2, "controls": ("steer", "thrust")},
    )


def test_load_train_app_config_composes_track_registry_entry(
    isolated_repo_layout: tuple[Path, Path],
) -> None:
    project_root, config_root = isolated_repo_layout
    artifacts_dir = project_root / "local" / "test-artifacts"
    runs_dir = project_root / "local" / "runs"
    artifacts_dir.mkdir(parents=True)
    core_path = artifacts_dir / "core.so"
    rom_path = artifacts_dir / "rom.n64"
    baseline_path = artifacts_dir / "time-attack.state"
    core_path.touch()
    rom_path.touch()
    baseline_path.write_bytes(b"baseline")

    _write_yaml(
        config_root / "vehicles" / "blue_falcon.yaml",
        [
            "vehicle:",
            "  id: blue_falcon",
            "  display_name: Blue Falcon",
            "  engine_settings:",
            "    balanced:",
            "      raw_value: 50",
        ],
    )
    _write_yaml(
        config_root / "tracks" / "mute_city_test.yaml",
        [
            "track:",
            "  id: mute_city_test",
            "  display_name: Mute City Test",
            "  course_index: 0",
            "  mode: time_attack",
            "  vehicle: blue_falcon",
            "  engine_setting: balanced",
            "  baseline_state_path: local/test-artifacts/time-attack.state",
            "  records:",
            "    source_label: F-Zero X WR History",
            "    source_url: https://www.fzerowrs.com/x/display.php?track=Mute+City",
            "    non_agg_best:",
            "      time_ms: 67953",
            "      player: Daniel",
            "      date: '2009-07-14'",
            "      mode: PAL",
            "    non_agg_worst:",
            "      time_ms: 69492",
            "      player: POP",
            "      date: '1998-11-xx'",
            "      mode: NTSC",
            "emulator:",
            "  baseline_state_path: ${track.baseline_state_path}",
        ],
    )
    config_path = config_root / "local" / "train.track.yaml"
    _write_yaml(
        config_path,
        [
            "defaults:",
            "  - /tracks/mute_city_test@_global_",
            "  - _self_",
            "emulator:",
            "  core_path: local/test-artifacts/core.so",
            "  rom_path: local/test-artifacts/rom.n64",
            "env:",
            "  track_sampling:",
            "    enabled: true",
            "    sampling_mode: balanced",
            "    entries:",
            "      - id: mute_city_test",
            "        baseline_state_path: local/test-artifacts/time-attack.state",
            "        weight: 0.4",
            "curriculum:",
            "  enabled: true",
            "  stages:",
            "    - name: silence_bias",
            "      track_sampling:",
            "        enabled: true",
            "        sampling_mode: random",
            "        entries:",
            "          - id: silence_test",
            "            baseline_state_path: local/test-artifacts/time-attack.state",
            "            weight: 0.6",
            "train:",
            "  output_root: local/runs",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.track.id == "mute_city_test"
    assert config.track.baseline_state_path == baseline_path.resolve()
    assert config.track.records is not None
    assert config.track.records.non_agg_best is not None
    assert config.track.records.non_agg_best.time_ms == 67953
    assert config.track.records.non_agg_worst is not None
    assert config.track.records.non_agg_worst.player == "POP"
    assert config.emulator.baseline_state_path == baseline_path.resolve()
    assert config.env.track_sampling.enabled is True
    assert config.env.track_sampling.sampling_mode == "balanced"
    assert config.env.track_sampling.entries[0].id == "mute_city_test"
    assert config.env.track_sampling.entries[0].display_name == "Mute City Test"
    assert config.env.track_sampling.entries[0].baseline_state_path == baseline_path.resolve()
    assert config.env.track_sampling.entries[0].weight == 0.4
    assert config.env.track_sampling.entries[0].course_index == 0
    assert config.env.track_sampling.entries[0].records is not None
    assert config.env.track_sampling.entries[0].records.non_agg_best is not None
    assert config.env.track_sampling.entries[0].records.non_agg_best.time_ms == 67953
    assert config.curriculum.stages[0].track_sampling is not None
    assert config.curriculum.stages[0].track_sampling.sampling_mode == "random"
    assert config.curriculum.stages[0].track_sampling.entries[0].baseline_state_path == (
        baseline_path.resolve()
    )
    assert config.curriculum.stages[0].track_sampling.entries[0].weight == 0.6
    assert config.train.output_root == runs_dir.resolve()


def test_load_train_app_config_expands_course_registry_selection(
    isolated_repo_layout: tuple[Path, Path],
) -> None:
    project_root, config_root = isolated_repo_layout
    artifacts_dir = project_root / "local" / "test-artifacts"
    artifacts_dir.mkdir(parents=True)
    core_path = artifacts_dir / "core.so"
    rom_path = artifacts_dir / "rom.n64"
    baseline_path = artifacts_dir / "silence.state"
    core_path.touch()
    rom_path.touch()
    baseline_path.write_bytes(b"baseline")

    _write_yaml(
        config_root / "vehicles" / "blue_falcon.yaml",
        [
            "vehicle:",
            "  id: blue_falcon",
            "  display_name: Blue Falcon",
            "  engine_settings:",
            "    balanced:",
            "      raw_value: 50",
        ],
    )
    _write_yaml(
        config_root / "courses" / "jack" / "silence_test.yaml",
        [
            "course:",
            "  id: silence_test",
            "  display_name: Silence Test",
            "  course_index: 1",
        ],
    )
    _write_yaml(
        config_root / "tracks" / "jack" / "metadata_selected_silence.yaml",
        [
            "track:",
            "  id: silence_test_time_attack_blue_falcon_balanced",
            "  display_name: Silence Test Time Attack - Blue Falcon Balanced",
            "  course_ref: jack/silence_test",
            "  mode: time_attack",
            "  vehicle: blue_falcon",
            "  engine_setting: balanced",
            "  baseline_state_path: local/test-artifacts/silence.state",
        ],
    )
    config_path = config_root / "local" / "train.courses.yaml"
    _write_yaml(
        config_path,
        [
            "emulator:",
            "  core_path: local/test-artifacts/core.so",
            "  rom_path: local/test-artifacts/rom.n64",
            "env:",
            "  track_sampling:",
            "    enabled: true",
            "    sampling_mode: balanced",
            "    courses:",
            "      - silence_test",
            "    baseline:",
            "      mode: time_attack",
            "      vehicle: blue_falcon",
            "      engine_setting: 50",
        ],
    )

    config = load_train_app_config(config_path)

    entry = config.env.track_sampling.entries[0]
    assert entry.id == "silence_test_time_attack_blue_falcon_balanced"
    assert entry.course_ref == "jack/silence_test"
    assert entry.course_id == "silence_test"
    assert entry.course_name == "Silence Test"
    assert entry.course_index == 1
    assert entry.mode == "time_attack"
    assert entry.vehicle == "blue_falcon"
    assert entry.vehicle_name == "Blue Falcon"
    assert entry.engine_setting == "balanced"
    assert entry.baseline_state_path == baseline_path.resolve()


def test_load_train_app_config_reads_policy_activation(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  action_repeat: 2",
            "policy:",
            "  activation: relu",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.policy.activation == "relu"


def test_load_train_app_config_reads_course_context(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  observation:",
            "    mode: image_state",
            "    state_profile: race_core",
            "    course_context: one_hot_builtin",
            "    ground_effect_context: effect_flags",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.env.observation.course_context == "one_hot_builtin"
    assert config.env.observation.ground_effect_context == "effect_flags"


def test_load_train_app_config_reads_auto_extractor_features_dim(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "policy:",
            "  extractor:",
            "    features_dim: auto",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.policy.extractor.features_dim == "auto"


def test_load_train_app_config_reads_compact_deep_extractor_profile(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  observation:",
            "    preset: native_crop_v4",
            "policy:",
            "  extractor:",
            "    conv_profile: compact_deep",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.env.observation.preset == "native_crop_v4"
    assert config.policy.extractor.conv_profile == "compact_deep"


def test_load_train_app_config_resolves_init_run_dir(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    run_dir = tmp_path / "runs" / "ppo_cnn_0042"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    run_dir.mkdir(parents=True)
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "train:",
            "  total_timesteps: 1000",
            f"  init_run_dir: {run_dir}",
            "  init_artifact: latest",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.train.init_run_dir == run_dir.resolve()
    assert config.train.init_artifact == "latest"


def test_load_train_app_config_reads_state_extractor_features_dim(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "policy:",
            "  extractor:",
            "    state_features_dim: 32",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.policy.extractor.state_features_dim == 32


def test_load_train_app_config_reads_fusion_extractor_features_dim(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "policy:",
            "  extractor:",
            "    fusion_features_dim: 512",
            "train:",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.policy.extractor.fusion_features_dim == 512


def test_load_train_app_config_reads_maskable_curriculum_fields(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  action:",
            "    mask:",
            "      lean: [idle]",
            "curriculum:",
            "  enabled: true",
            "  smoothing_episodes: 4",
            "  min_stage_episodes: 2",
            "  stages:",
            "    - name: basic_drive",
            "      until:",
            "        race_laps_completed_mean_gte: 3.0",
            "      action_mask:",
            "        lean: [idle]",
            "      train:",
            "        learning_rate: 0.0001",
            "        n_epochs: 3",
            "        batch_size: 512",
            "        clip_range: 0.15",
            "        ent_coef: 0.01",
            "    - name: lean_enabled",
            "      action_mask:",
            "        lean: [idle, left, right]",
            "train:",
            "  algorithm: maskable_ppo",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.train.algorithm == "maskable_ppo"
    assert config.env.action.mask is not None
    assert config.env.action.mask.lean == ("idle",)
    assert config.env.action.mask.branch_overrides() == {"lean": (0,)}
    assert config.curriculum.enabled is True
    assert config.curriculum.smoothing_episodes == 4
    assert config.curriculum.min_stage_episodes == 2
    assert len(config.curriculum.stages) == 2
    assert config.curriculum.stages[0].until is not None
    assert config.curriculum.stages[0].until.race_laps_completed_mean_gte == 3.0
    assert config.curriculum.stages[0].train is not None
    assert config.curriculum.stages[0].train.learning_rate == 0.0001
    assert config.curriculum.stages[0].train.n_epochs == 3
    assert config.curriculum.stages[0].train.batch_size == 512
    assert config.curriculum.stages[0].train.clip_range == 0.15
    assert config.curriculum.stages[0].train.ent_coef == 0.01
    assert config.curriculum.stages[0].action_mask is not None
    assert config.curriculum.stages[0].action_mask.branch_overrides() == {"lean": (0,)}
    assert config.curriculum.stages[1].action_mask is not None
    assert config.curriculum.stages[1].action_mask.branch_overrides() == {"lean": (0, 1, 2)}


def test_load_train_app_config_reads_recurrent_policy_fields(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "policy:",
            "  recurrent:",
            "    enabled: true",
            "    hidden_size: 512",
            "    n_lstm_layers: 1",
            "    shared_lstm: false",
            "    enable_critic_lstm: true",
            "train:",
            "  algorithm: maskable_recurrent_ppo",
            "  total_timesteps: 1000",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.train.algorithm == "maskable_recurrent_ppo"
    assert config.policy.recurrent.enabled is True
    assert config.policy.recurrent.hidden_size == 512
    assert config.policy.recurrent.n_lstm_layers == 1
    assert config.policy.recurrent.shared_lstm is False
    assert config.policy.recurrent.enable_critic_lstm is True


def test_load_train_app_config_reads_sac_fields(tmp_path: Path) -> None:
    core_path = tmp_path / "mupen64plus_next_libretro.so"
    rom_path = tmp_path / "fzerox.n64"
    config_path = tmp_path / "train.yaml"
    core_path.touch()
    rom_path.touch()
    _write_yaml(
        config_path,
        [
            "seed: 7",
            "emulator:",
            f"  core_path: {core_path}",
            f"  rom_path: {rom_path}",
            "env:",
            "  action:",
            "    name: continuous_steer_drive_lean",
            "    steer_response_power: 0.7",
            "    continuous_drive_mode: always_accelerate",
            "    continuous_drive_deadzone: 0.15",
            "    continuous_lean_deadzone: 0.25",
            "train:",
            "  algorithm: sac",
            "  total_timesteps: 1000",
            "  buffer_size: 30000",
            "  learning_starts: 5000",
            "  train_freq: 1",
            "  gradient_steps: 1",
            "  ent_coef: auto",
        ],
    )

    config = load_train_app_config(config_path)

    assert config.env.action.name == "continuous_steer_drive_lean"
    assert config.env.action.steer_response_power == 0.7
    assert config.env.action.continuous_drive_mode == "always_accelerate"
    assert config.env.action.continuous_drive_deadzone == 0.15
    assert config.env.action.continuous_lean_deadzone == 0.25
    assert config.train.algorithm == "sac"
    assert config.train.buffer_size == 30_000
    assert config.train.learning_starts == 5_000
    assert config.train.ent_coef == "auto"
