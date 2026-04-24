# tests/core/training/test_training_inference.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from fzerox_emulator.arrays import (
    ActionMask,
    BoolArray,
    ContinuousAction,
    DiscreteAction,
    NumpyArray,
    PolicyState,
)
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.training.inference import LoadedPolicy, PolicyRunner
from rl_fzerox.core.training.inference.loader import (
    _artifact_kind_from_policy_path,
    _load_saved_policy,
    _load_saved_policy_algorithm,
)


class _FakePolicy:
    def __init__(self, action: list[int]) -> None:
        self._action = np.array(action, dtype=np.int64)
        self.deterministic_calls: list[bool] = []
        self.state_calls: list[PolicyState] = []
        self.episode_start_calls: list[BoolArray | None] = []

    def predict(
        self,
        observation: ObservationValue,
        state: PolicyState = None,
        episode_start: BoolArray | None = None,
        deterministic: bool = True,
    ) -> tuple[DiscreteAction, PolicyState]:
        _ = observation
        self.deterministic_calls.append(deterministic)
        self.state_calls.append(state)
        self.episode_start_calls.append(
            None if episode_start is None else np.array(episode_start, copy=True)
        )
        return self._action.copy(), None


class _FakeMaskablePolicy(_FakePolicy):
    def __init__(self, action: list[int]) -> None:
        super().__init__(action)
        self.action_masks_calls: list[ActionMask | None] = []

    def predict(
        self,
        observation: ObservationValue,
        state: PolicyState = None,
        episode_start: BoolArray | None = None,
        deterministic: bool = True,
        action_masks: ActionMask | None = None,
    ) -> tuple[DiscreteAction, PolicyState]:
        _ = observation
        self.deterministic_calls.append(deterministic)
        self.state_calls.append(state)
        self.episode_start_calls.append(
            None if episode_start is None else np.array(episode_start, copy=True)
        )
        self.action_masks_calls.append(None if action_masks is None else np.array(action_masks))
        return self._action.copy(), None


class _FakeContinuousPolicy:
    def predict(
        self,
        observation: ObservationValue,
        state: PolicyState = None,
        episode_start: BoolArray | None = None,
        deterministic: bool = True,
    ) -> tuple[ContinuousAction, PolicyState]:
        _ = (observation, state, episode_start, deterministic)
        return np.array([0.25, -0.75], dtype=np.float32), None


class _FakeHybridPolicy:
    def predict(
        self,
        observation: ObservationValue,
        state: PolicyState = None,
        episode_start: BoolArray | None = None,
        deterministic: bool = True,
    ) -> tuple[dict[str, NumpyArray], PolicyState]:
        _ = (observation, state, episode_start, deterministic)
        return {
            "continuous": np.array([0.25, 0.5], dtype=np.float32),
            "discrete": np.array([1], dtype=np.int64),
        }, None


class _FakeRecurrentMaskablePolicy(_FakeMaskablePolicy):
    def __init__(self, action: list[int]) -> None:
        super().__init__(action)
        self._next_state_id = 1

    def predict(
        self,
        observation: ObservationValue,
        state: PolicyState = None,
        episode_start: BoolArray | None = None,
        deterministic: bool = True,
        action_masks: ActionMask | None = None,
    ) -> tuple[DiscreteAction, PolicyState]:
        action, _ = super().predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
            action_masks=action_masks,
        )
        next_state = (
            np.full((1, 1, 1), self._next_state_id, dtype=np.float32),
            np.full((1, 1, 1), self._next_state_id + 1, dtype=np.float32),
        )
        self._next_state_id += 2
        return action, next_state


def _array_action(action: object) -> NumpyArray:
    assert isinstance(action, np.ndarray)
    return action


def test_policy_runner_reloads_updated_policy_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"v1")

    loaded_policy = LoadedPolicy(
        run_dir=tmp_path,
        policy_path=policy_path,
        artifact="latest",
    )
    runner = PolicyRunner(loaded_policy, _FakePolicy([2, 0]))

    observation = np.zeros((84, 116, 12), dtype=np.uint8)
    assert _array_action(runner.predict(observation)).tolist() == [2, 0]

    policy_path.write_bytes(b"v2")
    stat = policy_path.stat()
    os.utime(policy_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    monkeypatch.setattr(
        "rl_fzerox.core.training.inference.resolve_policy_artifact_path",
        lambda run_dir, *, artifact: policy_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.core.training.inference._load_saved_policy",
        lambda path, *, run_dir=None, device="cpu": _FakePolicy([4, 1]),
    )

    assert _array_action(runner.predict(observation)).tolist() == [4, 1]
    assert runner.reload_age_seconds < 1.0


def test_policy_runner_reports_reload_age_since_initial_load(tmp_path: Path) -> None:
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"v1")

    loaded_policy = LoadedPolicy(
        run_dir=tmp_path,
        policy_path=policy_path,
        artifact="latest",
    )
    runner = PolicyRunner(loaded_policy, _FakePolicy([2, 0]))

    time.sleep(0.01)

    assert runner.reload_age_seconds > 0.0


def test_policy_runner_can_sample_non_deterministic_actions(tmp_path: Path) -> None:
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"v1")
    fake_policy = _FakePolicy([2, 0])

    runner = PolicyRunner(
        LoadedPolicy(
            run_dir=tmp_path,
            policy_path=policy_path,
            artifact="latest",
        ),
        fake_policy,
    )

    observation = np.zeros((84, 116, 12), dtype=np.uint8)
    assert _array_action(runner.predict(observation, deterministic=False)).tolist() == [2, 0]

    assert fake_policy.deterministic_calls == [False]


def test_policy_runner_preserves_continuous_action_values(tmp_path: Path) -> None:
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"v1")

    runner = PolicyRunner(
        LoadedPolicy(
            run_dir=tmp_path,
            policy_path=policy_path,
            artifact="latest",
        ),
        _FakeContinuousPolicy(),
    )

    observation = np.zeros((84, 116, 12), dtype=np.uint8)
    action = _array_action(runner.predict(observation))

    assert action.dtype == np.float32
    assert action.tolist() == [0.25, -0.75]


def test_policy_runner_preserves_hybrid_action_dict(tmp_path: Path) -> None:
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"v1")

    runner = PolicyRunner(
        LoadedPolicy(
            run_dir=tmp_path,
            policy_path=policy_path,
            artifact="latest",
        ),
        _FakeHybridPolicy(),
    )

    observation = np.zeros((84, 116, 12), dtype=np.uint8)
    action = runner.predict(observation)

    assert runner.supports_action_masks is False
    assert isinstance(action, dict)
    assert _array_action(action["continuous"]).tolist() == [0.25, 0.5]
    assert _array_action(action["discrete"]).tolist() == [1]


def test_policy_runner_passes_action_masks_to_maskable_policies(tmp_path: Path) -> None:
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"v1")
    fake_policy = _FakeMaskablePolicy([2, 0])

    runner = PolicyRunner(
        LoadedPolicy(
            run_dir=tmp_path,
            policy_path=policy_path,
            artifact="latest",
        ),
        fake_policy,
    )

    observation = np.zeros((84, 116, 12), dtype=np.uint8)
    action_masks = np.array([True, False, True], dtype=bool)

    assert runner.supports_action_masks is True
    assert _array_action(runner.predict(observation, action_masks=action_masks)).tolist() == [2, 0]
    assert len(fake_policy.action_masks_calls) == 1
    recorded_masks = fake_policy.action_masks_calls[0]
    assert recorded_masks is not None
    assert np.array_equal(recorded_masks, action_masks)


def test_policy_runner_tracks_recurrent_state_across_predictions(tmp_path: Path) -> None:
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"v1")
    fake_policy = _FakeRecurrentMaskablePolicy([2, 0])

    runner = PolicyRunner(
        LoadedPolicy(
            run_dir=tmp_path,
            policy_path=policy_path,
            artifact="latest",
        ),
        fake_policy,
    )

    observation = np.zeros((84, 116, 12), dtype=np.uint8)

    assert _array_action(runner.predict(observation)).tolist() == [2, 0]
    assert fake_policy.state_calls[0] is None
    first_episode_start = fake_policy.episode_start_calls[0]
    assert first_episode_start is not None
    assert np.array_equal(first_episode_start, np.array([True]))

    assert _array_action(runner.predict(observation)).tolist() == [2, 0]
    second_state = fake_policy.state_calls[1]
    assert second_state is not None
    assert np.array_equal(second_state[0], np.full((1, 1, 1), 1, dtype=np.float32))
    second_episode_start = fake_policy.episode_start_calls[1]
    assert second_episode_start is not None
    assert np.array_equal(second_episode_start, np.array([False]))

    runner.reset()

    assert _array_action(runner.predict(observation)).tolist() == [2, 0]
    assert fake_policy.state_calls[2] is None
    third_episode_start = fake_policy.episode_start_calls[2]
    assert third_episode_start is not None
    assert np.array_equal(third_episode_start, np.array([True]))


def test_policy_runner_exposes_reload_error_until_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"v1")

    runner = PolicyRunner(
        LoadedPolicy(
            run_dir=tmp_path,
            policy_path=policy_path,
            artifact="latest",
        ),
        _FakePolicy([2, 0]),
    )

    policy_path.write_bytes(b"v2")
    stat = policy_path.stat()
    os.utime(policy_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    monkeypatch.setattr(
        "rl_fzerox.core.training.inference.resolve_policy_artifact_path",
        lambda run_dir, *, artifact: policy_path,
    )
    monkeypatch.setattr(
        "rl_fzerox.core.training.inference._load_saved_policy",
        lambda path, *, run_dir=None, device="cpu": (_ for _ in ()).throw(
            RuntimeError("bad checkpoint")
        ),
    )

    observation = np.zeros((84, 116, 12), dtype=np.uint8)
    assert _array_action(runner.predict(observation)).tolist() == [2, 0]
    assert runner.reload_error == "bad checkpoint"
    assert runner.last_reload_error == "bad checkpoint"

    monkeypatch.setattr(
        "rl_fzerox.core.training.inference._load_saved_policy",
        lambda path, *, run_dir=None, device="cpu": _FakePolicy([4, 1]),
    )

    assert _array_action(runner.predict(observation)).tolist() == [4, 1]
    assert runner.reload_error is None
    assert runner.last_reload_error == "bad checkpoint"


def test_policy_runner_refreshes_metadata_without_policy_zip_change(tmp_path: Path) -> None:
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"v1")
    runner = PolicyRunner(
        LoadedPolicy(
            run_dir=tmp_path,
            policy_path=policy_path,
            artifact="latest",
        ),
        _FakePolicy([2, 0]),
    )

    assert runner.checkpoint_curriculum_stage_index is None
    assert runner.checkpoint_curriculum_stage is None
    assert runner.checkpoint_num_timesteps is None

    metadata_path = tmp_path / "latest_policy.metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "curriculum_stage_index": 1,
                "curriculum_stage_name": "lean_enabled",
                "num_timesteps": 660_000,
            }
        ),
        encoding="utf-8",
    )

    runner.refresh()

    assert runner.checkpoint_curriculum_stage_index == 1
    assert runner.checkpoint_curriculum_stage == "lean_enabled"
    assert runner.checkpoint_num_timesteps == 660_000


def test_load_saved_policy_algorithm_rejects_invalid_train_config(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config_path = tmp_path / "train_config.yaml"
    OmegaConf.save(
        config=OmegaConf.create(
            {
                "seed": 7,
                "emulator": {
                    "core_path": str(core_path),
                    "rom_path": str(rom_path),
                },
                "env": {"action": {"name": "steer_drive_boost_lean"}},
                "policy": {
                    "recurrent": {
                        "enabled": True,
                    }
                },
                "train": {
                    "algorithm": "maskable_ppo",
                    "total_timesteps": 1000,
                },
            }
        ),
        f=str(config_path),
    )

    with pytest.raises(ValueError, match="requires a recurrent train.algorithm"):
        _load_saved_policy_algorithm(tmp_path)


def test_load_saved_policy_algorithm_recognizes_maskable_recurrent_ppo(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config_path = tmp_path / "train_config.yaml"
    OmegaConf.save(
        config=OmegaConf.create(
            {
                "seed": 7,
                "emulator": {
                    "core_path": str(core_path),
                    "rom_path": str(rom_path),
                },
                "env": {"action": {"name": "steer_drive_boost_lean"}},
                "policy": {
                    "recurrent": {
                        "enabled": True,
                    }
                },
                "train": {
                    "algorithm": "maskable_recurrent_ppo",
                    "total_timesteps": 1000,
                },
            }
        ),
        f=str(config_path),
    )

    assert _load_saved_policy_algorithm(tmp_path) == "maskable_recurrent_ppo"


def test_load_saved_policy_algorithm_recognizes_sac(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config_path = tmp_path / "train_config.yaml"
    OmegaConf.save(
        config=OmegaConf.create(
            {
                "seed": 7,
                "emulator": {
                    "core_path": str(core_path),
                    "rom_path": str(rom_path),
                },
                "env": {"action": {"name": "continuous_steer_drive"}},
                "policy": {},
                "train": {
                    "algorithm": "sac",
                    "ent_coef": "auto",
                    "total_timesteps": 1000,
                },
            }
        ),
        f=str(config_path),
    )

    assert _load_saved_policy_algorithm(tmp_path) == "sac"


def test_load_saved_policy_algorithm_recognizes_maskable_hybrid_action_sac(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config_path = tmp_path / "train_config.yaml"
    OmegaConf.save(
        config=OmegaConf.create(
            {
                "seed": 7,
                "emulator": {
                    "core_path": str(core_path),
                    "rom_path": str(rom_path),
                },
                "env": {"action": {"name": "hybrid_steer_drive_boost_lean"}},
                "policy": {},
                "train": {
                    "algorithm": "maskable_hybrid_action_sac",
                    "ent_coef": "auto",
                    "total_timesteps": 1000,
                },
            }
        ),
        f=str(config_path),
    )

    assert _load_saved_policy_algorithm(tmp_path) == "maskable_hybrid_action_sac"


def test_load_saved_policy_algorithm_recognizes_maskable_hybrid_action_ppo(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config_path = tmp_path / "train_config.yaml"
    OmegaConf.save(
        config=OmegaConf.create(
            {
                "seed": 7,
                "emulator": {
                    "core_path": str(core_path),
                    "rom_path": str(rom_path),
                },
                "env": {"action": {"name": "hybrid_steer_drive_boost_lean"}},
                "policy": {},
                "train": {
                    "algorithm": "maskable_hybrid_action_ppo",
                    "total_timesteps": 1000,
                },
            }
        ),
        f=str(config_path),
    )

    assert _load_saved_policy_algorithm(tmp_path) == "maskable_hybrid_action_ppo"


def test_load_saved_policy_algorithm_recognizes_maskable_hybrid_recurrent_ppo(
    tmp_path: Path,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config_path = tmp_path / "train_config.yaml"
    OmegaConf.save(
        config=OmegaConf.create(
            {
                "seed": 7,
                "emulator": {
                    "core_path": str(core_path),
                    "rom_path": str(rom_path),
                },
                "env": {"action": {"name": "hybrid_steer_drive_boost_lean"}},
                "policy": {
                    "recurrent": {
                        "enabled": True,
                    }
                },
                "train": {
                    "algorithm": "maskable_hybrid_recurrent_ppo",
                    "total_timesteps": 1000,
                },
            }
        ),
        f=str(config_path),
    )

    assert _load_saved_policy_algorithm(tmp_path) == "maskable_hybrid_recurrent_ppo"


def test_load_saved_policy_uses_full_model_artifact_for_recurrent_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config_path = tmp_path / "train_config.yaml"
    OmegaConf.save(
        config=OmegaConf.create(
            {
                "seed": 7,
                "emulator": {
                    "core_path": str(core_path),
                    "rom_path": str(rom_path),
                },
                "env": {"action": {"name": "steer_drive_boost_lean"}},
                "policy": {
                    "recurrent": {
                        "enabled": True,
                    }
                },
                "train": {
                    "algorithm": "maskable_recurrent_ppo",
                    "total_timesteps": 1000,
                },
            }
        ),
        f=str(config_path),
    )
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"policy")
    model_path = tmp_path / "latest_model.zip"
    model_path.write_bytes(b"model")

    captured: dict[str, object] = {}

    class _FakeLoadedRecurrentModel:
        def predict(
            self,
            observation: ObservationValue,
            state: PolicyState = None,
            episode_start: BoolArray | None = None,
            deterministic: bool = True,
            action_masks: ActionMask | None = None,
        ) -> tuple[DiscreteAction, PolicyState]:
            _ = (observation, state, episode_start, deterministic, action_masks)
            return np.array([0], dtype=np.int64), None

    def _fake_load(path: str, *, device: str) -> _FakeLoadedRecurrentModel:
        captured["path"] = path
        captured["device"] = device
        return _FakeLoadedRecurrentModel()

    monkeypatch.setattr("sb3x.MaskableRecurrentPPO.load", _fake_load)

    loaded = _load_saved_policy(policy_path, run_dir=tmp_path, device="cpu")

    assert isinstance(loaded, _FakeLoadedRecurrentModel)
    assert captured == {
        "path": str(model_path.resolve()),
        "device": "cpu",
    }


def test_load_saved_policy_uses_full_model_artifact_for_maskable_hybrid_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config_path = tmp_path / "train_config.yaml"
    OmegaConf.save(
        config=OmegaConf.create(
            {
                "seed": 7,
                "emulator": {
                    "core_path": str(core_path),
                    "rom_path": str(rom_path),
                },
                "env": {"action": {"name": "hybrid_steer_drive_boost_lean"}},
                "policy": {},
                "train": {
                    "algorithm": "maskable_hybrid_action_ppo",
                    "total_timesteps": 1000,
                },
            }
        ),
        f=str(config_path),
    )
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"policy")
    model_path = tmp_path / "latest_model.zip"
    model_path.write_bytes(b"model")

    captured: dict[str, object] = {}

    class _FakeLoadedMaskableHybridModel:
        def predict(
            self,
            observation: ObservationValue,
            state: PolicyState = None,
            episode_start: BoolArray | None = None,
            deterministic: bool = True,
            action_masks: ActionMask | None = None,
        ) -> tuple[dict[str, NumpyArray], PolicyState]:
            _ = (observation, state, episode_start, deterministic, action_masks)
            return {
                "continuous": np.array([0.0, 0.0], dtype=np.float32),
                "discrete": np.array([0, 0], dtype=np.int64),
            }, None

    def _fake_load(path: str, *, device: str) -> _FakeLoadedMaskableHybridModel:
        captured["path"] = path
        captured["device"] = device
        return _FakeLoadedMaskableHybridModel()

    monkeypatch.setattr("sb3x.MaskableHybridActionPPO.load", _fake_load)

    loaded = _load_saved_policy(policy_path, run_dir=tmp_path, device="cpu")

    assert isinstance(loaded, _FakeLoadedMaskableHybridModel)
    assert captured == {
        "path": str(model_path.resolve()),
        "device": "cpu",
    }


def test_load_saved_policy_uses_full_model_artifact_for_maskable_hybrid_sac_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config_path = tmp_path / "train_config.yaml"
    OmegaConf.save(
        config=OmegaConf.create(
            {
                "seed": 7,
                "emulator": {
                    "core_path": str(core_path),
                    "rom_path": str(rom_path),
                },
                "env": {"action": {"name": "hybrid_steer_drive_boost_lean"}},
                "policy": {},
                "train": {
                    "algorithm": "maskable_hybrid_action_sac",
                    "ent_coef": "auto",
                    "total_timesteps": 1000,
                },
            }
        ),
        f=str(config_path),
    )
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"policy")
    model_path = tmp_path / "latest_model.zip"
    model_path.write_bytes(b"model")

    captured: dict[str, object] = {}

    class _FakeLoadedMaskableHybridSacModel:
        def predict(
            self,
            observation: ObservationValue,
            state: PolicyState = None,
            episode_start: BoolArray | None = None,
            deterministic: bool = True,
            action_masks: ActionMask | None = None,
        ) -> tuple[dict[str, NumpyArray], PolicyState]:
            _ = (observation, state, episode_start, deterministic, action_masks)
            return {
                "continuous": np.array([0.0, 0.0], dtype=np.float32),
                "discrete": np.array([0, 0], dtype=np.int64),
            }, None

    def _fake_load(path: str, *, device: str) -> _FakeLoadedMaskableHybridSacModel:
        captured["path"] = path
        captured["device"] = device
        return _FakeLoadedMaskableHybridSacModel()

    monkeypatch.setattr("sb3x.MaskableHybridActionSAC.load", _fake_load)

    loaded = _load_saved_policy(policy_path, run_dir=tmp_path, device="cpu")

    assert isinstance(loaded, _FakeLoadedMaskableHybridSacModel)
    assert captured == {
        "path": str(model_path.resolve()),
        "device": "cpu",
    }


def test_load_saved_policy_uses_full_model_artifact_for_maskable_hybrid_recurrent_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config_path = tmp_path / "train_config.yaml"
    OmegaConf.save(
        config=OmegaConf.create(
            {
                "seed": 7,
                "emulator": {
                    "core_path": str(core_path),
                    "rom_path": str(rom_path),
                },
                "env": {"action": {"name": "hybrid_steer_drive_boost_lean"}},
                "policy": {
                    "recurrent": {
                        "enabled": True,
                    }
                },
                "train": {
                    "algorithm": "maskable_hybrid_recurrent_ppo",
                    "total_timesteps": 1000,
                },
            }
        ),
        f=str(config_path),
    )
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(b"policy")
    model_path = tmp_path / "latest_model.zip"
    model_path.write_bytes(b"model")

    captured: dict[str, object] = {}

    class _FakeLoadedMaskableHybridRecurrentModel:
        def predict(
            self,
            observation: ObservationValue,
            state: PolicyState = None,
            episode_start: BoolArray | None = None,
            deterministic: bool = True,
            action_masks: ActionMask | None = None,
        ) -> tuple[dict[str, NumpyArray], PolicyState]:
            _ = (observation, state, episode_start, deterministic, action_masks)
            return {
                "continuous": np.array([0.0, 0.0], dtype=np.float32),
                "discrete": np.array([0, 0], dtype=np.int64),
            }, None

    def _fake_load(path: str, *, device: str) -> _FakeLoadedMaskableHybridRecurrentModel:
        captured["path"] = path
        captured["device"] = device
        return _FakeLoadedMaskableHybridRecurrentModel()

    monkeypatch.setattr("sb3x.MaskableHybridRecurrentPPO.load", _fake_load)

    loaded = _load_saved_policy(policy_path, run_dir=tmp_path, device="cpu")

    assert isinstance(loaded, _FakeLoadedMaskableHybridRecurrentModel)
    assert captured == {
        "path": str(model_path.resolve()),
        "device": "cpu",
    }


def test_artifact_kind_from_policy_path_uses_standard_prefixes() -> None:
    assert _artifact_kind_from_policy_path(Path("latest_policy.zip")) == "latest"
    assert _artifact_kind_from_policy_path(Path("best_policy.zip")) == "best"
    assert _artifact_kind_from_policy_path(Path("final_policy.zip")) == "final"
