# tests/ui/test_watch_snapshots.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from fzerox_emulator import ControllerState
from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.config.schema import EmulatorConfig, WatchAppConfig
from rl_fzerox.ui.watch.runtime.ipc import WatchSnapshot
from rl_fzerox.ui.watch.runtime.snapshots import _publish_step_snapshots


class _SnapshotQueue:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def put_nowait(self, obj: object) -> None:
        self.messages.append(obj)

    def get_nowait(self) -> object:
        return self.messages.pop(0)


class _Backend:
    @property
    def native_fps(self) -> float:
        return 60.0


class _Env:
    @property
    def backend(self) -> _Backend:
        return _Backend()

    def render(self) -> RgbFrame:
        return _rgb(0)

    def action_mask_branches(self) -> dict[str, tuple[bool, ...]]:
        return {}


class _Emulator:
    def try_read_telemetry(self) -> None:
        return None


def test_publish_step_snapshots_marks_action_repeat_hold_frames(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
    )
    queue = _SnapshotQueue()
    policy_action = np.array([7, 1], dtype=np.int64)

    _publish_step_snapshots(
        config=config,
        env=_Env(),
        emulator=_Emulator(),
        snapshot_queue=queue,
        display_frames=(_rgb(1), _rgb(2), _rgb(3)),
        previous_observation=_rgb(10),
        previous_info={"phase": "previous"},
        previous_episode_reward=12.0,
        previous_telemetry=None,
        final_observation=_rgb(20),
        final_info={"phase": "final"},
        final_episode_reward=15.0,
        final_telemetry=None,
        reset_info={},
        episode=4,
        control_fps=20.0,
        target_control_fps=20.0,
        target_control_seconds=None,
        control_state=ControllerState(left_stick_x=0.5),
        gas_level=1.0,
        boost_lamp_level=0.0,
        action_mask_branches={"lean": (True, False, True)},
        policy_action=policy_action,
        policy_runner=None,
        deterministic_policy=True,
        manual_control_enabled=True,
        policy_reload_error=None,
        cnn_activations=None,
        best_finish_position=None,
        best_finish_times={"mute": 98_000},
        latest_finish_times={"mute": 101_000},
        latest_finish_deltas_ms={"mute": 3_000},
        failed_track_attempts=frozenset({"silence"}),
    )

    snapshots: list[WatchSnapshot] = []
    for message in queue.messages:
        assert isinstance(message, WatchSnapshot)
        snapshots.append(message)
    assert len(snapshots) == 3
    assert [snapshot.action_hold_frame for snapshot in snapshots] == [1, 2, 3]
    assert [snapshot.action_hold_frames for snapshot in snapshots] == [3, 3, 3]
    assert [snapshot.policy_decision_frame for snapshot in snapshots] == [
        False,
        False,
        True,
    ]
    assert [_pixel(snapshot.raw_frame) for snapshot in snapshots] == [1, 2, 3]
    assert [_pixel(snapshot.observation_image) for snapshot in snapshots] == [10, 10, 20]
    assert [snapshot.info["phase"] for snapshot in snapshots] == [
        "previous",
        "previous",
        "final",
    ]
    assert [snapshot.episode_reward for snapshot in snapshots] == [12.0, 12.0, 15.0]
    for snapshot in snapshots:
        assert isinstance(snapshot.policy_action, np.ndarray)
        assert np.array_equal(snapshot.policy_action, policy_action)
        assert snapshot.manual_control_enabled is True
        assert snapshot.action_mask_branches == {"lean": (True, False, True)}
        assert snapshot.best_finish_times == {"mute": 98_000}
        assert snapshot.latest_finish_times == {"mute": 101_000}
        assert snapshot.latest_finish_deltas_ms == {"mute": 3_000}
        assert snapshot.failed_track_attempts == frozenset({"silence"})


def _rgb(value: int) -> RgbFrame:
    return np.full((2, 2, 3), value, dtype=np.uint8)


def _pixel(frame: RgbFrame) -> int:
    return int(frame[0, 0, 0])
