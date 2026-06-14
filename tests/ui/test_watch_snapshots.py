# tests/ui/test_watch_snapshots.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from fzerox_emulator import RaceControlState
from fzerox_emulator.arrays import RgbFrame
from rl_fzerox.core.runtime_spec.schema import EmulatorConfig, WatchAppConfig
from rl_fzerox.ui.watch.live_series import EpisodeLiveSeriesSnapshot
from rl_fzerox.ui.watch.runtime.ipc import WatchSnapshot
from rl_fzerox.ui.watch.runtime.snapshots import (
    _audio_chunks_for_frames,
    _build_snapshot,
    _publish_step_snapshots,
)
from tests.ui.viewer_support import record_book, record_entry


class _SnapshotQueue:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def put_nowait(self, obj: object) -> None:
        self.messages.append(obj)

    def get_nowait(self) -> object:
        return self.messages.pop(0)


def test_audio_chunks_for_frames_splits_interleaved_pcm() -> None:
    chunks = _audio_chunks_for_frames(
        np.array([1, -1, 2, -2, 3, -3], dtype=np.int16),
        np.array([1, 2], dtype=np.uint32),
        frame_count=2,
    )

    assert [np.asarray(chunk).tolist() for chunk in chunks] == [[1, -1], [2, -2, 3, -3]]


def test_audio_chunks_for_frames_returns_silence_for_mismatched_counts() -> None:
    chunks = _audio_chunks_for_frames(
        np.array([1, -1], dtype=np.int16),
        np.array([1], dtype=np.uint32),
        frame_count=2,
    )

    assert chunks == ((), ())


def race_control_state(
    *,
    control_mask: int = 0,
    stick_x: float = 0.0,
    pitch: float = 0.0,
) -> RaceControlState:
    return RaceControlState.from_mask(
        control_mask,
        stick_x=stick_x,
        pitch=pitch,
    )


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
    live_series = EpisodeLiveSeriesSnapshot(
        episode=4,
        env_steps=(1,),
        speed_kph=(640.0,),
        step_rewards=(12.5,),
        progress_speed_multiplier=(1.0,),
        position_progress_multiplier=(1.0,),
        progress_speed_position_multiplier=(1.0,),
        edge_ratio=(0.0,),
        outside_edge_excess_ratio=(0.0,),
        height_above_ground=(0.0,),
        ko_star_events=(),
        current_ko_star_count=0,
        current_return=12.5,
        current_progress=0.1,
        max_progress=0.1,
    )

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
        control_state=race_control_state(stick_x=0.5),
        gas_level=1.0,
        boost_lamp_level=0.0,
        action_mask_branches={"lean": (True, False, True)},
        policy_action=policy_action,
        policy_runner=None,
        deterministic_policy=True,
        manual_control_enabled=True,
        policy_reload_error=None,
        cnn_activations=None,
        active_track_sampling=None,
        track_record_book=record_book(
            {
                "mute": record_entry(
                    best_finish_rank=1,
                    best_finish_rank_time_ms=99_000,
                    best_finish_rank_setup={"vehicle_name": "Blue Falcon"},
                    best_finish_time_ms=98_000,
                    best_finish_time_rank=2,
                    best_finish_time_setup={"engine_setting_raw_value": 60},
                    latest_finish_rank=3,
                    latest_finish_time_ms=101_000,
                    latest_finish_delta_ms=3_000,
                    latest_finish_setup={
                        "vehicle_name": "Blue Falcon",
                        "engine_setting_raw_value": 40,
                    },
                    attempt_stats={
                        "attempts": 2,
                        "finishes": 1,
                        "completion_samples": 2,
                        "completion_sum": 1.5,
                        "best_completion": 1.0,
                    },
                ),
                "silence": record_entry(failed_attempt=True),
            }
        ),
        live_episode_series=live_series,
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
    assert [
        _pixel(snapshot.policy_observation.image)
        for snapshot in snapshots
        if snapshot.policy_observation is not None
    ] == [10, 10, 20]
    assert [snapshot.info["phase"] for snapshot in snapshots] == [
        "previous",
        "previous",
        "final",
    ]
    assert [snapshot.episode_reward for snapshot in snapshots] == [12.0, 12.0, 15.0]
    assert [
        snapshot.track_record_book.entries["mute"].best_finish_rank for snapshot in snapshots
    ] == [1, 1, 1]
    assert [snapshot.live_episode_series for snapshot in snapshots] == [
        None,
        None,
        live_series,
    ]
    for snapshot in snapshots:
        assert isinstance(snapshot.policy_action, np.ndarray)
        assert np.array_equal(snapshot.policy_action, policy_action)
        assert snapshot.manual_control_enabled is True
        assert snapshot.action_mask_branches == {"lean": (True, False, True)}
        entry = snapshot.track_record_book.entries["mute"]
        assert entry.best_finish_rank_time_ms == 99_000
        assert entry.best_finish_rank_setup == {"vehicle_name": "Blue Falcon"}
        assert entry.best_finish_time_ms == 98_000
        assert entry.best_finish_time_rank == 2
        assert entry.best_finish_time_setup == {"engine_setting_raw_value": 60}
        assert entry.latest_finish_rank == 3
        assert entry.latest_finish_time_ms == 101_000
        assert entry.latest_finish_delta_ms == 3_000
        assert entry.latest_finish_setup == {
            "vehicle_name": "Blue Falcon",
            "engine_setting_raw_value": 40,
        }
        assert entry.attempt_stats.as_mapping() == {
            "attempts": 2,
            "finishes": 1,
            "completion_samples": 2,
            "completion_sum": 1.5,
            "best_completion": 1.0,
        }
        assert snapshot.track_record_book.entries["silence"].failed_attempt is True


def test_publish_step_snapshots_uses_exact_display_controller_masks(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
    )
    queue = _SnapshotQueue()
    previous_action = np.array([0], dtype=np.int64)
    final_action = np.array([2], dtype=np.int64)

    _publish_step_snapshots(
        config=config,
        env=_Env(),
        emulator=_Emulator(),
        snapshot_queue=queue,
        display_frames=(_rgb(1), _rgb(2), _rgb(3)),
        display_controller_masks=np.array([11, 12, 14], dtype=np.uint16),
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
        previous_control_state=race_control_state(control_mask=1, stick_x=-0.5),
        final_control_state=race_control_state(control_mask=2, stick_x=0.5),
        previous_gas_level=0.0,
        final_gas_level=1.0,
        previous_action_mask_branches={"spin": (True, False, False)},
        final_action_mask_branches={"spin": (True, True, True)},
        previous_policy_action=previous_action,
        final_policy_action=final_action,
        boost_lamp_level=0.0,
        policy_runner=None,
        deterministic_policy=True,
        manual_control_enabled=True,
        policy_reload_error=None,
        cnn_activations=None,
        active_track_sampling=None,
        track_record_book=record_book(),
    )

    snapshots: list[WatchSnapshot] = []
    for message in queue.messages:
        assert isinstance(message, WatchSnapshot)
        snapshots.append(message)

    assert [snapshot.control_state.control_mask for snapshot in snapshots] == [11, 12, 14]
    assert [snapshot.control_state.stick_x for snapshot in snapshots] == [0.5, 0.5, 0.5]
    assert [snapshot.action_mask_branches for snapshot in snapshots] == [
        {"spin": (True, True, True)},
        {"spin": (True, True, True)},
        {"spin": (True, True, True)},
    ]
    for snapshot in snapshots:
        assert isinstance(snapshot.policy_action, np.ndarray)
        assert np.array_equal(snapshot.policy_action, final_action)


def test_menu_snapshot_has_no_policy_observation_shape(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
    )

    snapshot = _build_snapshot(
        config=config,
        env=_Env(),
        emulator=_Emulator(),
        observation=None,
        info={"game_mode": "select_mode"},
        reset_info={},
        episode=0,
        episode_reward=0.0,
        control_fps=60.0,
        target_control_fps=60.0,
        control_state=race_control_state(),
        gas_level=0.0,
        boost_lamp_level=0.0,
        action_mask_branches={},
        policy_action=None,
        policy_runner=None,
        deterministic_policy=False,
        manual_control_enabled=False,
        policy_reload_error=None,
        cnn_activations=None,
        active_track_sampling=None,
        track_record_book=record_book(),
    )

    assert snapshot.policy_observation is None
    assert snapshot.policy_observation_shape is None
    assert _pixel(snapshot.raw_frame) == 0


def test_career_menu_snapshot_has_no_policy_observation_shape(tmp_path: Path) -> None:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    core_path.touch()
    rom_path.touch()
    config = WatchAppConfig(
        emulator=EmulatorConfig(core_path=core_path, rom_path=rom_path),
    )
    config = config.model_copy(
        update={
            "watch": config.watch.model_copy(
                update={"managed_save_game_id": "save-a"},
            ),
        }
    )

    snapshot = _build_snapshot(
        config=config,
        env=_Env(),
        emulator=_Emulator(),
        observation=None,
        info={"game_mode": "select_mode"},
        reset_info={},
        episode=0,
        episode_reward=0.0,
        control_fps=60.0,
        target_control_fps=60.0,
        control_state=race_control_state(),
        gas_level=0.0,
        boost_lamp_level=0.0,
        action_mask_branches={},
        policy_action=None,
        policy_runner=None,
        deterministic_policy=False,
        manual_control_enabled=False,
        policy_reload_error=None,
        cnn_activations=None,
        active_track_sampling=None,
        track_record_book=record_book(),
    )

    assert snapshot.policy_observation is None
    assert snapshot.policy_observation_shape is None


def _rgb(value: int) -> RgbFrame:
    return np.full((2, 2, 3), value, dtype=np.uint8)


def _pixel(frame: RgbFrame) -> int:
    return int(frame[0, 0, 0])
