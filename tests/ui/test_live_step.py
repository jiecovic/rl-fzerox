# tests/ui/test_live_step.py
"""Live Watch step-boundary tests.

These cases keep the extracted `runtime.live.step` helper honest without
starting a worker process. The fake env models only the state that the helper
must preserve across policy/manual stepping and snapshot publication.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fzerox_emulator import RaceControlState
from fzerox_emulator.arrays import ActionMask, DiscreteAction, RgbFrame
from rl_fzerox.core.envs.actions import ActionValue
from rl_fzerox.core.envs.engine.controls import ActionMaskSnapshot
from rl_fzerox.core.envs.engine.stepping import WatchEnvStep
from rl_fzerox.core.envs.observations import ObservationValue
from rl_fzerox.core.runtime_spec.schema import EmulatorConfig, WatchAppConfig
from rl_fzerox.core.training.inference import LoadedPolicy, PolicyRunner
from rl_fzerox.ui.watch.live_series import EpisodeLiveSeriesTracker
from rl_fzerox.ui.watch.runtime.ipc import WatchSnapshot
from rl_fzerox.ui.watch.runtime.live.notices import _TimedWatchNotice
from rl_fzerox.ui.watch.runtime.live.step import LiveStepRequest, step_policy_or_manual
from rl_fzerox.ui.watch.runtime.policy.cnn import (
    DEFAULT_CNN_ACTIVATION_NORMALIZATION,
    CnnActivationSampler,
)
from rl_fzerox.ui.watch.runtime.timing import RateMeter
from tests.ui.viewer_support import record_book


def test_live_step_policy_path_preserves_previous_and_final_snapshot_state(
    tmp_path: Path,
) -> None:
    config = _watch_config(tmp_path)
    env = _LiveStepEnv()
    queue = _SnapshotQueue()
    fake_policy = _FakeMaskablePolicy(action=[1])
    policy_runner = PolicyRunner(
        LoadedPolicy(
            run_dir=tmp_path,
            policy_path=tmp_path / "policy.zip",
            artifact="latest",
        ),
        fake_policy,
    )
    committed_action = np.array([0], dtype=np.int64)

    result = step_policy_or_manual(
        _request(
            config=config,
            env=env,
            queue=queue,
            policy_runner=policy_runner,
            observation=_rgb(10),
            info={"phase": "previous", "episode_step": 0},
            episode_reward=5.0,
            manual_control_enabled=False,
            current_gas_level=0.25,
            committed_policy_action=committed_action,
        )
    )

    assert fake_policy.action_masks_calls
    assert fake_policy.action_masks_calls[0].tolist() == [True, True]
    assert [action.tolist() for action in env.policy_actions] == [[1]]
    assert result.episode_reward == 8.5
    assert result.policy_action is not None
    assert np.asarray(result.policy_action).tolist() == [1]
    assert result.gas_level == 0.8
    snapshots = _snapshots(queue)
    assert len(snapshots) == 2
    assert [snapshot.info["phase"] for snapshot in snapshots] == ["previous", "policy"]
    assert [snapshot.episode_reward for snapshot in snapshots] == [5.0, 8.5]
    assert snapshots[0].gas_level == 0.25
    assert snapshots[1].gas_level == 0.8
    assert snapshots[0].policy_decision_frame is False
    assert snapshots[1].policy_decision_frame is True


def test_live_step_manual_single_frame_uses_render_fallback_and_clears_policy_action(
    tmp_path: Path,
) -> None:
    config = _watch_config(tmp_path)
    env = _LiveStepEnv()
    queue = _SnapshotQueue()
    committed_action = np.array([1], dtype=np.int64)

    result = step_policy_or_manual(
        _request(
            config=config,
            env=env,
            queue=queue,
            policy_runner=None,
            observation=_rgb(11),
            info={"phase": "previous", "episode_step": 0},
            episode_reward=2.0,
            manual_control_enabled=True,
            single_frame_manual=True,
            current_control_state=RaceControlState.from_mask(7, stick_x=0.25),
            current_gas_level=0.4,
            committed_policy_action=committed_action,
        )
    )

    assert env.step_frame_calls == [(7, "none")]
    assert result.policy_action is None
    assert result.episode_reward == 3.25
    snapshots = _snapshots(queue)
    assert len(snapshots) == 1
    assert _pixel(snapshots[0].raw_frame) == 99
    assert snapshots[0].info["phase"] == "manual_frame"
    assert snapshots[0].manual_control_enabled is True
    assert snapshots[0].policy_action is None
    assert snapshots[0].control_state.control_mask == 7


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


class _LiveStepEnv:
    def __init__(self) -> None:
        self._backend = _Backend()
        self.last_requested_control_state = RaceControlState()
        self.last_gas_level = 0.0
        self.policy_actions: list[DiscreteAction] = []
        self.step_frame_calls: list[tuple[int, str]] = []
        self._branches = {"throttle": (True, True)}
        self._flat_mask: ActionMask = np.array([True, True], dtype=np.bool_)

    @property
    def backend(self) -> _Backend:
        return self._backend

    def render(self) -> RgbFrame:
        return _rgb(99)

    def action_mask_branches(self) -> dict[str, tuple[bool, ...]]:
        return self._branches

    def action_mask_snapshot(self) -> ActionMaskSnapshot:
        return ActionMaskSnapshot(flat=self._flat_mask, branches=self._branches)

    def step_watch(self, action: ActionValue) -> WatchEnvStep:
        self.policy_actions.append(np.asarray(action, dtype=np.int64).reshape(-1))
        self.last_requested_control_state = RaceControlState.from_mask(3, stick_x=0.5)
        self.last_gas_level = 0.8
        return WatchEnvStep(
            observation=_rgb(20),
            reward=3.5,
            terminated=False,
            truncated=False,
            info={"phase": "policy", "episode_step": 4, "step_reward": 3.5},
            display_frames=(_rgb(21), _rgb(22)),
            display_controller_masks=(3, 3),
        )

    def step_control_watch(
        self,
        control_state: RaceControlState,
        *,
        spin_request: str = "none",
    ) -> WatchEnvStep:
        self.last_requested_control_state = control_state
        self.last_gas_level = 0.6
        return WatchEnvStep(
            observation=_rgb(30),
            reward=2.0,
            terminated=False,
            truncated=False,
            info={
                "phase": f"manual_control:{spin_request}",
                "episode_step": 4,
                "step_reward": 2.0,
            },
            display_frames=(_rgb(31), _rgb(32)),
            display_controller_masks=(control_state.control_mask,) * 2,
        )

    def step_frame(
        self,
        control_state: RaceControlState | None = None,
        *,
        spin_request: str = "none",
    ) -> tuple[ObservationValue, float, bool, bool, dict[str, object]]:
        state = RaceControlState() if control_state is None else control_state
        self.step_frame_calls.append((state.control_mask, spin_request))
        self.last_requested_control_state = state
        self.last_gas_level = 0.5
        return (
            _rgb(40),
            1.25,
            False,
            False,
            {"phase": "manual_frame", "episode_step": 1, "step_reward": 1.25},
        )


class _FakeMaskablePolicy:
    def __init__(self, *, action: list[int]) -> None:
        self._action = np.array(action, dtype=np.int64)
        self.action_masks_calls: list[ActionMask] = []

    def predict(
        self,
        observation: ObservationValue,
        state: object = None,
        episode_start: object = None,
        deterministic: bool = True,
        action_masks: ActionMask | None = None,
    ) -> tuple[DiscreteAction, None]:
        del observation, state, episode_start, deterministic
        if action_masks is not None:
            self.action_masks_calls.append(np.array(action_masks, copy=True))
        return self._action.copy(), None


def _request(
    *,
    config: WatchAppConfig,
    env: _LiveStepEnv,
    queue: _SnapshotQueue,
    policy_runner: PolicyRunner | None,
    observation: ObservationValue,
    info: dict[str, object],
    episode_reward: float,
    manual_control_enabled: bool,
    single_frame_manual: bool = False,
    current_control_state: RaceControlState | None = None,
    current_gas_level: float,
    committed_policy_action: DiscreteAction | None,
) -> LiveStepRequest:
    return LiveStepRequest(
        config=config,
        env=env,
        emulator=_TelemetryReader(),
        snapshot_queue=queue,
        policy_runner=policy_runner,
        observation=observation,
        info=info,
        reset_info=dict(info),
        episode=3,
        episode_reward=episode_reward,
        control_rate=RateMeter(window=4),
        target_control_fps=None,
        target_control_seconds=None,
        deterministic_policy=True,
        manual_control_enabled=manual_control_enabled,
        single_frame_manual=single_frame_manual,
        current_control_state=(
            RaceControlState() if current_control_state is None else current_control_state
        ),
        current_gas_level=current_gas_level,
        spin_request="none",
        boost_lamp_level=0.0,
        committed_policy_action=committed_policy_action,
        committed_action_mask_branches=env.action_mask_branches(),
        current_auxiliary_predictions=None,
        current_auxiliary_targets=None,
        cnn_visualization_enabled=False,
        cnn_normalization=DEFAULT_CNN_ACTIVATION_NORMALIZATION,
        cnn_sampler=CnnActivationSampler(refresh_interval_steps=1),
        auxiliary_visualization_enabled=False,
        auxiliary_target_names=(),
        watch_zeroed_state_features=frozenset(),
        live_visualization_enabled=False,
        live_series=EpisodeLiveSeriesTracker(),
        last_live_series_publish_time=0.0,
        track_record_book=record_book(),
        save_notice=_TimedWatchNotice(),
        policy_reload_error=None,
    )


class _TelemetryReader:
    def try_read_telemetry(self) -> None:
        return None


def _watch_config(tmp_path: Path) -> WatchAppConfig:
    core_path = tmp_path / "core.so"
    rom_path = tmp_path / "rom.n64"
    policy_path = tmp_path / "policy.zip"
    core_path.touch()
    rom_path.touch()
    policy_path.touch()
    return WatchAppConfig(
        emulator=EmulatorConfig(
            core_path=core_path,
            rom_path=rom_path,
        )
    )


def _snapshots(queue: _SnapshotQueue) -> list[WatchSnapshot]:
    snapshots: list[WatchSnapshot] = []
    for message in queue.messages:
        assert isinstance(message, WatchSnapshot)
        snapshots.append(message)
    return snapshots


def _rgb(value: int) -> RgbFrame:
    return np.full((2, 2, 3), value, dtype=np.uint8)


def _pixel(frame: RgbFrame) -> int:
    return int(frame[0, 0, 0])
