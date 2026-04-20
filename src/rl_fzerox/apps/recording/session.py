# src/rl_fzerox/apps/recording/session.py
from __future__ import annotations

from fzerox_emulator import Emulator
from rl_fzerox.apps.recording.models import RecordingSession
from rl_fzerox.apps.recording.video import resolve_video_fps
from rl_fzerox.core.config.schema import WatchAppConfig
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.seed import seed_process
from rl_fzerox.core.training.inference import load_policy_runner


def open_recording_session(
    config: WatchAppConfig,
    *,
    fps: float | None,
) -> RecordingSession:
    if config.watch.policy_run_dir is None:
        raise ValueError("--run-dir or watch.policy_run_dir is required for policy recording")
    seed_process(config.seed)
    emulator = Emulator(
        core_path=config.emulator.core_path,
        rom_path=config.emulator.rom_path,
        runtime_dir=config.emulator.runtime_dir,
        baseline_state_path=config.emulator.baseline_state_path,
        renderer=config.emulator.renderer,
    )
    env = FZeroXEnv(
        backend=emulator,
        config=config.env,
        reward_config=config.reward,
        curriculum_config=config.curriculum,
    )
    policy_runner = load_policy_runner(
        config.watch.policy_run_dir,
        artifact=config.watch.policy_artifact,
        device=config.watch.device,
    )
    env.sync_checkpoint_curriculum_stage(policy_runner.checkpoint_curriculum_stage_index)
    output_fps = resolve_video_fps(native_fps=env.backend.native_fps, override=fps)
    return RecordingSession(env=env, policy_runner=policy_runner, output_fps=output_fps)
