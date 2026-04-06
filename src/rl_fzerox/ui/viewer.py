# src/rl_fzerox/ui/viewer.py
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rl_fzerox.core.config.models import WatchAppConfig
from rl_fzerox.core.emulator import Emulator
from rl_fzerox.core.emulator.video import display_size
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.seed import episode_seed, seed_process


@dataclass(frozen=True)
class ViewerInput:
    """Normalized viewer input state for one polling cycle."""

    quit_requested: bool = False
    toggle_pause: bool = False
    step_once: bool = False


def run_viewer(config: WatchAppConfig) -> None:
    """Run the real-time watch UI against the configured emulator."""

    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError(
            "pygame is required for watching emulator output. "
            "Install with `pip install -e .[watch]`."
        ) from exc

    seed_process(config.seed)

    emulator = Emulator(
        core_path=config.emulator.core_path,
        rom_path=config.emulator.rom_path,
    )
    env = FZeroXEnv(backend=emulator, config=config.env)
    pygame.init()

    try:
        screen = None
        font = None
        paused = False
        target_seconds: float | None = None

        for episode in range(config.watch.episodes):
            frame, info = env.reset(seed=episode_seed(config.seed, episode))

            if screen is None or font is None:
                screen = _create_screen(pygame, emulator.display_size)
                font = pygame.font.Font(None, 24)
                target_fps = config.watch.fps or (env.backend.native_fps / config.env.action_repeat)
                target_seconds = 1.0 / target_fps

            viewer_input = _poll_viewer_input(pygame)
            if viewer_input.quit_requested:
                return
            if viewer_input.toggle_pause:
                paused = not paused

            screen = _ensure_screen(pygame, screen, emulator.display_size)

            _draw_frame(
                pygame=pygame,
                screen=screen,
                font=font,
                frame=frame,
                episode=episode,
                info=info,
                episode_reward=0.0,
                paused=paused,
            )

            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):
                viewer_input = _poll_viewer_input(pygame)
                if viewer_input.quit_requested:
                    return
                if viewer_input.toggle_pause:
                    paused = not paused
                if paused and not viewer_input.step_once:
                    _draw_frame(
                        pygame=pygame,
                        screen=screen,
                        font=font,
                        frame=frame,
                        episode=episode,
                        info=info,
                        episode_reward=episode_reward,
                        paused=True,
                    )
                    time.sleep(0.01)
                    continue

                if paused and viewer_input.step_once:
                    frame_step = env.backend.step_frame()
                    frame = frame_step.frame
                    episode_reward += frame_step.reward
                    terminated = frame_step.terminated
                    truncated = frame_step.truncated
                    info = dict(frame_step.info)
                    _draw_frame(
                        pygame=pygame,
                        screen=screen,
                        font=font,
                        frame=frame,
                        episode=episode,
                        info=info,
                        episode_reward=episode_reward,
                        paused=True,
                    )
                    continue

                frame_start = time.perf_counter()
                frame, reward, terminated, truncated, info = env.step(0)
                episode_reward += reward

                screen = _ensure_screen(pygame, screen, emulator.display_size)
                _draw_frame(
                    pygame=pygame,
                    screen=screen,
                    font=font,
                    frame=frame,
                    episode=episode,
                    info=info,
                    episode_reward=episode_reward,
                    paused=paused,
                )

                if paused:
                    continue

                if target_seconds is None:
                    raise RuntimeError("Watch target_seconds was not initialized")
                elapsed = time.perf_counter() - frame_start
                delay = max(0.0, target_seconds - elapsed)
                if delay:
                    time.sleep(delay)
    finally:
        env.close()
        pygame.quit()


def _create_screen(pygame, display_size: tuple[int, int]):
    screen = pygame.display.set_mode(display_size)
    pygame.display.set_caption("F-Zero X Watch")
    return screen


def _ensure_screen(pygame, screen, display_size: tuple[int, int]):
    if screen.get_size() == display_size:
        return screen
    return _create_screen(pygame, display_size)


def _draw_frame(
    *,
    pygame,
    screen,
    font,
    frame: np.ndarray,
    episode: int,
    info: dict[str, object],
    episode_reward: float,
    paused: bool,
) -> None:
    frame_height, frame_width, _ = frame.shape
    surface = pygame.image.frombuffer(frame.tobytes(), (frame_width, frame_height), "RGB")
    target_size = display_size(frame.shape, _display_aspect_ratio(info))
    if surface.get_size() != target_size:
        surface = pygame.transform.scale(surface, target_size)
    screen.blit(surface, (0, 0))

    overlay = font.render(
        (
            f"{'paused' if paused else 'running'}  "
            "P pause/resume  N next frame  "
            f"episode={episode} frame={info.get('frame_index', 0):6d} "
            f"reward={episode_reward:8.2f}"
        ),
        True,
        (255, 255, 255),
    )
    overlay_background = pygame.Surface(
        (overlay.get_width() + 12, overlay.get_height() + 8),
        pygame.SRCALPHA,
    )
    overlay_background.fill((0, 0, 0, 180))
    screen.blit(overlay_background, (8, 8))
    screen.blit(overlay, (8, 8))
    pygame.display.flip()


def _display_aspect_ratio(info: dict[str, object]) -> float:
    value = info.get("display_aspect_ratio")
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _poll_viewer_input(pygame) -> ViewerInput:
    quit_requested = False
    toggle_pause = False
    step_once = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_requested = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                toggle_pause = True
            elif event.key == pygame.K_n:
                step_once = True

    return ViewerInput(
        quit_requested=quit_requested,
        toggle_pause=toggle_pause,
        step_once=step_once,
    )
