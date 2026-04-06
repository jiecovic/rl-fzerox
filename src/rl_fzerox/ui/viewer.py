# src/rl_fzerox/ui/viewer.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from rl_fzerox._native import (
    JOYPAD_A,
    JOYPAD_B,
    JOYPAD_DOWN,
    JOYPAD_LEFT,
    JOYPAD_RIGHT,
    JOYPAD_SELECT,
    JOYPAD_START,
    JOYPAD_UP,
    joypad_mask,
)
from rl_fzerox.core.config.models import WatchAppConfig
from rl_fzerox.core.emulator import Emulator
from rl_fzerox.core.emulator.video import display_size
from rl_fzerox.core.envs import FZeroXEnv
from rl_fzerox.core.game import FZeroXTelemetry, read_telemetry
from rl_fzerox.core.seed import seed_process

Color = tuple[int, int, int]


class TextSurface(Protocol):
    """Minimal surface contract used by the panel layout code."""

    def get_width(self) -> int: ...

    def get_height(self) -> int: ...


class RenderFont(Protocol):
    """Minimal font contract used by the panel renderer."""

    def render(
        self,
        text: str,
        antialias: bool,
        color: Color,
    ) -> TextSurface: ...


@dataclass(frozen=True)
class ViewerLayout:
    """Spacing and sizing used by the watch window layout."""

    panel_width: int = 456
    panel_padding: int = 12
    column_gap: int = 16
    title_gap: int = 2
    title_section_gap: int = 8
    section_gap: int = 8
    section_title_gap: int = 4
    section_rule_gap: int = 4
    line_gap: int = 2


@dataclass(frozen=True)
class ViewerPalette:
    """Color palette for the watch window."""

    app_background: Color = (11, 14, 18)
    panel_background: Color = (20, 24, 30)
    panel_border: Color = (46, 54, 66)
    text_primary: Color = (238, 241, 245)
    text_muted: Color = (155, 165, 178)
    text_accent: Color = (126, 214, 170)
    text_warning: Color = (241, 206, 108)


@dataclass(frozen=True)
class PanelLine:
    """One labeled line in the side panel."""

    label: str
    value: str
    color: Color


@dataclass(frozen=True)
class PanelSection:
    """One titled section in the side panel."""

    title: str
    lines: list[PanelLine]


@dataclass(frozen=True)
class PanelColumns:
    """Left and right panel columns rendered next to the game view."""

    left: list[PanelSection]
    right: list[PanelSection]


@dataclass(frozen=True)
class ViewerFonts:
    """Font bundle used by the watch panel renderer."""

    title: RenderFont
    section: RenderFont
    body: RenderFont
    small: RenderFont


@dataclass(frozen=True)
class ViewerFontSizes:
    """Point sizes for the watch panel fonts."""

    title: int = 24
    section: int = 20
    body: int = 18
    small: int = 16


LAYOUT = ViewerLayout()
PALETTE = ViewerPalette()
FONT_SIZES = ViewerFontSizes()

BUTTON_LABELS: tuple[tuple[int, str], ...] = (
    (JOYPAD_UP, "Up"),
    (JOYPAD_DOWN, "Down"),
    (JOYPAD_LEFT, "Left"),
    (JOYPAD_RIGHT, "Right"),
    (JOYPAD_A, "A"),
    (JOYPAD_B, "B"),
    (JOYPAD_START, "Start"),
    (JOYPAD_SELECT, "Select"),
)


@dataclass(frozen=True)
class ViewerInput:
    """Normalized viewer input state for one polling cycle."""

    quit_requested: bool = False
    toggle_pause: bool = False
    step_once: bool = False
    save_state: bool = False
    joypad_mask: int = 0


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
        runtime_dir=config.emulator.runtime_dir,
        baseline_state_path=config.emulator.baseline_state_path,
    )
    env = FZeroXEnv(backend=emulator, config=config.env)
    pygame.init()

    try:
        screen = None
        fonts = None
        paused = False
        target_seconds: float | None = None
        episode = 0
        while config.watch.episodes is None or episode < config.watch.episodes:
            reset_seed = config.seed if episode == 0 else None
            frame, info = env.reset(seed=reset_seed)
            reset_info = dict(info)
            current_joypad_mask = 0
            telemetry = _read_live_telemetry(emulator)

            if screen is None or fonts is None:
                screen = _create_screen(pygame, emulator.display_size)
                fonts = _create_fonts(pygame)
                target_fps = config.watch.fps or (env.backend.native_fps / config.env.action_repeat)
                target_seconds = 1.0 / target_fps

            viewer_input = _poll_viewer_input(pygame)
            if viewer_input.quit_requested:
                return
            if viewer_input.toggle_pause:
                paused = not paused
            current_joypad_mask = viewer_input.joypad_mask
            emulator.set_joypad_mask(viewer_input.joypad_mask)
            if viewer_input.save_state:
                _save_baseline_state(
                    emulator=emulator,
                    baseline_state_path=config.emulator.baseline_state_path,
                )

            screen = _ensure_screen(pygame, screen, emulator.display_size)

            _draw_frame(
                pygame=pygame,
                screen=screen,
                fonts=fonts,
                frame=frame,
                episode=episode,
                info=info,
                reset_info=reset_info,
                episode_reward=0.0,
                paused=paused,
                joypad_mask=current_joypad_mask,
                telemetry=telemetry,
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
                current_joypad_mask = viewer_input.joypad_mask
                emulator.set_joypad_mask(viewer_input.joypad_mask)
                if viewer_input.save_state:
                    _save_baseline_state(
                        emulator=emulator,
                        baseline_state_path=config.emulator.baseline_state_path,
                    )
                if paused and not viewer_input.step_once:
                    _draw_frame(
                        pygame=pygame,
                        screen=screen,
                        fonts=fonts,
                        frame=frame,
                        episode=episode,
                        info=info,
                        reset_info=reset_info,
                        episode_reward=episode_reward,
                        paused=True,
                        joypad_mask=current_joypad_mask,
                        telemetry=telemetry,
                    )
                    time.sleep(0.01)
                    continue

                if paused and viewer_input.step_once:
                    frame, reward, terminated, truncated, info = env.step_frame()
                    episode_reward += reward
                    telemetry = _read_live_telemetry(emulator)
                    _draw_frame(
                        pygame=pygame,
                        screen=screen,
                        fonts=fonts,
                        frame=frame,
                        episode=episode,
                        info=info,
                        reset_info=reset_info,
                        episode_reward=episode_reward,
                        paused=True,
                        joypad_mask=current_joypad_mask,
                        telemetry=telemetry,
                    )
                    continue

                frame_start = time.perf_counter()
                frame, reward, terminated, truncated, info = env.step(0)
                episode_reward += reward
                telemetry = _read_live_telemetry(emulator)

                screen = _ensure_screen(pygame, screen, emulator.display_size)
                _draw_frame(
                    pygame=pygame,
                    screen=screen,
                    fonts=fonts,
                    frame=frame,
                    episode=episode,
                    info=info,
                    reset_info=reset_info,
                    episode_reward=episode_reward,
                    paused=paused,
                    joypad_mask=current_joypad_mask,
                    telemetry=telemetry,
                )

                if paused:
                    continue

                if target_seconds is None:
                    raise RuntimeError("Watch target_seconds was not initialized")
                elapsed = time.perf_counter() - frame_start
                delay = max(0.0, target_seconds - elapsed)
                if delay:
                    time.sleep(delay)
            episode += 1
    finally:
        env.close()
        pygame.quit()


def _create_screen(pygame, game_display_size: tuple[int, int]):
    screen = pygame.display.set_mode(_window_size(game_display_size))
    pygame.display.set_caption("F-Zero X Watch")
    return screen


def _ensure_screen(pygame, screen, game_display_size: tuple[int, int]):
    if screen.get_size() == _window_size(game_display_size):
        return screen
    return _create_screen(pygame, game_display_size)


def _create_fonts(pygame) -> ViewerFonts:
    return ViewerFonts(
        title=pygame.font.Font(None, FONT_SIZES.title),
        section=pygame.font.Font(None, FONT_SIZES.section),
        body=pygame.font.Font(None, FONT_SIZES.body),
        small=pygame.font.Font(None, FONT_SIZES.small),
    )


def _draw_frame(
    *,
    pygame,
    screen,
    fonts,
    frame: np.ndarray,
    episode: int,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode_reward: float,
    paused: bool,
    joypad_mask: int,
    telemetry: FZeroXTelemetry | None,
) -> None:
    frame_height, frame_width, _ = frame.shape
    surface = pygame.image.frombuffer(frame.tobytes(), (frame_width, frame_height), "RGB")
    target_size = display_size(frame.shape, _display_aspect_ratio(info))
    if surface.get_size() != target_size:
        surface = pygame.transform.scale(surface, target_size)
    screen.fill(PALETTE.app_background)
    screen.blit(surface, (0, 0))
    panel_rect = pygame.Rect(
        target_size[0],
        0,
        LAYOUT.panel_width,
        _window_size(target_size)[1],
    )
    _draw_side_panel(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        panel_rect=panel_rect,
        episode=episode,
        info=info,
        reset_info=reset_info,
        episode_reward=episode_reward,
        paused=paused,
        joypad_mask=joypad_mask,
        game_display_size=target_size,
        telemetry=telemetry,
    )
    pygame.display.flip()


def _draw_side_panel(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    panel_rect,
    episode: int,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode_reward: float,
    paused: bool,
    joypad_mask: int,
    game_display_size: tuple[int, int],
    telemetry: FZeroXTelemetry | None,
) -> None:
    pygame.draw.rect(screen, PALETTE.panel_background, panel_rect)
    pygame.draw.line(
        screen,
        PALETTE.panel_border,
        panel_rect.topleft,
        panel_rect.bottomleft,
        width=2,
    )

    x = panel_rect.x + LAYOUT.panel_padding
    y = panel_rect.y + LAYOUT.panel_padding
    panel_width = panel_rect.width - (2 * LAYOUT.panel_padding)
    columns = _build_panel_columns(
        episode=episode,
        info=info,
        reset_info=reset_info,
        episode_reward=episode_reward,
        paused=paused,
        joypad_mask=joypad_mask,
        game_display_size=game_display_size,
        telemetry=telemetry,
    )

    y = _draw_panel_title(
        screen=screen,
        fonts=fonts,
        x=x,
        y=y,
        title="F-Zero X Watch",
        subtitle="live emulator session",
    )
    y += LAYOUT.title_section_gap

    content_width = panel_width
    left_column_width = (content_width - LAYOUT.column_gap) // 2
    right_column_width = content_width - LAYOUT.column_gap - left_column_width
    left_x = x
    right_x = x + left_column_width + LAYOUT.column_gap

    _draw_column(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=left_x,
        y=y,
        width=left_column_width,
        sections=columns.left,
    )
    _draw_column(
        pygame=pygame,
        screen=screen,
        fonts=fonts,
        x=right_x,
        y=y,
        width=right_column_width,
        sections=columns.right,
    )


def _build_panel_columns(
    *,
    episode: int,
    info: dict[str, object],
    reset_info: dict[str, object],
    episode_reward: float,
    paused: bool,
    joypad_mask: int,
    game_display_size: tuple[int, int],
    telemetry: FZeroXTelemetry | None,
) -> PanelColumns:
    return PanelColumns(
        left=[
            PanelSection(
                title="Session",
                lines=[
                    _panel_line(
                        "State",
                        "paused" if paused else "running",
                        PALETTE.text_warning if paused else PALETTE.text_accent,
                    ),
                    _panel_line("Episode", str(episode), PALETTE.text_primary),
                    _panel_line("Frame", str(info.get("frame_index", 0)), PALETTE.text_primary),
                    _panel_line(
                        "Step",
                        f"{_float_info(info, 'step_reward'):.2f}",
                        PALETTE.text_primary,
                    ),
                    _panel_line("Return", f"{episode_reward:.2f}", PALETTE.text_primary),
                ],
            ),
            PanelSection(
                title="Reset",
                lines=[
                    _panel_line(
                        "Mode",
                        str(reset_info.get("reset_mode", "baseline")),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Baseline",
                        str(reset_info.get("baseline_kind", "unknown")),
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "Boot",
                        str(reset_info.get("boot_state", "-")),
                        PALETTE.text_primary,
                    ),
                ],
            ),
            PanelSection(
                title="Input",
                lines=[
                    _panel_line(
                        "Held",
                        _pressed_button_labels(joypad_mask),
                        PALETTE.text_primary,
                    ),
                ],
            ),
            PanelSection(
                title="Controls",
                lines=[
                    _panel_line("Pad", "Arrows", PALETTE.text_muted),
                    _panel_line("A / B", "X / Z", PALETTE.text_muted),
                    _panel_line("Menu", "Enter / Backspace", PALETTE.text_muted),
                    _panel_line("Playback", "P pause | N step", PALETTE.text_muted),
                    _panel_line("Baseline", "K save", PALETTE.text_muted),
                ],
            ),
        ],
        right=[
            _game_section(telemetry),
            PanelSection(
                title="Display",
                lines=[
                    _panel_line(
                        "Game",
                        f"{game_display_size[0]}x{game_display_size[1]}",
                        PALETTE.text_primary,
                    ),
                    _panel_line(
                        "FPS",
                        f"{_float_info(info, 'native_fps'):.1f}",
                        PALETTE.text_primary,
                    ),
                ],
            ),
        ],
    )


def _draw_column(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    sections: list[PanelSection],
) -> None:
    current_y = y
    for section_index, section in enumerate(sections):
        current_y = _draw_section(
            pygame=pygame,
            screen=screen,
            fonts=fonts,
            x=x,
            y=current_y,
            width=width,
            section=section,
        )
        if section_index < len(sections) - 1:
            current_y += LAYOUT.section_gap


def _draw_panel_title(*, screen, fonts, x: int, y: int, title: str, subtitle: str) -> int:
    title_surface = fonts.title.render(title, True, PALETTE.text_primary)
    subtitle_surface = fonts.small.render(subtitle, True, PALETTE.text_muted)
    screen.blit(title_surface, (x, y))
    subtitle_y = y + title_surface.get_height() + LAYOUT.title_gap
    screen.blit(subtitle_surface, (x, subtitle_y))
    return subtitle_y + subtitle_surface.get_height()


def _draw_section(
    *,
    pygame,
    screen,
    fonts: ViewerFonts,
    x: int,
    y: int,
    width: int,
    section: PanelSection,
) -> int:
    section_title = fonts.section.render(section.title, True, PALETTE.text_primary)
    screen.blit(section_title, (x, y))
    y += section_title.get_height() + LAYOUT.section_title_gap
    pygame.draw.line(screen, PALETTE.panel_border, (x, y), (x + width, y), width=1)
    y += LAYOUT.section_rule_gap

    for line in section.lines:
        if line.label:
            label_surface = fonts.small.render(line.label, True, PALETTE.text_muted)
            screen.blit(label_surface, (x, y))
            value_surface = fonts.body.render(line.value, True, line.color)
            value_x = x + width - value_surface.get_width()
            screen.blit(value_surface, (value_x, y - 1))
            y += max(label_surface.get_height(), value_surface.get_height()) + LAYOUT.line_gap
            continue

        value_surface = fonts.small.render(line.value, True, line.color)
        screen.blit(value_surface, (x, y))
        y += value_surface.get_height() + LAYOUT.line_gap

    return y


def _panel_line(label: str, value: str, color: Color) -> PanelLine:
    return PanelLine(label=label, value=value, color=color)


def _game_section(telemetry: FZeroXTelemetry | None) -> PanelSection:
    if telemetry is None:
        return PanelSection(
            title="Game",
            lines=[
                _panel_line("Status", "unavailable", PALETTE.text_warning),
            ],
        )

    return PanelSection(
        title="Game",
        lines=[
            _panel_line("Mode", _format_mode_name(telemetry.game_mode_name), PALETTE.text_primary),
            _panel_line("Course", str(telemetry.course_index), PALETTE.text_primary),
            _panel_line(
                "Time",
                _format_race_time_ms(telemetry.player.race_time_ms),
                PALETTE.text_primary,
            ),
            _panel_line("Speed", f"{telemetry.player.speed_kph:.1f} km/h", PALETTE.text_primary),
            _panel_line("Boost", str(telemetry.player.boost_timer), PALETTE.text_primary),
            _panel_line(
                "Energy",
                f"{telemetry.player.energy:.1f} / {telemetry.player.max_energy:.1f}",
                PALETTE.text_primary,
            ),
            _panel_line("Lap", str(telemetry.player.lap), PALETTE.text_primary),
            _panel_line("Pos", str(telemetry.player.position), PALETTE.text_primary),
            _panel_line(
                "Progress",
                _format_distance(telemetry.player.race_distance),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Lap prog",
                _format_distance(telemetry.player.lap_distance),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Lap base",
                _format_distance(telemetry.player.laps_completed_distance),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Sort key",
                _format_distance(telemetry.player.race_distance_position),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Flags",
                _format_state_labels(telemetry.player.state_labels),
                PALETTE.text_primary,
            ),
        ],
    )


def _panel_content_height(fonts: ViewerFonts, columns: PanelColumns) -> int:
    title_surface = fonts.title.render("F-Zero X Watch", True, PALETTE.text_primary)
    subtitle_surface = fonts.small.render("live emulator session", True, PALETTE.text_muted)
    y = LAYOUT.panel_padding
    y += title_surface.get_height() + LAYOUT.title_gap + subtitle_surface.get_height()
    y += LAYOUT.title_section_gap

    left_height = _column_content_height(fonts, columns.left)
    right_height = _column_content_height(fonts, columns.right)
    return y + max(left_height, right_height) + LAYOUT.panel_padding


def _column_content_height(fonts: ViewerFonts, sections: list[PanelSection]) -> int:
    y = 0
    for section_index, section in enumerate(sections):
        section_title = fonts.section.render(section.title, True, PALETTE.text_primary)
        y += section_title.get_height() + LAYOUT.section_title_gap
        y += LAYOUT.section_rule_gap
        for line in section.lines:
            if line.label:
                label_surface = fonts.small.render(line.label, True, PALETTE.text_muted)
                value_surface = fonts.body.render(line.value, True, line.color)
                y += max(label_surface.get_height(), value_surface.get_height()) + LAYOUT.line_gap
            else:
                value_surface = fonts.small.render(line.value, True, line.color)
                y += value_surface.get_height() + LAYOUT.line_gap
        if section_index < len(sections) - 1:
            y += LAYOUT.section_gap
    return y


def _window_size(game_display_size: tuple[int, int]) -> tuple[int, int]:
    return game_display_size[0] + LAYOUT.panel_width, game_display_size[1]


def _pressed_button_labels(joypad_mask_value: int) -> str:
    pressed = [
        label
        for button_id, label in BUTTON_LABELS
        if joypad_mask_value & (1 << button_id)
    ]
    return " ".join(pressed) if pressed else "none"


def _display_aspect_ratio(info: dict[str, object]) -> float:
    value = info.get("display_aspect_ratio")
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _float_info(info: dict[str, object], key: str) -> float:
    value = info.get(key)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _format_mode_name(mode_name: str) -> str:
    return mode_name.replace("_", " ")


def _format_race_time_ms(race_time_ms: int) -> str:
    minutes, remainder = divmod(max(0, race_time_ms), 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{minutes:02d}'{seconds:02d}\"{milliseconds:03d}"


def _format_distance(distance: float) -> str:
    return f"{distance:,.1f}"


def _format_state_labels(state_labels: tuple[str, ...]) -> str:
    if not state_labels:
        return "none"
    return " | ".join(state_labels)


def _read_live_telemetry(emulator: Emulator) -> FZeroXTelemetry | None:
    try:
        return read_telemetry(emulator)
    except RuntimeError:
        return None


def _poll_viewer_input(pygame) -> ViewerInput:
    quit_requested = False
    toggle_pause = False
    step_once = False
    save_state = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_requested = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                toggle_pause = True
            elif event.key == pygame.K_n:
                step_once = True
            elif event.key == pygame.K_k:
                save_state = True

    keys = pygame.key.get_pressed()
    pressed_buttons: list[int] = []
    if keys[pygame.K_UP]:
        pressed_buttons.append(JOYPAD_UP)
    if keys[pygame.K_DOWN]:
        pressed_buttons.append(JOYPAD_DOWN)
    if keys[pygame.K_LEFT]:
        pressed_buttons.append(JOYPAD_LEFT)
    if keys[pygame.K_RIGHT]:
        pressed_buttons.append(JOYPAD_RIGHT)
    if keys[pygame.K_x]:
        pressed_buttons.append(JOYPAD_A)
    if keys[pygame.K_z]:
        pressed_buttons.append(JOYPAD_B)
    if keys[pygame.K_RETURN]:
        pressed_buttons.append(JOYPAD_START)
    if keys[pygame.K_BACKSPACE]:
        pressed_buttons.append(JOYPAD_SELECT)

    return ViewerInput(
        quit_requested=quit_requested,
        toggle_pause=toggle_pause,
        step_once=step_once,
        save_state=save_state,
        joypad_mask=joypad_mask(*pressed_buttons),
    )


def _save_baseline_state(*, emulator: Emulator, baseline_state_path: Path | None) -> None:
    emulator.capture_current_as_baseline(baseline_state_path)
