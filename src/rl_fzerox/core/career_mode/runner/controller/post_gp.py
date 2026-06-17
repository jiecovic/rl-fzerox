# src/rl_fzerox/core/career_mode/runner/controller/post_gp.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.career_mode.navigation import MENU_TIMING, MenuFacts


@dataclass(slots=True)
class PostGpCutsceneTracker:
    """Delay post-GP completion until the winning ceremony has been recorded."""

    start_frame: int | None = None
    polls: int = 0

    def progress_info(
        self,
        *,
        facts: MenuFacts,
        info: dict[str, object],
    ) -> dict[str, object]:
        if not facts.is_gp_end_cutscene:
            self.reset()
            return info

        frame_index = _int_info(info, "frame_index")
        if self.start_frame is None:
            self.start_frame = frame_index
            self.polls = 0
        self.polls += 1

        elapsed_frames = (
            frame_index - self.start_frame
            if frame_index is not None and self.start_frame is not None
            else self.polls * MENU_TIMING.menu_hold_frames
        )
        if elapsed_frames < MENU_TIMING.post_gp_cutscene_record_frames:
            return info

        annotated = dict(info)
        annotated["career_mode_post_gp_cutscene_complete"] = True
        return annotated

    def reset(self) -> None:
        self.start_frame = None
        self.polls = 0


def _int_info(info: dict[str, object], key: str) -> int | None:
    value = info.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
