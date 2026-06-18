# src/rl_fzerox/core/career_mode/controller/lifecycle/post_gp.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.career_mode.navigation import MenuFacts


@dataclass(slots=True)
class PostGpCutsceneTracker:
    """Track the post-GP ceremony without cutting the recording early.

    Completion is the visible FSM transition out of `gp_end_cutscene` into
    credits/menu/title flow. The controller must not infer this from elapsed
    frames because GP ceremonies vary and the full ceremony belongs in the
    recording.
    """

    seen: bool = False

    def progress_info(
        self,
        *,
        facts: MenuFacts,
        info: dict[str, object],
    ) -> dict[str, object]:
        if facts.is_gp_end_cutscene:
            self.seen = True
            return info
        if not self.seen:
            return info

        self.reset()
        annotated = dict(info)
        annotated["career_mode_post_gp_cutscene_complete"] = True
        return annotated

    def reset(self) -> None:
        self.seen = False
