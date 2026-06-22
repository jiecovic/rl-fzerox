# src/rl_fzerox/core/career_mode/controller/lifecycle/progress_sync.py
from __future__ import annotations

from dataclasses import dataclass, field

from rl_fzerox.core.career_mode.controller.lifecycle.post_gp import PostGpCutsceneTracker
from rl_fzerox.core.career_mode.controller.lifecycle.post_race import PostRaceContinuation
from rl_fzerox.core.career_mode.controller.lifecycle.recording import (
    CareerRecordingSegmentTracker,
)
from rl_fzerox.core.career_mode.controller.lifecycle.terminal import (
    post_terminal_progress_screen,
)
from rl_fzerox.core.career_mode.execution.save_file import SaveRamRuntimeSession
from rl_fzerox.core.career_mode.navigation import MenuFacts
from rl_fzerox.core.career_mode.progress.attempt import (
    CareerAttemptProgress,
    CareerProgressTransition,
)
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig


@dataclass(slots=True)
class CareerPostTerminalProgressSync:
    """Sync save progress from post-race screens after visible GP result flow.

    A finished GP can show a winning ceremony before credits/menu state. The
    controller polls that visible flow here and only returns a progress
    transition once the screen is a stable lifecycle boundary.
    """

    cutscene: PostGpCutsceneTracker = field(default_factory=PostGpCutsceneTracker)
    last_sync_key: tuple[str | None, object | None, object | None, bool] | None = None

    def reset(self) -> None:
        self.cutscene.reset()
        self.last_sync_key = None

    def sync(
        self,
        *,
        session: SaveRamRuntimeSession,
        setup: CareerModeRaceSetupConfig,
        progress: CareerAttemptProgress,
        recording: CareerRecordingSegmentTracker,
        post_race: PostRaceContinuation,
        facts: MenuFacts,
        info: dict[str, object],
    ) -> CareerProgressTransition | None:
        if not post_terminal_progress_screen(facts):
            self.cutscene.reset()
            return None

        progress_info = self.cutscene.progress_info(facts=facts, info=info)
        sync_key = (
            facts.game_mode,
            progress_info.get("career_mode_gp_final_rank"),
            progress_info.get("career_mode_gp_points"),
            progress_info.get("career_mode_post_gp_cutscene_complete") is True,
        )
        if self.last_sync_key == sync_key:
            return CareerProgressTransition(attempt_finished=False)

        recording.observe_progress_screen(facts, progress_info)
        post_race.mark_progress_synced()
        self.last_sync_key = sync_key
        return progress.sync_post_terminal_progress(
            session=session,
            setup=setup,
            info=progress_info,
        )
