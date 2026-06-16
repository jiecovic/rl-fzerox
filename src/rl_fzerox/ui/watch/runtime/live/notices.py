# src/rl_fzerox/ui/watch/runtime/live/notices.py
from __future__ import annotations

from dataclasses import dataclass

_WATCH_SAVE_NOTICE_SECONDS = 3.0


@dataclass(slots=True)
class _TimedWatchNotice:
    message: str | None = None
    expires_at: float = 0.0

    def show(self, message: str, *, now: float) -> None:
        self.message = message
        self.expires_at = now + _WATCH_SAVE_NOTICE_SECONDS

    def apply(self, info: dict[str, object], *, now: float) -> dict[str, object]:
        if self.message is None or now >= self.expires_at:
            return info
        with_notice = dict(info)
        with_notice["watch_save_notice"] = self.message
        return with_notice
