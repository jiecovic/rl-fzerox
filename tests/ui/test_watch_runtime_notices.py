# tests/ui/test_watch_runtime_notices.py
"""Watch runtime tests for transient UI notices.

Timed notices are copied into frame info only while active and must not mutate
the source info object owned by the runtime loop.
"""

from rl_fzerox.ui.watch.runtime.live.notices import _TimedWatchNotice


def test_timed_watch_notice_persists_until_expiry_without_mutating_source() -> None:
    notice = _TimedWatchNotice()
    source: dict[str, object] = {"track_id": "mute_city"}

    notice.show("alt baseline saved", now=10.0)
    active_info = notice.apply(source, now=12.0)
    expired_info = notice.apply(source, now=14.0)

    assert active_info == {
        "track_id": "mute_city",
        "watch_save_notice": "alt baseline saved",
    }
    assert source == {"track_id": "mute_city"}
    assert expired_info is source
