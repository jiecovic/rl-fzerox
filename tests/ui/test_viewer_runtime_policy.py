# tests/ui/test_viewer_runtime_policy.py
"""Watch runtime tests for policy reload diagnostics.

Policy reload failures are persisted outside the live worker so the UI can show
useful errors without repeatedly appending the same message.
"""

from pathlib import Path

from rl_fzerox.ui.watch.runtime.policy import _persist_reload_error


def test_persist_reload_error_writes_full_message_once(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "watch" / "runtime"
    runtime_dir.mkdir(parents=True)

    logged_error = _persist_reload_error(
        reload_error="PyTorchStreamReader failed reading file data/0",
        runtime_dir=runtime_dir,
        last_logged_reload_error=None,
    )

    assert logged_error == "PyTorchStreamReader failed reading file data/0"
    assert (tmp_path / "watch" / "reload_error.log").read_text(encoding="utf-8") == (
        "PyTorchStreamReader failed reading file data/0\n"
    )
