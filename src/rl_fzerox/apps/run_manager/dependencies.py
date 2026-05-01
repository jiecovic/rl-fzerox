# src/rl_fzerox/apps/run_manager/dependencies.py
from __future__ import annotations

from importlib import import_module

from rl_fzerox.apps.run_manager.streamlit_types import StreamlitModule


def streamlit_module() -> StreamlitModule:
    """Load Streamlit lazily so normal imports/tests do not require the UI extra."""

    try:
        module = import_module("streamlit")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Streamlit is not installed. Install the manager extra, then run: "
            "python -m streamlit run src/rl_fzerox/apps/run_manager/app.py"
        ) from exc
    if not isinstance(module, StreamlitModule):
        raise RuntimeError("Imported streamlit module does not expose the expected UI API")
    return module
