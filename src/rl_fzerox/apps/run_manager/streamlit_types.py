# src/rl_fzerox/apps/run_manager/streamlit_types.py
from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Protocol, runtime_checkable


class StreamlitCommands(Protocol):
    """Small typed subset of Streamlit used by the run manager UI."""

    session_state: MutableMapping[str, object]

    def markdown(self, body: str, **kwargs: object) -> object: ...
    def subheader(self, body: str, **kwargs: object) -> object: ...
    def info(self, body: str, **kwargs: object) -> object: ...
    def success(self, body: str, **kwargs: object) -> object: ...
    def error(self, body: str, **kwargs: object) -> object: ...
    def text_input(self, label: str, **kwargs: object) -> str: ...
    def number_input(self, label: str, **kwargs: object) -> float: ...
    def selectbox(self, label: str, options: Sequence[str], **kwargs: object) -> str: ...
    def button(self, label: str, **kwargs: object) -> bool: ...
    def tabs(self, tabs: Sequence[str]) -> tuple[StreamlitContainer, ...]: ...
    def form_submit_button(self, label: str, **kwargs: object) -> bool: ...
    def form(self, key: str, **kwargs: object) -> StreamlitContainer: ...
    def container(self, **kwargs: object) -> StreamlitContainer: ...
    def rerun(self) -> object: ...
    def columns(
        self,
        spec: int | Sequence[float],
        **kwargs: object,
    ) -> tuple[StreamlitContainer, ...]: ...


class StreamlitContainer(StreamlitCommands, Protocol):
    """Streamlit container/column/form object used as a context manager."""

    def __enter__(self) -> StreamlitContainer: ...
    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> bool | None: ...


@runtime_checkable
class StreamlitModule(StreamlitCommands, Protocol):
    """Typed subset of the imported streamlit module."""

    def set_page_config(self, **kwargs: object) -> object: ...
    def title(self, body: str, **kwargs: object) -> object: ...
    def caption(self, body: str, **kwargs: object) -> object: ...


def two_columns(
    st: StreamlitCommands,
    spec: int | tuple[float, float],
    *,
    gap: str | None = None,
) -> tuple[StreamlitContainer, StreamlitContainer]:
    """Return exactly two Streamlit columns with a checked shape."""

    columns = st.columns(spec, gap=gap)
    if len(columns) != 2:
        raise RuntimeError(f"Expected two Streamlit columns, got {len(columns)}")
    return columns[0], columns[1]


def three_tabs(
    st: StreamlitCommands,
    labels: tuple[str, str, str],
) -> tuple[StreamlitContainer, StreamlitContainer, StreamlitContainer]:
    """Return exactly three Streamlit tabs with a checked shape."""

    tabs = st.tabs(labels)
    if len(tabs) != 3:
        raise RuntimeError(f"Expected three Streamlit tabs, got {len(tabs)}")
    return tabs[0], tabs[1], tabs[2]
