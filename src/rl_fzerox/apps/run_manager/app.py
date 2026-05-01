# src/rl_fzerox/apps/run_manager/app.py
from __future__ import annotations

import sqlite3

from pydantic import ValidationError

from rl_fzerox.apps.run_manager.config_form import render_config_form
from rl_fzerox.apps.run_manager.dependencies import streamlit_module
from rl_fzerox.apps.run_manager.run_list import render_drafts, render_runs
from rl_fzerox.apps.run_manager.streamlit_types import StreamlitCommands, three_tabs
from rl_fzerox.apps.run_manager.style import apply_style
from rl_fzerox.core.manager import ManagerStore


def main() -> None:
    """Render the local SQLite-backed run manager Streamlit app."""

    st = streamlit_module()
    st.set_page_config(page_title="F-Zero X Runs", layout="wide")
    apply_style(st)

    st.title("F-Zero X runs")
    st.caption("Local run configuration and lifecycle control.")

    store = ManagerStore()
    store.initialize()

    configurator_tab, drafts_tab, runs_tab = three_tabs(
        st,
        ("Run configurator", "Drafts", "Runs"),
    )
    with configurator_tab:
        _render_configurator(st, store=store)
    with drafts_tab:
        render_drafts(
            st,
            drafts=store.list_drafts(),
            delete_draft=store.delete_draft,
        )
    with runs_tab:
        render_runs(
            st,
            runs=store.list_runs(),
        )


def _render_configurator(
    st: StreamlitCommands,
    *,
    store: ManagerStore,
) -> None:
    st.subheader("Run configurator")
    templates = store.list_templates()
    template_names = tuple(template.name for template in templates)
    selected_template_name = st.selectbox("Template", template_names)
    selected_template = next(
        template for template in templates if template.name == selected_template_name
    )
    draft = render_config_form(st, base=selected_template.config)
    if draft is None:
        st.button("Train", disabled=True, help="Training launch is intentionally not wired yet.")
        return

    try:
        saved_draft = store.create_draft(
            name=draft.name,
            config=draft.config,
        )
    except (OSError, sqlite3.Error, ValidationError, ValueError) as exc:
        st.error(str(exc))
        return

    st.success(f"Saved draft {saved_draft.name}")
    st.button("Train", disabled=True, help="Next slice: freeze draft, create run, launch trainer.")


if __name__ == "__main__":
    main()
