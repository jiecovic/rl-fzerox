# src/rl_fzerox/apps/run_manager/app.py
from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import ValidationError

from rl_fzerox.apps.run_manager.config_form import render_create_run_form
from rl_fzerox.apps.run_manager.dependencies import streamlit_module
from rl_fzerox.apps.run_manager.run_list import render_runs
from rl_fzerox.apps.run_manager.streamlit_types import StreamlitCommands, two_columns
from rl_fzerox.apps.run_manager.style import apply_style
from rl_fzerox.core.manager import ManagerStore, default_manager_db_path


def main() -> None:
    """Render the local SQLite-backed run manager Streamlit app."""

    st = streamlit_module()
    st.set_page_config(page_title="F-Zero X Runs", layout="wide")
    apply_style(st)

    st.title("F-Zero X runs")
    st.caption("SQLite is the source of truth. Manifests are snapshots only.")

    db_path = Path(
        st.sidebar.text_input(
            "Manager DB",
            value=str(default_manager_db_path()),
            help="New managed runs are stored here. Old local/runs are intentionally ignored.",
        )
    ).expanduser()
    managed_runs_root = Path(
        st.sidebar.text_input(
            "Managed run root",
            value=str(Path("local/managed_runs").resolve()),
        )
    ).expanduser()

    store = ManagerStore(db_path)
    store.initialize()

    create_column, runs_column = two_columns(st, (0.9, 1.1), gap="large")
    with create_column:
        _render_create_run(st, store=store, managed_runs_root=managed_runs_root)
    with runs_column:
        render_runs(st, store.list_runs())


def _render_create_run(
    st: StreamlitCommands,
    *,
    store: ManagerStore,
    managed_runs_root: Path,
) -> None:
    st.subheader("Create run")
    templates = store.list_templates()
    template_names = tuple(template.name for template in templates)
    selected_template_name = st.selectbox("Template", template_names)
    selected_template = next(
        template for template in templates if template.name == selected_template_name
    )
    draft = render_create_run_form(st, base=selected_template.config)
    if draft is None:
        return

    try:
        run = store.create_run(
            name=draft.name,
            config=draft.config,
            managed_runs_root=managed_runs_root,
        )
    except (OSError, sqlite3.Error, ValidationError, ValueError) as exc:
        st.error(str(exc))
        return

    st.success(f"Created {run.id}")
    st.code(str(run.run_dir), language="text")
