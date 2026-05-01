# src/rl_fzerox/apps/run_manager/run_list.py
from __future__ import annotations

from rl_fzerox.apps.run_manager.streamlit_types import StreamlitCommands
from rl_fzerox.core.manager import ManagedRun


def render_runs(st: StreamlitCommands, runs: tuple[ManagedRun, ...]) -> None:
    """Render DB-managed runs without inspecting old local/runs folders."""

    st.subheader("Managed runs")
    if not runs:
        st.info("No DB-managed runs yet.")
        return

    st.dataframe(
        [
            {
                "created": run.created_at,
                "status": run.status,
                "name": run.name,
                "envs": run.config.train.num_envs,
                "steps": run.config.train.total_timesteps,
                "lr": run.config.train.learning_rate,
                "entropy": run.config.train.ent_coef,
                "hash": run.config_hash,
                "run_dir": str(run.run_dir),
            }
            for run in runs
        ],
        hide_index=True,
        use_container_width=True,
    )

    selected_id = st.selectbox("Inspect run", tuple(run.id for run in runs))
    selected = next(run for run in runs if run.id == selected_id)
    st.code(str(selected.run_dir), language="text")
    st.json(selected.config.model_dump(mode="json"), expanded=False)
