# src/rl_fzerox/apps/run_manager/run_list.py
from __future__ import annotations

from collections.abc import Callable

from rl_fzerox.apps.run_manager.streamlit_types import StreamlitCommands, two_columns
from rl_fzerox.core.manager import ManagedRun, ManagedRunDraft
from rl_fzerox.core.manager.config import ManagedRunConfig

DRAFT_SELECT_KEY = "run_manager_draft_select"
RUN_SELECT_KEY = "run_manager_run_select"
VISIBLE_RUN_STATUSES = frozenset(("running", "paused", "stopped", "finished", "failed"))


def render_drafts(
    st: StreamlitCommands,
    *,
    drafts: tuple[ManagedRunDraft, ...],
    delete_draft: Callable[[str], bool],
) -> None:
    """Render SQLite-only run drafts with one selected inspector."""

    st.subheader("Drafts")
    if not drafts:
        st.info("No drafts yet. Save one from the configurator.")
        return

    selected = _select_draft(st, drafts)
    _render_draft_inspector(st, selected=selected, delete_draft=delete_draft)


def render_runs(st: StreamlitCommands, *, runs: tuple[ManagedRun, ...]) -> None:
    """Render launched training runs only."""

    st.subheader("Runs")
    visible_runs = tuple(run for run in runs if run.status in VISIBLE_RUN_STATUSES)
    if not visible_runs:
        st.info("No launched runs yet. The Train button is intentionally disabled in this slice.")
        return

    selected = _select_run(st, visible_runs)
    _render_run_inspector(st, selected=selected)


def _select_draft(
    st: StreamlitCommands,
    drafts: tuple[ManagedRunDraft, ...],
) -> ManagedRunDraft:
    draft_ids = tuple(draft.id for draft in drafts)
    selected_id = st.selectbox(
        "Draft",
        draft_ids,
        key=DRAFT_SELECT_KEY,
        format_func={draft.id: draft.name for draft in drafts}.__getitem__,
    )
    return next(draft for draft in drafts if draft.id == selected_id)


def _select_run(st: StreamlitCommands, runs: tuple[ManagedRun, ...]) -> ManagedRun:
    run_ids = tuple(run.id for run in runs)
    selected_id = st.selectbox(
        "Run",
        run_ids,
        key=RUN_SELECT_KEY,
        format_func={run.id: f"{run.name} ({run.status})" for run in runs}.__getitem__,
    )
    return next(run for run in runs if run.id == selected_id)


def _render_draft_inspector(
    st: StreamlitCommands,
    *,
    selected: ManagedRunDraft,
    delete_draft: Callable[[str], bool],
) -> None:
    with st.container(border=True):
        st.markdown(f"### {selected.name}")
        st.markdown(
            f"<span class='manager-muted'>saved {selected.updated_at}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("Saved configuration. Training launch is not wired yet.")
        _render_config_summary(st, selected.config)
        action_column, delete_column = two_columns(st, (0.28, 0.72), gap="medium")
        with action_column:
            st.button(
                "Train",
                disabled=True,
                help="Next slice: freeze this draft and start training.",
                use_container_width=True,
            )
        with delete_column:
            if st.button("Delete draft", key=f"delete-draft-{selected.id}"):
                delete_draft(selected.id)
                st.session_state.pop(DRAFT_SELECT_KEY, None)
                st.rerun()


def _render_run_inspector(st: StreamlitCommands, *, selected: ManagedRun) -> None:
    with st.container(border=True):
        st.markdown(f"### {selected.name}")
        st.markdown(
            f"<span class='manager-muted'>created {selected.created_at}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Status:** {selected.status}")
        _render_config_summary(st, selected.config)
        action_column, inspect_column = two_columns(st, (0.28, 0.72), gap="medium")
        with action_column:
            st.button("Fork", disabled=True, use_container_width=True)
        with inspect_column:
            st.button("Inspect", disabled=True)


def _render_config_summary(st: StreamlitCommands, config: ManagedRunConfig) -> None:
    rows = (
        (
            "Training",
            f"{config.train.num_envs} envs, {config.train.total_timesteps:,} target steps, "
            f"rollout {config.train.n_steps}, batch {config.train.batch_size}, "
            f"lr {config.train.learning_rate:.2e}, entropy {config.train.ent_coef:g}",
        ),
        (
            "Observation",
            f"{config.observation.stack_mode} x{config.observation.frame_stack}, "
            f"{config.observation.preset}, progress {config.observation.progress_source}",
        ),
        (
            "Policy",
            f"{config.policy.conv_profile}, LSTM {config.policy.recurrent_hidden_size}, "
            f"fusion {config.policy.fusion_features_dim}, pi {list(config.policy.pi_net_arch)}, "
            f"vf {list(config.policy.vf_net_arch)}",
        ),
        (
            "Reward",
            f"boost use {config.reward.manual_boost_reward:g}, "
            f"boost pad {config.reward.boost_pad_reward:g}, "
            f"lean request {config.reward.lean_request_penalty:g}, "
            f"low-speed lean {config.reward.lean_low_speed_penalty:g}",
        ),
    )
    for label, value in rows:
        st.markdown(f"**{label}:** {value}")
