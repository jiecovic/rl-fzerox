# src/rl_fzerox/apps/run_manager/style.py
from __future__ import annotations

from rl_fzerox.apps.run_manager.streamlit_types import StreamlitCommands


def apply_style(st: StreamlitCommands) -> None:
    """Apply restrained app-local Streamlit styling."""

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; max-width: 1380px; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        div[data-testid="stForm"] {
            border: 1px solid rgba(120, 120, 120, 0.22);
            border-radius: 8px;
            padding: 16px 18px 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
