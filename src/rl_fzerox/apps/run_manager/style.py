# src/rl_fzerox/apps/run_manager/style.py
from __future__ import annotations

from rl_fzerox.apps.run_manager.streamlit_types import StreamlitCommands


def apply_style(st: StreamlitCommands) -> None:
    """Apply restrained app-local Streamlit styling."""

    st.markdown(
        """
        <style>
        [data-testid="stHeader"] {
            background: transparent;
        }
        .block-container { padding-top: 1.4rem; max-width: 1380px; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        div[data-testid="stForm"] {
            border-radius: 8px;
            padding: 18px 20px 14px;
        }
        div[data-testid="stForm"] div[data-testid="stVerticalBlock"] {
            gap: 0.65rem;
        }
        div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {
            gap: 1.25rem;
        }
        .manager-muted {
            opacity: 0.64;
            font-size: 0.86rem;
        }
        [aria-label="Record a screencast"],
        [title="Record a screencast"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
