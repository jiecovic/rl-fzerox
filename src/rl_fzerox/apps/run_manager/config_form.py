# src/rl_fzerox/apps/run_manager/config_form.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.apps.run_manager.streamlit_types import StreamlitCommands, two_columns
from rl_fzerox.core.manager import ManagedRunConfig


@dataclass(frozen=True, slots=True)
class CreateRunDraft:
    """Validated user input ready to be frozen into a DB-managed run."""

    name: str
    config: ManagedRunConfig


def render_create_run_form(
    st: StreamlitCommands,
    *,
    base: ManagedRunConfig,
) -> CreateRunDraft | None:
    """Render the first run-creation form and return a submitted config draft."""

    with st.form("create-managed-run", clear_on_submit=False):
        run_name = st.text_input("Run name", value="ppo_allcups_recurrent")
        seed = int(st.number_input("Seed", value=base.seed, step=1))

        st.markdown("**Training**")
        train_col_a, train_col_b = two_columns(st, 2)
        with train_col_a:
            num_envs = int(
                st.number_input("Env count", min_value=1, value=base.train.num_envs, step=1)
            )
            n_steps = int(
                st.number_input("Rollout steps", min_value=1, value=base.train.n_steps, step=256)
            )
            batch_size = int(
                st.number_input("Batch size", min_value=1, value=base.train.batch_size, step=128)
            )
        with train_col_b:
            total_timesteps = int(
                st.number_input(
                    "Target steps",
                    min_value=1,
                    value=base.train.total_timesteps,
                    step=1_000_000,
                )
            )
            learning_rate = float(
                st.number_input(
                    "Learning rate",
                    min_value=1e-8,
                    value=base.train.learning_rate,
                    format="%.8f",
                )
            )
            ent_coef = float(
                st.number_input(
                    "Entropy coefficient",
                    min_value=0.0,
                    value=base.train.ent_coef,
                    format="%.5f",
                )
            )

        st.markdown("**Observation / policy**")
        obs_col_a, obs_col_b = two_columns(st, 2)
        with obs_col_a:
            frame_stack = int(
                st.number_input(
                    "Frame stack",
                    min_value=1,
                    max_value=8,
                    value=base.observation.frame_stack,
                    step=1,
                )
            )
            stack_mode = st.selectbox(
                "Stack mode",
                ("rgb", "gray", "luma_chroma"),
                index=("rgb", "gray", "luma_chroma").index(base.observation.stack_mode),
            )
            progress_source = st.selectbox(
                "Progress scalar",
                ("lap_progress", "segment_progress", "none"),
                index=("lap_progress", "segment_progress", "none").index(
                    base.observation.progress_source
                ),
            )
        with obs_col_b:
            conv_profile = st.selectbox(
                "CNN profile",
                ("nature", "nature_32_64_128", "nature_wide"),
                index=("nature", "nature_32_64_128", "nature_wide").index(
                    base.policy.conv_profile
                ),
            )
            recurrent_hidden_size = int(
                st.number_input(
                    "LSTM hidden size",
                    min_value=1,
                    value=base.policy.recurrent_hidden_size,
                    step=64,
                )
            )
            fusion_features_dim = int(
                st.number_input(
                    "Fusion features",
                    min_value=1,
                    value=base.policy.fusion_features_dim,
                    step=128,
                )
            )

        st.markdown("**Reward**")
        reward_col_a, reward_col_b = two_columns(st, 2)
        with reward_col_a:
            manual_boost_reward = float(
                st.number_input(
                    "Boost use reward",
                    min_value=0.0,
                    value=base.reward.manual_boost_reward,
                    format="%.4f",
                )
            )
            boost_pad_reward = float(
                st.number_input(
                    "Boost pad reward",
                    min_value=0.0,
                    value=base.reward.boost_pad_reward,
                    format="%.2f",
                )
            )
            airborne_pitch_up_penalty = float(
                st.number_input(
                    "Airborne pitch-up penalty",
                    max_value=0.0,
                    value=base.reward.airborne_pitch_up_penalty,
                    format="%.4f",
                )
            )
        with reward_col_b:
            lean_request_penalty = float(
                st.number_input(
                    "Lean request penalty",
                    max_value=0.0,
                    value=base.reward.lean_request_penalty,
                    format="%.4f",
                )
            )
            lean_low_speed_penalty = float(
                st.number_input(
                    "Low-speed lean penalty",
                    max_value=0.0,
                    value=base.reward.lean_low_speed_penalty,
                    format="%.4f",
                )
            )
            lean_low_speed_penalty_max_speed_kph = float(
                st.number_input(
                    "Low-speed lean cutoff kph",
                    min_value=0.0,
                    value=base.reward.lean_low_speed_penalty_max_speed_kph,
                    format="%.1f",
                )
            )

        if not st.form_submit_button("Create immutable run"):
            return None

    config = ManagedRunConfig.model_validate(
        {
            "seed": seed,
            "train": {
                **base.train.model_dump(mode="json"),
                "num_envs": num_envs,
                "total_timesteps": total_timesteps,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "ent_coef": ent_coef,
            },
            "observation": {
                **base.observation.model_dump(mode="json"),
                "frame_stack": frame_stack,
                "stack_mode": stack_mode,
                "progress_source": progress_source,
            },
            "policy": {
                **base.policy.model_dump(mode="json"),
                "conv_profile": conv_profile,
                "recurrent_hidden_size": recurrent_hidden_size,
                "fusion_features_dim": fusion_features_dim,
            },
            "reward": {
                **base.reward.model_dump(mode="json"),
                "manual_boost_reward": manual_boost_reward,
                "boost_pad_reward": boost_pad_reward,
                "lean_request_penalty": lean_request_penalty,
                "lean_low_speed_penalty": lean_low_speed_penalty,
                "lean_low_speed_penalty_max_speed_kph": lean_low_speed_penalty_max_speed_kph,
                "airborne_pitch_up_penalty": airborne_pitch_up_penalty,
            },
        }
    )
    return CreateRunDraft(name=run_name, config=config)
