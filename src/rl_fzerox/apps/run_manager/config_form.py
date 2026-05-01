# src/rl_fzerox/apps/run_manager/config_form.py
from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.apps.run_manager.streamlit_types import (
    StreamlitCommands,
    three_tabs,
    two_columns,
)
from rl_fzerox.core.manager import ManagedRunConfig


@dataclass(frozen=True, slots=True)
class DraftConfigInput:
    """Validated user input ready to be saved as a SQLite draft."""

    name: str
    config: ManagedRunConfig


@dataclass(frozen=True, slots=True)
class TrainingFormValues:
    num_envs: int
    total_timesteps: int
    n_steps: int
    batch_size: int
    learning_rate: float
    ent_coef: float


@dataclass(frozen=True, slots=True)
class ModelFormValues:
    frame_stack: int
    stack_mode: str
    progress_source: str
    conv_profile: str
    recurrent_hidden_size: int
    fusion_features_dim: int


@dataclass(frozen=True, slots=True)
class RewardFormValues:
    manual_boost_reward: float
    boost_pad_reward: float
    airborne_pitch_up_penalty: float
    lean_request_penalty: float
    lean_low_speed_penalty: float
    lean_low_speed_penalty_max_speed_kph: float


def render_config_form(
    st: StreamlitCommands,
    *,
    base: ManagedRunConfig,
) -> DraftConfigInput | None:
    """Render the first run configurator and return a submitted draft."""

    with st.form("save-managed-draft", clear_on_submit=False, enter_to_submit=False):
        draft_name = st.text_input("Draft name", value="ppo_allcups_recurrent")
        seed = int(st.number_input("Seed", value=base.seed, step=1))

        training_tab, model_tab, reward_tab = three_tabs(
            st,
            ("Training", "Observation / policy", "Reward"),
        )
        with training_tab:
            training = _render_training_form(st, base)
        with model_tab:
            model = _render_model_form(st, base)
        with reward_tab:
            reward = _render_reward_form(st, base)

        if not st.form_submit_button("Save draft"):
            return None

    config = ManagedRunConfig.model_validate(
        {
            "seed": seed,
            "train": {
                **base.train.model_dump(mode="json"),
                "num_envs": training.num_envs,
                "total_timesteps": training.total_timesteps,
                "n_steps": training.n_steps,
                "batch_size": training.batch_size,
                "learning_rate": training.learning_rate,
                "ent_coef": training.ent_coef,
            },
            "observation": {
                **base.observation.model_dump(mode="json"),
                "frame_stack": model.frame_stack,
                "stack_mode": model.stack_mode,
                "progress_source": model.progress_source,
            },
            "policy": {
                **base.policy.model_dump(mode="json"),
                "conv_profile": model.conv_profile,
                "recurrent_hidden_size": model.recurrent_hidden_size,
                "fusion_features_dim": model.fusion_features_dim,
            },
            "reward": {
                **base.reward.model_dump(mode="json"),
                "manual_boost_reward": reward.manual_boost_reward,
                "boost_pad_reward": reward.boost_pad_reward,
                "lean_request_penalty": reward.lean_request_penalty,
                "lean_low_speed_penalty": reward.lean_low_speed_penalty,
                "lean_low_speed_penalty_max_speed_kph": (
                    reward.lean_low_speed_penalty_max_speed_kph
                ),
                "airborne_pitch_up_penalty": reward.airborne_pitch_up_penalty,
            },
        }
    )
    return DraftConfigInput(name=draft_name, config=config)


def _render_training_form(
    st: StreamlitCommands,
    base: ManagedRunConfig,
) -> TrainingFormValues:
    left, right = two_columns(st, 2)
    with left:
        num_envs = int(
            st.number_input("Env count", min_value=1, value=base.train.num_envs, step=1)
        )
        n_steps = int(
            st.number_input("Rollout steps", min_value=1, value=base.train.n_steps, step=256)
        )
        batch_size = int(
            st.number_input("Batch size", min_value=1, value=base.train.batch_size, step=128)
        )
    with right:
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
                format="%.2e",
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
    return TrainingFormValues(
        num_envs=num_envs,
        total_timesteps=total_timesteps,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
    )


def _render_model_form(
    st: StreamlitCommands,
    base: ManagedRunConfig,
) -> ModelFormValues:
    left, right = two_columns(st, 2)
    with left:
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
    with right:
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
    return ModelFormValues(
        frame_stack=frame_stack,
        stack_mode=stack_mode,
        progress_source=progress_source,
        conv_profile=conv_profile,
        recurrent_hidden_size=recurrent_hidden_size,
        fusion_features_dim=fusion_features_dim,
    )


def _render_reward_form(
    st: StreamlitCommands,
    base: ManagedRunConfig,
) -> RewardFormValues:
    left, right = two_columns(st, 2)
    with left:
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
    with right:
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
    return RewardFormValues(
        manual_boost_reward=manual_boost_reward,
        boost_pad_reward=boost_pad_reward,
        airborne_pitch_up_penalty=airborne_pitch_up_penalty,
        lean_request_penalty=lean_request_penalty,
        lean_low_speed_penalty=lean_low_speed_penalty,
        lean_low_speed_penalty_max_speed_kph=lean_low_speed_penalty_max_speed_kph,
    )
