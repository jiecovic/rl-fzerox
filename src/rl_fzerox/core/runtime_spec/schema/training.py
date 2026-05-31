# src/rl_fzerox/core/runtime_spec/schema/training.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from rl_fzerox.core.domain.training_algorithms import (
    TRAINING_ALGORITHMS,
    TrainAlgorithmName,
)

ResumeArtifact = Literal["latest", "best", "final"]
ResumeMode = Literal["weights_only", "full_model"]


class StateFeatureDropoutGroupConfig(BaseModel):
    """Episode-scoped state-feature dropout for one feature or grouped bundle."""

    model_config = ConfigDict(extra="forbid")

    feature_names: tuple[str, ...]
    dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("feature_names")
    @classmethod
    def _validate_feature_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("train.state_feature_dropout_groups[].feature_names must not be empty")
        if len(set(value)) != len(value):
            raise ValueError(
                "train.state_feature_dropout_groups[].feature_names must not contain duplicates"
            )
        return value


class TrainActorRegularizationConfig(BaseModel):
    """Optional policy-side actor regularization terms."""

    model_config = ConfigDict(extra="forbid")

    grounded_pitch_neutral_loss_weight: NonNegativeFloat = 0.0

    def requires_auxiliary_targets(self) -> bool:
        return self.grounded_pitch_neutral_loss_weight > 0.0


class TrainConfig(BaseModel):
    """Training settings for the current run."""

    model_config = ConfigDict(extra="forbid")

    algorithm: TrainAlgorithmName = TRAINING_ALGORITHMS.default
    vec_env: Literal["dummy", "subproc"] = "dummy"
    num_envs: PositiveInt = 1
    total_timesteps: PositiveInt = 1_000_000
    n_steps: PositiveInt = 1_024
    n_epochs: PositiveInt = 10
    batch_size: PositiveInt = 256
    learning_rate: PositiveFloat = 3e-4
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, gt=0.0, le=1.0)
    clip_range: PositiveFloat = 0.2
    clip_range_vf: PositiveFloat | None = None
    ent_coef: NonNegativeFloat = 0.01
    vf_coef: PositiveFloat = 0.5
    max_grad_norm: PositiveFloat = 0.5
    normalize_advantage: bool = True
    target_kl: PositiveFloat | None = None
    entropy_group_weights: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    actor_regularization: TrainActorRegularizationConfig = Field(
        default_factory=TrainActorRegularizationConfig
    )
    stats_window_size: PositiveInt = 100
    checkpoint_every_rollouts: PositiveInt | None = None
    save_latest_checkpoint: bool = True
    save_best_checkpoint: bool = True
    save_recent_checkpoints: bool = False
    recent_checkpoint_limit: PositiveInt | None = 5
    state_feature_dropout_groups: tuple[StateFeatureDropoutGroupConfig, ...] = ()
    verbose: int = Field(default=0, ge=0, le=2)
    device: str = "auto"
    save_freq: PositiveInt = 1_000
    output_root: Path = Path("local/runs")
    run_name: str = "ppo_cnn"
    tensorboard_step_offset: NonNegativeInt = 0
    explicit_run_dir: Path | None = None
    continue_run_dir: Path | None = None
    resume_run_dir: Path | None = None
    resume_source_algorithm: TrainAlgorithmName | None = None
    resume_source_auxiliary_state_enabled: bool | None = None
    resume_source_auxiliary_state_head_arch: tuple[PositiveInt, ...] = ()
    resume_artifact: ResumeArtifact = "latest"
    resume_mode: ResumeMode = "weights_only"

    @model_validator(mode="after")
    def _validate_algorithm_specific_values(self) -> TrainConfig:
        if self.resume_run_dir is None and self.resume_mode == "full_model":
            raise ValueError("train.resume_mode=full_model requires train.resume_run_dir")
        if (
            self.explicit_run_dir is not None
            and self.continue_run_dir is not None
            and self.explicit_run_dir != self.continue_run_dir
        ):
            raise ValueError(
                "train.explicit_run_dir must match train.continue_run_dir when both are set"
            )
        if self.continue_run_dir is not None:
            if self.resume_run_dir is None:
                raise ValueError("train.continue_run_dir requires train.resume_run_dir")
            if self.resume_run_dir != self.continue_run_dir:
                raise ValueError(
                    "train.continue_run_dir must match train.resume_run_dir "
                    "for in-place continuation"
                )
            if self.resume_mode != "full_model":
                raise ValueError("train.continue_run_dir requires train.resume_mode=full_model")
        group_keys = [group.feature_names for group in self.state_feature_dropout_groups]
        if len(set(group_keys)) != len(group_keys):
            raise ValueError(
                "train.state_feature_dropout_groups must not contain duplicate feature groups"
            )
        return self
