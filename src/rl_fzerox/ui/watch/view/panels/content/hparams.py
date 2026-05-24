# src/rl_fzerox/ui/watch/view/panels/content/hparams.py
from __future__ import annotations

from rl_fzerox.core.runtime_spec.schema import PolicyConfig, TrainConfig
from rl_fzerox.ui.watch.view.panels.core.lines import panel_line as _panel_line
from rl_fzerox.ui.watch.view.screen.theme import PALETTE
from rl_fzerox.ui.watch.view.screen.types import PanelSection


def training_hparam_sections(
    *,
    train_config: TrainConfig | None,
    policy_config: PolicyConfig | None,
) -> list[PanelSection]:
    """Build the watch tab that summarizes the loaded run's training hparams."""

    if train_config is None:
        return [
            PanelSection(
                title="Training",
                lines=[
                    _panel_line(
                        "Source",
                        "no train run metadata",
                        PALETTE.text_muted,
                    ),
                ],
            )
        ]

    return [
        PanelSection(
            title="Training",
            lines=[
                _panel_line("Algorithm", str(train_config.algorithm), PALETTE.text_primary),
                _panel_line("Device", str(train_config.device), PALETTE.text_primary),
                _panel_line("Vec env", str(train_config.vec_env), PALETTE.text_primary),
                _panel_line("Num envs", str(train_config.num_envs), PALETTE.text_primary),
                _panel_line(
                    "Target steps",
                    _format_int(train_config.total_timesteps),
                    PALETTE.text_primary,
                ),
                _panel_line("Batch", str(train_config.batch_size), PALETTE.text_primary),
                _panel_line("LR", _format_lr(train_config.learning_rate), PALETTE.text_primary),
                _panel_line("Gamma", _format_float(train_config.gamma), PALETTE.text_primary),
                _panel_line("Ent coef", str(train_config.ent_coef), PALETTE.text_primary),
            ],
        ),
        _algorithm_section(train_config),
        _policy_section(policy_config),
    ]


def _algorithm_section(train_config: TrainConfig) -> PanelSection:
    return PanelSection(
        title="PPO HParams",
        lines=[
            _panel_line("N steps", str(train_config.n_steps), PALETTE.text_primary),
            _panel_line("Epochs", str(train_config.n_epochs), PALETTE.text_primary),
            _panel_line("GAE lambda", _format_float(train_config.gae_lambda), PALETTE.text_primary),
            _panel_line("Clip", _format_float(train_config.clip_range), PALETTE.text_primary),
            _panel_line("VF coef", _format_float(train_config.vf_coef), PALETTE.text_primary),
            _panel_line(
                "Max grad",
                _format_float(train_config.max_grad_norm),
                PALETTE.text_primary,
            ),
        ],
    )


def _policy_section(policy_config: PolicyConfig | None) -> PanelSection:
    if policy_config is None:
        return PanelSection(
            title="Policy",
            lines=[_panel_line("Source", "no policy metadata", PALETTE.text_muted)],
        )

    extractor = policy_config.extractor
    recurrent = policy_config.recurrent
    return PanelSection(
        title="Policy",
        lines=[
            _panel_line("Conv", str(extractor.conv_profile), PALETTE.text_primary),
            _panel_line("Features", str(extractor.features_dim), PALETTE.text_primary),
            _panel_line(
                "State dim",
                str(extractor.state_features_dim),
                PALETTE.text_primary,
            ),
            _panel_line("Activation", str(policy_config.activation), PALETTE.text_primary),
            _panel_line(
                "Pi net",
                _format_int_tuple(policy_config.net_arch.pi),
                PALETTE.text_primary,
            ),
            _panel_line(
                "V/Q net",
                _format_int_tuple(policy_config.net_arch.vf),
                PALETTE.text_primary,
            ),
            _panel_line(
                "Recurrent",
                "on" if recurrent.enabled else "off",
                PALETTE.text_primary,
            ),
        ],
    )


def _format_int(value: int) -> str:
    return f"{value:,}"


def _format_float(value: float) -> str:
    return f"{value:g}"


def _format_lr(value: float) -> str:
    return f"{value:.0e}"


def _format_int_tuple(values: tuple[int, ...]) -> str:
    return f"[{', '.join(str(value) for value in values)}]" if values else "[]"
