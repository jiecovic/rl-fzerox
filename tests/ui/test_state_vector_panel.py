# tests/ui/test_state_vector_panel.py
from rl_fzerox.ui.watch.view.panels.content.state_vector_panel.formatting import (
    format_state_vector_value,
)
from rl_fzerox.ui.watch.view.panels.content.state_vector_panel.model import state_vector_groups


def test_state_vector_groups_keep_unknown_features_visible() -> None:
    groups = state_vector_groups(
        (
            "vehicle_state.speed_norm",
            "custom_scalar",
        )
    )

    assert [(group.title, group.prefix) for group in groups] == [
        ("Vehicle", "vehicle_state."),
        ("Other", None),
    ]


def test_state_vector_formats_obs_scalars_with_four_decimal_places() -> None:
    formatted = format_state_vector_value(
        feature_name="machine_context.engine",
        auxiliary_name=None,
        show_aux_columns=True,
        observation_value=103 / 128,
        reference_value=None,
        prediction=None,
        target=None,
    )

    assert formatted.endswith("|  0.8047")
