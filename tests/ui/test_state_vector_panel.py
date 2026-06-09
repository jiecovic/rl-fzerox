# tests/ui/test_state_vector_panel.py
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
