from fzerox_emulator import RaceControlState
from rl_fzerox.core.envs.actions import RACE_CONTROL_MASKS
from rl_fzerox.core.envs.engine.controls.action_history import ActionHistoryBuffer


def test_action_history_records_split_lean_history_separately() -> None:
    history = ActionHistoryBuffer(
        length=1,
        controls=("lean",),
        split_lean_history=True,
    )

    history.record(
        RaceControlState.from_mask(
            RACE_CONTROL_MASKS.lean_left | RACE_CONTROL_MASKS.lean_right
        ),
        gas_level=None,
    )

    assert history.fields() == {
        "prev_lean_left_1": 1.0,
        "prev_lean_right_1": 1.0,
    }
