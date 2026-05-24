# tests/core/emulator/test_native_input.py
from fzerox_emulator import JOYPAD_BUTTONS, joypad_mask


def test_native_joypad_mask_sets_expected_bits() -> None:
    mask = joypad_mask(JOYPAD_BUTTONS.up, JOYPAD_BUTTONS.a, JOYPAD_BUTTONS.start)

    assert mask == (
        (1 << JOYPAD_BUTTONS.up) | (1 << JOYPAD_BUTTONS.a) | (1 << JOYPAD_BUTTONS.start)
    )
