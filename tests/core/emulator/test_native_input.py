# tests/core/emulator/test_native_input.py
from fzerox_emulator import JOYPAD_A, JOYPAD_START, JOYPAD_UP, joypad_mask


def test_native_joypad_mask_sets_expected_bits() -> None:
    mask = joypad_mask(JOYPAD_UP, JOYPAD_A, JOYPAD_START)

    assert mask == ((1 << JOYPAD_UP) | (1 << JOYPAD_A) | (1 << JOYPAD_START))
