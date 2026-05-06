# src/rl_fzerox/core/envs/actions/buttons.py
from __future__ import annotations

from fzerox_emulator import JOYPAD_BUTTONS, joypad_mask

# Mupen64Plus-Next's standard RetroPad mapping exposes the in-game N64 A/C-Down
# buttons on RetroPad B/A. F-Zero X names those controls Accelerate/Air Brake.
ACCELERATE_MASK = joypad_mask(JOYPAD_BUTTONS.b)
AIR_BRAKE_MASK = joypad_mask(JOYPAD_BUTTONS.a)

# Stable aliases used throughout the env/watch surface.
THROTTLE_MASK = ACCELERATE_MASK
BRAKE_MASK = AIR_BRAKE_MASK

# N64 B/Z/R map to RetroPad Y/L2/R and correspond to boost and lean inputs.
BOOST_MASK = joypad_mask(JOYPAD_BUTTONS.y)
LEAN_LEFT_MASK = joypad_mask(JOYPAD_BUTTONS.left_trigger)
LEAN_RIGHT_MASK = joypad_mask(JOYPAD_BUTTONS.right_shoulder)
