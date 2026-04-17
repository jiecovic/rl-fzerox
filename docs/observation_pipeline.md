# Observation Pipeline

The repo now uses native observation presets for policy inputs:

- `native_crop_v1`
- `native_crop_v2`
- `native_crop_v3`
- `native_crop_v4`

Python no longer configures observation width, height, crop margins, or color
mode directly. Rust owns the per-frame geometry so the env, watch UI, and any
future observation presets stay consistent.

## Raw Frame

The libretro core currently produces a raw RGB framebuffer at:

- `640 x 240`

That raw buffer includes dark overscan margins.

## Native Crop

`native_crop_v1` first crops the raw frame by:

- top: `16`
- bottom: `16`
- left: `24`
- right: `24`

That yields a cropped native frame of:

- `592 x 208`

## Policy Observation

For the policy path, all four presets first apply the same aspect correction used
by the human watch display, then downscale to:

- `native_crop_v1`: `116 x 84 x 3`
- `native_crop_v2`: `124 x 92 x 3`
- `native_crop_v3`: `164 x 116 x 3`
- `native_crop_v4`: `130 x 98 x 3`

All four give the policy an approximately `4:3` view instead of the earlier
native-width-squashed observation. `native_crop_v1` and `native_crop_v2` use
an exact NatureCNN-style conv stack; `native_crop_v3` is the current larger
default and pairs with the older 4-layer `64,64,128,128` extractor shape.
`native_crop_v4` is a compact-deep experiment sized for a clean `4 x 6`
final CNN grid with the `64,64,128,128` stride-2 extractor.

With `frame_stack: 4`, the env observation space becomes:

- `native_crop_v1`: `84 x 116 x 12`
- `native_crop_v2`: `92 x 124 x 12`
- `native_crop_v3`: `116 x 164 x 12`
- `native_crop_v4`: `98 x 130 x 12`

## Observation Modes

The screen-only path remains available with:

- `env.observation.mode: image`

The mixed image plus scalar state path is:

- `env.observation.mode: image_state`

`image_state` returns a Gym `Dict` observation with:

- `image`: the same stacked RGB image used by screen-only mode
- `state`: a configurable scalar `float32` vector appended by the policy
  feature extractor

`env.observation.state_profile` selects the base telemetry profile. The default
profile keeps the original richer scalar set:

- `speed_norm`: `speed_kph / 1500`, clamped to `[0, 2]`
- `energy_frac`: `energy / max_energy`, clamped to `[0, 1]`
- `reverse_active`: `1` when the game reverse timer is above zero, else `0`
- `airborne`: `1` when the game state flag is active, else `0`
- `can_boost`: `1` when the game state flag is active, else `0`
- `boost_active`: `1` when `boost_timer > 0`, else `0`
- `left_lean_held`: `1` when the previous env step held the left lean
  input, else `0`
- `right_lean_held`: `1` when the previous env step held the right lean
  input, else `0`
- `left_press_age_norm`: frames since the last left-lean press edge,
  normalized and clipped to the game's 15-frame double-tap window
- `right_press_age_norm`: frames since the last right-lean press edge,
  normalized and clipped to the game's 15-frame double-tap window
- `recent_boost_pressure`: fraction of the recent 120 internal frames where the
  current action requested boost

`state_profile: race_core` is the current compact recurrent-training baseline.
It keeps only:

- `speed_norm`
- `energy_frac`
- `reverse_active`
- `airborne`
- `can_boost`
- `boost_active`

Previous controls are configured separately from the base telemetry profile.
Set `env.observation.action_history_len: null` to disable action history. Set it
to a positive integer to append fixed-width `prev_<control>_<age>` features for
the configured `env.observation.action_history_controls`.

For example, the active compact recurrent setup uses:

```yaml
env:
  observation:
    state_profile: race_core
    action_history_len: 2
    action_history_controls: [steer, gas, boost, lean]
```

That produces `6 + 2 * 4 = 14` scalar state features: six telemetry features
plus the previous two values for steer, gas, boost, and lean.

Training uses SB3 `CnnPolicy` for `image` mode and `MultiInputPolicy` for
`image_state` mode. The custom mixed extractor keeps the same CNN image branch
and concatenates a small MLP state branch before the PPO policy/value heads.

## Human Watch Display

The watch display is a separate native render target. It uses the same crop,
but applies human-facing display aspect correction before presenting the large
image.

Current display size after crop and aspect correction:

- `592 x 444 x 3`

This keeps the viewer easy to read without forcing the policy to train on a
stretched image.

## Why Rust Owns The Geometry

Rust now owns:

- crop
- preset name resolution
- resolved observation shape
- native observation rendering
- native frame stacking
- native watch display rendering

Python now owns:

- preset selection in config
- observation mode selection in config
- PPO/CNN wiring
- watch layout and controls

This keeps one source of truth for frame geometry and avoids the older split
where Python and native code could disagree about crop and resize behavior.

## Env-Step Ownership

The hot training step loop is now split like this:

- Rust owns:
  - repeated internal frame execution for one outer env step
  - native observation rendering and native frame stacking
  - step-local summary aggregation across repeated frames
- Python owns:
  - action decoding into one held controller state
  - reward shaping from the returned step summary
  - timeout / stuck / wrong-way limits from the returned step summary
  - Gym info assembly

So `env.step()` now crosses the Rust boundary once per outer RL step, rather
than manually looping repeated inner frames in Python.
