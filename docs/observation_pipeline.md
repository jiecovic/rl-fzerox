# Observation Pipeline

The repo now uses native observation presets for policy inputs:

- `native_crop_v1`
- `native_crop_v2`
- `native_crop_v3`

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

For the policy path, all three presets first apply the same aspect correction used
by the human watch display, then downscale to:

- `native_crop_v1`: `116 x 84 x 3`
- `native_crop_v2`: `124 x 92 x 3`
- `native_crop_v3`: `164 x 116 x 3`

Both give the policy an approximately `4:3` view instead of the earlier
native-width-squashed observation. `native_crop_v1` and `native_crop_v2` use
an exact NatureCNN-style conv stack; `native_crop_v3` is the larger default
and pairs with the older 4-layer `32,64,64,128` extractor shape.

With `frame_stack: 4`, the env observation space becomes:

- `native_crop_v1`: `84 x 116 x 12`
- `native_crop_v2`: `92 x 124 x 12`
- `native_crop_v3`: `116 x 164 x 12`

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
