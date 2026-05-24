# Observation Pipeline

The observation pipeline is split between Rust and Python.

Rust owns the hot image path: libretro frame capture, source cropping, resize
geometry, native observation rendering, native frame stacking, and native
multi-observation watch batches.

Python owns configuration, Gymnasium spaces, scalar state-vector features,
policy extractor wiring, and watch/run-manager presentation.

## Image Geometry

The shared image vocabulary lives in
`src/rl_fzerox/core/domain/observation_image.py`.

Current stable presets are:

- `crop_72x96`: `72 x 96`
- `crop_84x84`: `84 x 84`

`crop_84x84` is the default preset. `crop_72x96` is the current rectangular
IMPALA-oriented preset used by several recurrent experiments.

The run spec also supports:

- `custom`: fixed height/width inside bounded limits
- `source_crop`: renderer-native cropped frame without target downsampling

Current source-crop sizes are:

- `angrylion`: `208 x 592`
- `gliden64`: `208 x 296`

## Observation Modes

The image-only path returns a stacked image observation.

The image-plus-state path returns a Gym `Dict` observation:

- `image`: stacked rendered frames
- `state`: configurable scalar features derived from telemetry, control
  history, and selected state components

State components and control-history fields are managed by the runtime spec and
run-manager configurator. The backend validates requested state feature names so
saved configs cannot drift away from the active action/observation schema.

## Frame Stacking

Frame stacking is native-side for image observations. Python config chooses the
stack count and stack mode, while Rust builds the final image tensor for the
outer RL step.

For RGB stacks, channel count is `3 * frame_stack`. For grayscale or
luma/chroma modes, the channel count follows the selected native stack mode.

## Env-Step Ownership

One outer `env.step()` crosses into Rust once for repeated internal frames.

Rust owns:

- repeated internal frame execution
- held controller application
- native observation rendering
- frame stacking
- step-summary aggregation

Python owns:

- action decoding into controller state
- reward shaping
- action masks
- truncation and termination policy
- Gym-compatible info assembly

This keeps the hot loop small while leaving experiment-facing reward and policy
logic in Python.
