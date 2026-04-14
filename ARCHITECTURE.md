# Architecture

`rl-fzerox` is an RL-oriented emulator stack for F-Zero X. Rust owns the
libretro/emulator boundary and the hot frame/observation path; Python owns
configuration, Gymnasium integration, rewards, training orchestration, and the
watch UI.

## Runtime Flow

The main control flow is:

1. A CLI in `src/rl_fzerox/apps/` loads a Hydra config and validates it with
   Pydantic.
2. `src/rl_fzerox/core/envs/` builds a Gymnasium env around one native emulator
   instance.
3. The env decodes policy actions into controller state, steps the native host,
   receives telemetry and observations, then computes rewards, masks, truncation
   state, and Gym info.
4. Training code in `src/rl_fzerox/core/training/` wraps the env in SB3 vector
   envs, builds the configured algorithm, and writes run artifacts.
5. Watch code in `src/rl_fzerox/ui/watch/` loads a saved policy and run config,
   applies watch/CLI overrides, and renders live playback.

## Native Emulator

- `rust/`
  Native libretro host exposed to Python with `pyo3`. It loads
  Mupen64Plus-Next, manages libretro callbacks, applies controller input,
  steps frames, reads F-Zero X telemetry, captures frames, and restores
  savestates.

- `src/fzerox_emulator/`
  Thin Python package around the native module. It provides the Python-facing
  `Emulator` wrapper, controller dataclasses, video helpers, and type stubs.

Rust also owns native observation presets, crop/aspect geometry, frame stacking,
and repeated inner-frame execution for one outer env step. Python should not
duplicate that geometry logic.

## Core Python Modules

- `src/rl_fzerox/core/config/`
  Hydra composition, Pydantic schemas, and path resolution. Checked-in configs
  can use stable repo-root paths such as `local/...` and `checkpoints/...`;
  ad hoc external configs still resolve relative paths from their own file.

- `src/rl_fzerox/core/domain/`
  Small string-backed domain constants such as training algorithm names, action
  adapter names, camera settings, hybrid action keys, and shoulder-slide modes.

- `src/rl_fzerox/core/envs/`
  Gymnasium env, telemetry conversion, observations, rewards, and the runtime
  engine. The `engine/` submodule owns reset/step orchestration, camera sync,
  control-state history, masks, and info assembly.

- `src/rl_fzerox/core/envs/actions/`
  Action adapters from policy action spaces to controller state. Current
  adapters cover discrete steering, continuous steering/drive experiments, and
  hybrid continuous-plus-discrete action spaces. The shoulder inputs are the Z/R
  lean/slide buttons; legacy names containing `drift` remain only as checkpoint
  compatibility shims.

- `src/rl_fzerox/core/envs/rewards/race_v2/`
  Reward shaping for progress, milestones, lap/finish events, damage, energy,
  boost-pad usage, wrong-way driving, stuck detection, and progress-frontier
  stalls.

- `src/rl_fzerox/core/policy/`
  SB3 feature extractors. Image-only observations use a CNN extractor;
  `image_state` observations use a CNN image branch plus a small scalar-state
  branch and optional fusion MLP.

- `src/rl_fzerox/core/training/`
  Training run orchestration, artifact paths, checkpoint metadata, curriculum
  callbacks, model construction, warm-start loading, and inference loading.

## Algorithms

The PPO path is intentionally maskable-first. `auto` resolves to
`maskable_ppo` for older configs, but new configs should name the algorithm
explicitly.

Supported PPO-family algorithms:

- `maskable_ppo` from `sb3_contrib`
- `maskable_recurrent_ppo` from `sb3x`
- `maskable_hybrid_action_ppo` from `sb3x`
- `maskable_hybrid_recurrent_ppo` from `sb3x`

SAC remains available as a separate continuous-control experiment. It uses SB3
SAC and continuous action adapters; it does not support action masks or mask
curriculum stages.

## Config Layout

- `conf/train_base.yaml`
  Shared train defaults.

- `conf/train_ppo_base.yaml`
  PPO-family defaults.

- `conf/train_sac_base.yaml`
  SAC defaults.

- `conf/presets/`
  Checked-in starter training presets. These are safe public examples and
  should not contain machine-specific run history.

- `conf/local/`
  Local private overrides. Only `*.example.yaml` files are tracked; real local
  configs are ignored.

- `conf/watch.yaml`
  Default watch config.

## Artifacts

- `local/`
  Ignored workspace for ROMs, libretro cores, generated emulator runtime files,
  and normal training runs.

- `checkpoints/`
  Curated checkpoint drops that are intended to be shareable. Large model zips
  and baseline savestates are tracked through Git LFS.

Each training run stores a resolved `train_config.yaml`, model/policy artifacts,
policy metadata sidecars, TensorBoard logs, and a run-local baseline/runtime
layout. Watch mode can load those saved configs so policy playback uses the same
env/action/observation setup that produced the checkpoint.

## Data Ownership

- Rust owns emulator stepping, native frame rendering, frame stacking, and raw
  F-Zero X telemetry extraction.
- Python env code owns action decoding, game-specific masks, reward shaping,
  truncation/termination policy, and Gym-compatible info.
- Training code owns SB3/sb3x model construction, rollout collection, callbacks,
  checkpoint metadata, and warm starts.
- Watch code owns human-facing rendering, policy hot reload, FPS control, and
  manual baseline capture.
