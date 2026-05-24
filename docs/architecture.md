# Architecture

`rl-fzerox` is split into three layers:

1. **Experiment management** owns editable run specs, lineages, launch state,
   metrics, and local artifacts.
2. **RL runtime** owns Gymnasium envs, action decoding, observations, rewards,
   training, inference, and watch playback.
3. **Native emulator boundary** owns libretro/Mupen integration, frame stepping,
   savestates, telemetry extraction, and image observation rendering.

## Runtime Flow

```text
run manager UI
  -> FastAPI / SQLite manager store
  -> manager run spec
  -> runtime_spec projection
  -> baseline materialization
  -> Gym env + native emulator
  -> SB3/sb3x training or watch playback
```

The manager run spec is the editable source of truth. `runtime_spec` configs are
validated launch/watch snapshots derived from it.

## Main Packages

- `src/rl_fzerox/apps/run_manager/`
  Local FastAPI backend, worker launcher, and Vite frontend.

- `src/rl_fzerox/core/manager/`
  Canonical run specs, SQLite storage, projections into runtime configs, run
  registry operations, lineages, metrics, and artifact bookkeeping.

- `src/rl_fzerox/core/runtime_spec/`
  Lower-level Pydantic configs consumed by training, watch, and saved manifests.

- `src/rl_fzerox/core/training/runs/`
  Run paths, saved config IO, race-start helpers, and baseline materialization.

- `src/rl_fzerox/core/envs/`
  Gymnasium environment, action adapters, control masks, state/image
  observations, reward shaping, resets, stepping, and info assembly.

- `src/rl_fzerox/core/policy/` and `src/rl_fzerox/core/training/`
  Feature extractors, policy wiring, SB3/sb3x model construction, callbacks,
  warm starts, checkpoint metadata, and inference loading.

- `src/rl_fzerox/ui/watch/`
  Pygame playback UI for managed runs and saved run configs.

- `src/fzerox_emulator/`
  Thin Python shim over the compiled PyO3 module.

- `rust/`
  Native libretro host and emulator-hot code.

## Ownership Boundaries

- Rust owns emulator-hot work: stepping, frame capture, native observation
  rendering, frame stacking, savestates, and raw telemetry extraction.
- Python env code owns RL semantics: action decoding, masks, rewards,
  truncation/termination, and Gym-compatible observations/info.
- Manager code owns experiment editing and persistent run state.
- Training/watch code owns model lifecycle and human-facing playback.
- `local/` is the ignored machine-local workspace. It stores user-provided ROMs
  and emulator cores plus SQLite files, generated baselines, TensorBoard views,
  and run artifacts.
