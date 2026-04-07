# Architecture

`rl-fzerox` is a small RL-oriented emulator stack for F-Zero X.

## Layers

- `src/rl_fzerox/apps/`
  Thin CLI entrypoints such as `probe`, `smoke`, `telemetry`, and `watch`.

- `src/rl_fzerox/core/config/`
  Hydra composition plus Pydantic validation. Repo-owned configs resolve
  relative paths from the repo root; external configs resolve from their own
  directory.

- `src/rl_fzerox/core/envs/`
  The Gymnasium-facing environment layer. Right now it is intentionally slim:
  modular action adapters, fixed frame stepping, a first telemetry-based reward
  function, deterministic race reset helpers, action adapters, episode limits,
  and a stacked observation pipeline.

  The current policy-side action adapter is a 5-bucket steering plus 2-mode
  coast/throttle `MultiDiscrete` space.

  The current observation pipeline aspect-corrects the raw framebuffer,
  downsamples it to `160x120 RGB`, and stacks 4 returned observations. The
  stateless image transforms are candidates to move into Rust once the
  observation format is stable; the stack itself should stay tied to env
  reset/step semantics.

- `src/rl_fzerox/core/emulator/`
  Python wrapper around the native emulator host. It exposes reset, frame
  stepping, rendering, lightweight frame metadata, and raw system RAM reads.

- `src/rl_fzerox/core/game/`
  F-Zero X-specific runtime decoding built on top of raw RAM reads. Right now
  this is a small telemetry layer for race mode, timer, speed, energy, lap,
  and player state flags. This decoding currently lives in Python so offsets
  and field semantics are easy to iterate on while reverse engineering is still
  active. Once that field set is stable, the decoding should move into the Rust
  host so the native boundary owns both memory access and structured game-state
  reads.

- `src/rl_fzerox/core/policy.py`
  SB3-facing policy surface shared by training and inference. Right now it
  contains the first custom PPO CNN extractor, PPO model construction, and the
  small inference wrapper used by `watch --run-dir`.

- `src/rl_fzerox/core/training/`
  Training-session orchestration and run artifacts only. This layer owns run
  directories, saved train-config snapshots, and the PPO runner wiring around
  the env and policy modules.

- `rust/`
  Native libretro host implemented in Rust and exposed to Python with `pyo3`.
  This layer loads the `Mupen64Plus-Next` core, loads the ROM, manages the
  libretro callback surface, captures frames, and restores a deterministic
  baseline state on reset.

## Runtime Artifacts

- `local/libretro/`
  Native libretro core shared libraries.

- `local/roms/`
  ROM files.

- `local/runtime/`
  Emulator-generated runtime state such as `mupen64plus.ini`.

`local/` is ignored by git.

## Current Scope

- Working native probe path
- Working headless smoke path
- Working real-time watch path
- Deterministic reset via baseline savestate
- Optional external baseline savestate file for fixed race-start resets
- Optional scripted reset-to-race bootstrap from the boot baseline
- Soft continue-into-next-race reset after terminal episodes
- Manual watch controls for creating a baseline savestate in-project
- First modular controller/action mapping for policy actions
- First modular observation pipeline for policy input
- Live RDRAM telemetry for a first set of player/race values
- First telemetry-based reward shaping from race progress and terminal events
- First PPO training/inference wiring with saved run-config snapshots

Not implemented yet:

- Brake/boost action mapping and richer action spaces
