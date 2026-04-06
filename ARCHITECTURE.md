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
  raw emulator frames, fixed frame stepping, and an optional deterministic
  boot-to-race reset path.

- `src/rl_fzerox/core/emulator/`
  Python wrapper around the native emulator host. It exposes reset, frame
  stepping, rendering, lightweight frame metadata, and raw system RAM reads.

- `src/rl_fzerox/core/game/`
  F-Zero X-specific runtime decoding built on top of raw RAM reads. Right now
  this is a small telemetry layer for race mode, timer, speed, energy, lap,
  and player state flags.

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
- Manual watch controls for creating a baseline savestate in-project
- Live RDRAM telemetry for a first set of player/race values

Not implemented yet:

- Controller/action mapping
- Reward shaping
- Training CLI and policy inference
