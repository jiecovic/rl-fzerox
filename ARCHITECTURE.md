# Architecture

`rl-fzerox` is a small RL-oriented emulator stack for F-Zero X.

## Layers

- `src/rl_fzerox/apps/`
  Thin CLI entrypoints such as `probe`, `smoke`, and `watch`.

- `src/rl_fzerox/core/config/`
  Hydra composition plus Pydantic validation. Repo-owned configs resolve
  relative paths from the repo root; external configs resolve from their own
  directory.

- `src/rl_fzerox/core/envs/`
  The Gymnasium-facing environment layer. Right now it is intentionally slim:
  raw emulator frames, fixed frame stepping, no real action semantics yet.

- `src/rl_fzerox/core/emulator/`
  Python wrapper around the native emulator host. It exposes reset, frame
  stepping, rendering, and lightweight frame metadata.

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

Not implemented yet:

- Controller/action mapping
- Reward shaping
- Training CLI and policy inference
