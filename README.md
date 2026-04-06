# rl-fzerox

Experimental project to train a neural network to play F-Zero X with reinforcement learning.

The emulator boundary is handled by a custom Rust libretro host exposed to Python with `pyo3`. Python owns the Gymnasium env, config, smoke tooling, and watch UI.

## Requirements

- Python `3.11+`
- Rust toolchain via `rustup` / `cargo`
- `Mupen64Plus-Next` libretro core shared library (`.so`)
- An extracted F-Zero X ROM that you obtained yourself

The repo does not include the emulator core or ROM files. You need to download or build those yourself. ROMs are not committed for copyright reasons.

Official `Mupen64Plus-Next` links:

- Docs: https://docs.libretro.com/library/mupen64plus/
- Source: https://github.com/libretro/mupen64plus-libretro-nx

## Setup

```bash
curl https://sh.rustup.rs -sSf | sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev,watch]
```

## Local config

Keep local runtime artifacts such as the libretro core and ROM under `local/`. That directory is ignored by git.

Put your local paths in `conf/local/watch.local.yaml`:

```yaml
defaults:
  - /watch
  - _self_

seed: 123

emulator:
  core_path: local/libretro/mupen64plus_next_libretro.so
  rom_path: local/roms/F-Zero X (USA).n64
  runtime_dir: local/runtime
  baseline_state_path: local/states/first-race.state

env:
  reset_to_race: true

watch:
  fps: 30
```

Relative emulator paths in repo configs are resolved relative to the project root. `emulator.runtime_dir` is the optional root for generated emulator state such as `mupen64plus.ini`; the native host creates it when needed. `emulator.baseline_state_path` is an optional savestate file used as the reset baseline. The root `seed` is the master seed for Python/env randomness.

`env.reset_to_race: true` runs a deterministic first-race bootstrap from the boot baseline. It fast-forwards through the default menu path into the first Mute City grid. Once that path is stable for your setup, press `K` in `watch` to save a dedicated race-start baseline. When `emulator.baseline_state_path` already exists, that saved baseline takes precedence and the scripted bootstrap is skipped.

## Commands

Probe the core:

```bash
python -m rl_fzerox.apps.probe /abs/path/to/mupen64plus_next_libretro.so
```

Headless smoke test:

```bash
python -m rl_fzerox.apps.smoke /abs/path/to/mupen64plus_next_libretro.so /abs/path/to/fzerox.n64 --frames 60
```

Headless smoke test with race bootstrap:

```bash
python -m rl_fzerox.apps.smoke /abs/path/to/mupen64plus_next_libretro.so /abs/path/to/fzerox.n64 --reset-to-race --frames 60
```

Watch the game:

```bash
python -m rl_fzerox.apps.watch --config conf/local/watch.local.yaml
```

Watch controls:

- `P`: pause / resume
- `N`: advance one emulator frame while paused
- `Arrow keys`: D-pad
- `X`: A
- `Z`: B
- `Enter`: Start
- `Backspace`: Select
- `K`: promote the current state to the reset baseline and save it when `emulator.baseline_state_path` is set

## Quality

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo audit
ruff check .
pyright
pytest
```
