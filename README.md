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
pip install -e .[dev,watch]
```

## Local config

Put your local paths in `conf/local/watch.local.yaml`:

```yaml
seed: 123

emulator:
  core_path: /abs/path/to/mupen64plus_next_libretro.so
  rom_path: /abs/path/to/fzerox.n64
```

The root `seed` is the master seed for Python/env randomness. Emulator resets restore a deterministic baseline savestate.

## Commands

Probe the core:

```bash
python -m rl_fzerox.apps.probe /abs/path/to/mupen64plus_next_libretro.so
```

Headless smoke test:

```bash
python -m rl_fzerox.apps.smoke /abs/path/to/mupen64plus_next_libretro.so /abs/path/to/fzerox.n64 --frames 60
```

Watch the game:

```bash
python -m rl_fzerox.apps.watch --config conf/local/watch.local.yaml
```

Watch controls:

- `P`: pause / resume
- `N`: advance one emulator frame while paused

## Quality

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
ruff check .
pyright
pytest
```
