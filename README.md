# rl-fzerox

Experimental project to train a neural network to play F-Zero X with reinforcement learning.

The emulator boundary is handled by a custom Rust libretro host exposed to Python with `pyo3`. Python owns the Gymnasium env, config, telemetry tooling, and watch UI.

## Requirements

- Python `3.11+`
- Rust toolchain via `rustup` / `cargo`
- `Mupen64Plus-Next` libretro core shared library (`.so`)
- An extracted F-Zero X US ROM that you obtained yourself

The repo does not include the emulator core or ROM files. You need to download or build those yourself. ROMs are not committed for copyright reasons.

The reverse-engineered telemetry/RAM layout currently targets the US ROM build.
Other regional revisions are not supported by the current memory offsets.

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

For PPO training and policy watch mode, install the extra dependencies too:

```bash
python -m pip install -e .[train]
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
  fps: auto
```

Relative emulator paths in repo configs are resolved relative to the project root. `emulator.runtime_dir` is the optional root for generated emulator state such as `mupen64plus.ini`; the native host creates it when needed. `emulator.baseline_state_path` points to the saved race-start baseline used for fast deterministic resets once you create it. The root `seed` is the master seed for Python/env randomness.

`watch` uses its configured `emulator.runtime_dir` directly. `train` does not reuse that global runtime path anymore: each training run gets its own writable runtime root under the run directory, and each env instance gets its own subdirectory under that root.

`env.reset_to_race: true` runs a deterministic first-race bootstrap from the boot baseline. It fast-forwards through the default menu path into the first Mute City grid. Press `K` in `watch` once to save `local/states/first-race.state`; after that, both `watch` and `train` reset from that dedicated race-start baseline instead of replaying the menu path. Without a custom baseline, terminal episodes try to continue into the next race on the same emulator session before falling back to a full reset.

`watch.episodes` defaults to `null`, so the watch app keeps resetting until you quit it. `watch.fps: auto` means "run at the natural control-loop cadence", i.e. `60 / action_repeat`, and any explicit numeric `watch.fps` is capped to that same maximum.

The current policy action space is a `MultiDiscrete([5, 2])` adapter:

- 5 steering buckets from hard-left to hard-right
- 2 drive modes: `coast` and `throttle`

Brake and boost are intentionally not part of the first policy surface yet.
They need a cleaner verified mapping from libretro controls to F-Zero X inputs
before they belong in training code.

The current observation pipeline keeps the full game view, aspect-corrects the
raw `640x240` emulator framebuffer to `4:3`, downsamples it to `160x120 RGB`,
and returns a channels-last stack of the last 4 observations.

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

If you want to reuse a saved baseline state from the local watch flow, pass the same runtime directory as well:

```bash
python -m rl_fzerox.apps.smoke /abs/path/to/mupen64plus_next_libretro.so /abs/path/to/fzerox.n64 --runtime-dir /abs/path/to/runtime --baseline-state /abs/path/to/first-race.state --frames 0
```

Watch the game:

```bash
python -m rl_fzerox.apps.watch --config conf/local/watch.local.yaml
```

Train PPO:

```bash
python -m rl_fzerox.apps.train --config conf/local/train.local.yaml
```

For multi-env training, switch `train.vec_env` to `subproc` and raise `train.num_envs`. Each worker env gets an isolated runtime dir under the run folder, for example `local/runs/ppo_cnn_0007/runtime/env_000`.

Watch the latest saved policy artifact from one training run:

```bash
python -m rl_fzerox.apps.watch --config conf/local/watch.local.yaml --run-dir local/runs/ppo_cnn_0001
```

Watch the current best saved policy instead:

```bash
python -m rl_fzerox.apps.watch --config conf/local/watch.local.yaml --run-dir local/runs/ppo_cnn_0001 --artifact best
```

Read a live telemetry snapshot from emulated RAM:

```bash
python -m rl_fzerox.apps.telemetry --config conf/local/watch.local.yaml
```

Watch controls:

- `P`: pause / resume
- `N`: advance one emulator frame while paused
- `Left / Right`: steering and left/right D-pad
- `Up / Down`: D-pad
- `X`: A
- `Z`: B
- `Enter`: Start
- `Backspace`: Select
- `K`: promote the current state to the reset baseline and save it when `emulator.baseline_state_path` is set

The watch UI shows the raw game view, a preview of the latest processed policy
observation, and live telemetry plus the current step reward and total episode
return.

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
