# rl-fzerox

Experimental reinforcement-learning playground for training agents to race in
F-Zero X through an N64/libretro emulator.

It combines:

- a Rust libretro host exposed to Python with `pyo3`
- a Gymnasium environment around F-Zero X
- game-state telemetry for rewards and action masks
- a pygame watch UI
- PPO-family training pipelines and an experimental SAC pipeline

ROMs and emulator cores are not included. The current setup targets the US
F-Zero X ROM.

## What Works

- Train agents on emulator frames plus game state.
- Watch trained policies drive in the live game.
- Resume from saved checkpoints and compare runs.
- Experiment with discrete, continuous, recurrent, and masked action spaces.
- Start from a saved race baseline for faster iteration.

## Algorithms

The main training path is maskable PPO.

Implemented PPO-family variants:

- `maskable_ppo`
- `maskable_recurrent_ppo`
- `maskable_hybrid_action_ppo`
- `maskable_hybrid_recurrent_ppo`

The recurrent and hybrid PPO variants come from the companion
[`sb3x-extensions`](https://github.com/jiecovic/sb3x-extensions) package.

SAC is included as a separate continuous-control experiment.

## Requirements

- Python 3.11+
- Rust toolchain with `cargo`
- Mupen64Plus-Next libretro core shared library (`.so`)
- your own F-Zero X US ROM
- `sb3x-extensions` for recurrent and hybrid PPO variants

Mupen64Plus-Next libretro links: [docs](https://docs.libretro.com/library/mupen64plus/),
[source](https://github.com/libretro/mupen64plus-libretro-nx).

The RAM offsets currently target the US ROM build only.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev,watch,train]
just native
cp conf/local/watch.local.example.yaml conf/local/watch.local.yaml
```

Also install the companion `sb3x` package from
https://github.com/jiecovic/sb3x-extensions.

Put local runtime files under `local/`:

```text
local/libretro/mupen64plus_next_libretro.so
local/roms/F-Zero X (USA).n64
```

Training presets live under `conf/presets/`. Private machine-specific overrides
can live under `conf/local/`; examples are included there.

## Basic Commands

Watch manually or with a policy:

```bash
python -m rl_fzerox.apps.watch --config conf/local/watch.local.yaml
python -m rl_fzerox.apps.watch --config conf/local/watch.local.yaml --run-dir checkpoints/mute-city-3steer-recurrent-ppo-v1 --artifact best
```

Train recurrent PPO:

```bash
python -m rl_fzerox.apps.train --config conf/presets/train.maskable_recurrent_ppo.yaml
```

Train hybrid recurrent PPO:

```bash
python -m rl_fzerox.apps.train --config conf/presets/train.maskable_hybrid_recurrent_ppo.yaml
```

Train the SAC experiment:

```bash
python -m rl_fzerox.apps.train --config conf/presets/train.sac.yaml
```

## Checkpoints

A sample checkpoint is included:

```text
checkpoints/mute-city-3steer-recurrent-ppo-v1
```

The checkpoint includes the model artifacts, run config, and baseline state.
Edit the config paths if your ROM or libretro core live somewhere else.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local quality checks and development
notes.
