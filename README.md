# rl-fzerox

Experimental reinforcement-learning playground for training agents to race in
F-Zero X through an N64/libretro emulator.

It combines:

- a Rust libretro host exposed to Python with `pyo3`, supporting the
  Angrylion and GLideN64 renderer paths used by the observation pipeline
- a Gymnasium environment around F-Zero X
- game-state telemetry for rewards, action masks, state-vector observations,
  reset materialization, and the pygame watch UI
- PPO-family training pipelines using SB3-compatible maskable, recurrent, and
  hybrid-action algorithms for continuous plus multi-discrete controls

```text
!! LOCAL RUNTIME ASSETS REQUIRED !!
```

ROMs and emulator cores are not included. The current setup targets the US
F-Zero X ROM, and RAM offsets are only maintained for that build.

## Canonical Workflow

The canonical authoring surface is the local run manager:

- the frontend edits manager-owned run specs
- SQLite stores drafts, templates, and managed runs
- training/watch runtime configs are derived from that manager-owned spec
- saved `local/runs/**/train_config.yaml` files are readonly snapshots, not the
  source of truth

The managed training path currently targets maskable hybrid recurrent PPO. The
runtime layer still understands a small PPO family for saved-manifest loading
and lower-level resume/inference paths, but YAML/Hydra config authoring is no
longer a supported primary workflow.

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
```

Also install the companion `sb3x` package from
https://github.com/jiecovic/sb3x-extensions.

Put local runtime files under `local/`:

```text
local/libretro/mupen64plus_next_libretro.so
local/roms/F-Zero X (USA).n64
```

Managed configs live in the run-manager SQLite store. Local runtime artifacts
and saved manifests live under `local/`.

## Basic Commands

Launch the canonical run-manager workflow:

```bash
just run-manager
```

Watch a saved run directory directly:

```bash
python -m rl_fzerox.apps.watch --run-dir local/runs/<run-id>
```

Watch a managed run directly from the manager DB:

```bash
python -m rl_fzerox.apps.watch --managed-run-id <managed-run-id>
```

Resume an existing run in place from its saved manifest:

```bash
python -m rl_fzerox.apps.train --continue-run-dir local/runs/<run-id>
```

## Checkpoints

No model checkpoint is currently tracked. Training and watch runs should use
run-local artifacts under `local/runs/`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local quality checks and development
notes.
