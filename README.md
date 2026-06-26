# rl-fzerox

Experimental deep reinforcement-learning project for training agents to race in
F-Zero X through a libretro N64 emulator.

This repository is still under construction. The teaser shows the current
checkpoint in action; downloadable checkpoint bundles are available through the
run manager.

Teaser video: the current policy can unlock everything by completing all cups on
Master, the hardest F-Zero X difficulty. This run shows one example: winning the
full Joker Cup on Master. The full playlist shows every cup on Master with one
policy.

[![rl-fzerox teaser video - click to watch on YouTube](https://img.youtube.com/vi/r_4lAayCBBo/maxresdefault.jpg)](https://www.youtube.com/watch?v=r_4lAayCBBo)

<https://www.youtube.com/watch?v=r_4lAayCBBo>

Full Master cups playlist:
<https://www.youtube.com/playlist?list=PLD--3BNqNNAAzywGSzOT0VPQ6eSBUllQa>

## Requirements

- Linux or WSL2
- Python bootstrap: `python3` or `python` 3.10+. The project venv uses Python
  3.12, so either `uv` or `python3.12` must also be available.
- Rust toolchain with Cargo
- Node.js 20.19+, 22.12+, or 24+, with npm
- Mupen64Plus-Next libretro core shared library
- local US F-Zero X ROM

## Setup

```bash
git clone https://github.com/jiecovic/rl-fzerox.git
cd rl-fzerox
./install
./doctor
./fzerox
```

`./install` creates `.venv`, installs Python dependencies, builds the Rust/PyO3
emulator extension in release mode, installs the run manager frontend, and
creates the ignored `local/` folders.

Your system `python` does not have to be Python 3.12. It only needs to be new
enough to run the installer. If neither `uv` nor `python3.12` is installed,
install one of them first; otherwise the installer cannot create the project
`.venv`.

For NVIDIA CUDA training with a recent driver, use the CUDA 12.8 setup path:

```bash
./install --torch cu128
./doctor
./fzerox
```

If your driver/platform needs a different wheel, use the official PyTorch
selector: <https://pytorch.org/get-started/locally/>. Advanced install options:

```bash
./install --help
```

`local/` is the ignored machine-local workspace. After clone, it contains empty
folders for required runtime assets. Example paths used by the default app
config:

```text
local/libretro/mupen64plus_next_libretro.so
local/roms/fzerox_usa.n64
```

The ROM must be the US F-Zero X build; RAM offsets and telemetry are maintained
only for that version. The filename can be arbitrary: the runtime scans
`local/roms/` for `.n64`, `.z64`, and `.v64` files and uses the first compatible
ROM. The run manager also stores its SQLite DB, generated baselines, TensorBoard
views, and training runs under `local/`. None of those local files are included
in git.

The libretro core path and ROM folder are listed in
[docs/runtime_assets.md](docs/runtime_assets.md).

## F-Zero X App

The local UI/API is used for editing experiment specs, launching training,
watching policies, and tracking run state.

```bash
./fzerox
```

The UI runs at `http://localhost:5174`. The local API runs at
`http://127.0.0.1:8765`. The web server binds to loopback by default; pass
`--web-host 0.0.0.0` only when you intentionally want LAN access.

`just` is optional and mainly used as a developer shortcut. The equivalent
developer targets are `just setup`, `just setup-cuda`, `just doctor`, and
`just fzerox`.

## Quick Start With The Release Checkpoint

1. Open the run manager with `./fzerox`.
2. Put the libretro core and F-Zero X US ROM under the paths shown by
   `./doctor`. The UI also warns when either file is missing.
3. Open **Checkpoints** and install the published checkpoint.
4. Open the installed checkpoint. It behaves like a read-only run snapshot:
   you can watch it, inspect the config, view engine tuning, evaluate it, or
   fork it into an editable draft.
5. Use **Career** to let a checkpoint or run policy play through save-game
   progression. This is the path for checking what the policy can unlock or
   complete outside a single evaluation preset.
6. To train from scratch, create a new draft instead of forking a checkpoint,
   adjust the config, save it, and launch the run.

More project notes live in [docs](docs/index.md).
