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

- Linux userspace, either native Linux or Windows with WSL2
- Python 3.12 or newer
- Rust toolchain with Cargo
- Node.js 20.19+, 22.12+, or 24+, with npm
- `just`
- Mupen64Plus-Next libretro core shared library
- a local US F-Zero X ROM
- `sb3x-extensions`

## Setup

`just native` builds the Rust/PyO3 emulator extension in release mode for the
active Python environment.

Create a virtual environment with Python 3.12. `uv` is a convenient option on
distributions whose default `python` is newer; Conda, pyenv, or a system
`python3.12` are also fine.

```bash
git clone https://github.com/jiecovic/rl-fzerox.git
cd rl-fzerox

uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
python --version
python -m pip install -U pip
python -m pip install -e ".[dev,watch,train]"
python -m pip install "sb3x @ git+https://github.com/jiecovic/sb3x-extensions.git"

just native
just run-manager-install
```

If your active interpreter is already Python 3.12, `python -m venv .venv` also
works. Check `python --version` before installing dependencies.

`just run-manager-install` installs the local React frontend dependencies used
by `just fzerox`.

`local/` is the ignored machine-local workspace. After clone, it contains empty
folders for required runtime assets. Example paths used by the default app
config:

```text
local/libretro/mupen64plus_next_libretro.so
local/roms/F-Zero X (USA).n64
```

The ROM must be the US F-Zero X build; RAM offsets and telemetry are maintained
only for that version. If you use different local filenames, update the paths in
the run manager. The run manager also stores its SQLite DB, generated baselines,
TensorBoard views, and training runs under `local/`. None of those local files
are included in git.

The default ROM and libretro core paths are listed in
[docs/runtime_assets.md](docs/runtime_assets.md).

## F-Zero X App

The local UI/API is used for editing experiment specs, launching training,
watching policies, and tracking run state.

```bash
just fzerox
```

The UI runs at `http://localhost:5174`. The local API runs at
`http://127.0.0.1:8765`. The web server binds to loopback by default; pass
`--web-host 0.0.0.0` only when you intentionally want LAN access.

More project notes live in [docs](docs/index.md).
