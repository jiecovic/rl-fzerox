# rl-fzerox

Experimental deep reinforcement-learning project for training agents to race in
F-Zero X through a libretro N64 emulator.

This repository is still under construction. A trained checkpoint and fuller
documentation are planned soon.

[![rl-fzerox teaser video](https://img.youtube.com/vi/aWTHONOQbAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=aWTHONOQbAg)

## Requirements

- Linux userspace, either native Linux or Windows with WSL2
- Python 3.11
- Rust toolchain with Cargo
- Node.js and npm
- Mupen64Plus-Next libretro core shared library
- a local US F-Zero X ROM
- `sb3x-extensions`

## Setup

`just native` builds the Rust/PyO3 emulator extension in release mode for the
active Python environment.

```bash
git clone https://github.com/jiecovic/rl-fzerox.git
cd rl-fzerox

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,watch,train]"
python -m pip install "sb3x @ git+https://github.com/jiecovic/sb3x-extensions.git"

just native
just run-manager-install
```

`local/` is the ignored machine-local workspace. After clone, it contains empty
placeholder folders for required runtime assets. Example paths used by the
default run-manager config:

```text
local/libretro/mupen64plus_next_libretro.so
local/roms/F-Zero X (USA).n64
```

The ROM must be the US F-Zero X build; RAM offsets and telemetry are maintained
only for that version. If you use different local filenames, update the paths in
the run manager. The run manager also stores its SQLite DB, generated baselines,
TensorBoard views, and training runs under `local/`. None of those local files
are included in git.

Tracked runtime binary policy is documented in
[docs/runtime_assets.md](docs/runtime_assets.md).

## Run Manager

The run manager is the local UI/API for editing experiment specs, launching
training, watching policies, and tracking run state.

```bash
just run-manager
```

The UI runs at `http://localhost:5174`. The local API runs at
`http://127.0.0.1:8765`. The web server binds to loopback by default; pass
`--web-host 0.0.0.0` only when you intentionally want LAN access.

More project notes live in [docs](docs/index.md).
