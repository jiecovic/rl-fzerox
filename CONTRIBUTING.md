# Contributing

Use the repo-local toolchain when possible:
- `cargo` for Rust commands
- `just` for common quality checks

Common workflows:
- `just native` to rebuild the native extension in release mode
- `just native-dev` to rebuild it with the Rust dev profile
- `just fmt` to format Rust and Python code
- `just lint` to run `clippy`, `ruff`, and `pyright`
- `just test` to run Rust tests and Python tests
- `just check` to run the full local quality gate
- `just audit` to run `cargo audit`

Notes:
- The libretro core and ROM are user-supplied and are not distributed by this repo.
- `just native` uses `maturin develop -r` by default for runtime-oriented local builds.
- `just py-test` rebuilds the native extension first via `maturin`.
- `just` uses the active Python interpreter by default. Override it with
  `PYTHON=/path/to/python just <task>` if needed.
- Keep changes small, readable, and well-tested.
