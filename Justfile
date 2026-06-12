set shell := ["bash", "-cu"]

python_bin := `if [ -n "${PYTHON:-}" ]; then printf '%s' "${PYTHON}"; elif [ -x .venv/bin/python ]; then printf '%s' .venv/bin/python; else printf '%s' python; fi`

default:
    @just --list

# Build and install the native extension in release mode using the active
# Python interpreter, or an explicit `PYTHON=/path/to/python` override.
native:
    "{{python_bin}}" -m maturin develop -r -q --skip-install

# Build and install the native extension with the Rust dev profile.
native-dev:
    "{{python_bin}}" -m maturin develop -q --skip-install

# Apply Rust formatters only.
rust-fmt:
    cargo fmt

# Verify Rust formatting and lints.
rust-fmt-check:
    cargo fmt --check

rust-lint:
    cargo clippy --all-targets --all-features -- -D warnings

rust-test:
    cargo test

# Apply Python formatters only.
py-fmt:
    "{{python_bin}}" -m ruff format src tests scripts/check_numpy_typing.py

py-fmt-check:
    "{{python_bin}}" -m ruff format --check src tests scripts/check_numpy_typing.py

py-lint:
    "{{python_bin}}" scripts/check_numpy_typing.py src tests scripts
    "{{python_bin}}" -m ruff check src tests scripts/check_numpy_typing.py
    PYTHONPATH=src "{{python_bin}}" -m pyright src tests scripts/check_numpy_typing.py

py-test: native
    @PYTHONPATH=src "{{python_bin}}" -c 'import sys; print(f"pytest interpreter: {sys.executable}"); import fzerox_emulator._native as native; print(f"native module: {native.__file__}")'
    PYTHONPATH=src "{{python_bin}}" -m pytest

# Install the local React run-manager frontend dependencies.
run-manager-install:
    npm install --prefix web/run-manager

# Verify the local React run-manager frontend.
run-manager-check:
    npm run --prefix web/run-manager check

# Build the local React run-manager frontend.
run-manager-build:
    npm run --prefix web/run-manager build

# Launch the main F-Zero X app with the Python SQLite API.
fzerox:
    @EGL_PLATFORM="${EGL_PLATFORM:-x11}" PYTHONPATH=src "{{python_bin}}" -m rl_fzerox.apps.run_manager

# Compatibility alias for the old run-manager entrypoint.
run-manager: fzerox

# Aggregate repo-wide quality tasks.
fmt: rust-fmt py-fmt

fmt-check: rust-fmt-check py-fmt-check

lint: rust-lint py-lint

test: rust-test py-test

check: fmt-check lint test run-manager-check

# Audit Rust dependencies for published advisories.
audit:
    if command -v cargo-audit >/dev/null 2>&1; then cargo audit; else echo "cargo-audit is not installed"; exit 1; fi
