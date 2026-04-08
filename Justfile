set shell := ["bash", "-cu"]

default:
    @just --list

# Build and install the native extension in release mode using the active
# Python interpreter, or an explicit `PYTHON=/path/to/python` override.
native:
    PYTHON_BIN=${PYTHON:-python}; "$PYTHON_BIN" -m maturin develop -r -q

# Build and install the native extension with the Rust dev profile.
native-dev:
    PYTHON_BIN=${PYTHON:-python}; "$PYTHON_BIN" -m maturin develop -q

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
    PYTHON_BIN=${PYTHON:-python}; "$PYTHON_BIN" -m ruff format src tests

py-fmt-check:
    PYTHON_BIN=${PYTHON:-python}; "$PYTHON_BIN" -m ruff format --check src tests

py-lint:
    PYTHON_BIN=${PYTHON:-python}; "$PYTHON_BIN" -m ruff check src tests
    PYTHON_BIN=${PYTHON:-python}; PYTHONPATH=src "$PYTHON_BIN" -m pyright src tests

py-test: native
    PYTHON_BIN=${PYTHON:-python}; PYTHONPATH=src "$PYTHON_BIN" -m pytest

# Aggregate repo-wide quality tasks.
fmt: rust-fmt py-fmt

fmt-check: rust-fmt-check py-fmt-check

lint: rust-lint py-lint

test: rust-test py-test

check: fmt-check lint test

# Audit Rust dependencies for published advisories.
audit:
    if command -v cargo-audit >/dev/null 2>&1; then cargo audit; else echo "cargo-audit is not installed"; exit 1; fi
