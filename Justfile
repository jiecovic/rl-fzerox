set shell := ["bash", "-cu"]

python_bin := `if [ -n "${PYTHON:-}" ]; then printf '%s' "${PYTHON}"; elif [ -x .venv/bin/python ]; then printf '%s' .venv/bin/python; else printf '%s' python; fi`

default:
    @just --list

# Build and install the native extension in release mode using the active
# Python interpreter, or an explicit `PYTHON=/path/to/python` override.
native:
    "{{python_bin}}" -m maturin develop -r -q --skip-install

# Install Python, native, and frontend dependencies for local use.
setup:
    ./install

# Install with the CUDA 12.8 PyTorch wheel before project dependencies.
setup-cuda:
    ./install --torch cu128

# Check whether local dependencies and runtime assets are ready.
doctor:
    ./doctor

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
    "{{python_bin}}" -m ruff format src tests scripts

py-fmt-check:
    "{{python_bin}}" -m ruff format --check src tests scripts

py-lint:
    "{{python_bin}}" scripts/check_numpy_typing.py src tests scripts
    "{{python_bin}}" -m ruff check src tests scripts
    PYTHONPATH=src "{{python_bin}}" -m pyright src tests scripts

py-test: native
    @PYTHONPATH=src "{{python_bin}}" -c 'import sys; print(f"pytest interpreter: {sys.executable}"); import fzerox_emulator._native as native; print(f"native module: {native.__file__}")'
    PYTHONPATH=src "{{python_bin}}" -m pytest

# Run Python tests against the currently installed native extension.
py-test-no-native *pytest_args:
    @PYTHONPATH=src "{{python_bin}}" -c 'import sys; print(f"pytest interpreter: {sys.executable}"); import fzerox_emulator._native as native; print(f"native module: {native.__file__}")'
    PYTHONPATH=src "{{python_bin}}" -m pytest {{pytest_args}}

# Install the local React run-manager frontend dependencies.
run-manager-install:
    npm install --prefix web/run-manager

# Verify the local React run-manager frontend.
run-manager-check:
    npm run --prefix web/run-manager check

# Build the local React run-manager frontend.
run-manager-build:
    npm run --prefix web/run-manager build

# Build and open the local Markdown documentation preview.
docs: docs-open-all

# Render one Markdown file to local/preview with GitHub-style CSS and MathJax.
docs-preview file="README.md":
    @set -euo pipefail; \
    if ! command -v pandoc >/dev/null 2>&1; then \
        echo "pandoc is not installed. On Arch: sudo pacman -S pandoc"; \
        exit 1; \
    fi; \
    if ! command -v curl >/dev/null 2>&1; then \
        echo "curl is required to fetch the preview CSS"; \
        exit 1; \
    fi; \
    input="{{file}}"; \
    if [ ! -f "$input" ]; then \
        echo "Markdown file not found: $input"; \
        exit 1; \
    fi; \
    preview_dir="local/preview"; \
    css_path="$preview_dir/github-markdown.css"; \
    mkdir -p "$preview_dir"; \
    if [ ! -f "$css_path" ]; then \
        curl -fsSL https://raw.githubusercontent.com/sindresorhus/github-markdown-css/main/github-markdown.css -o "$css_path"; \
    fi; \
    stem="${input#./}"; \
    stem="${stem%.*}"; \
    title="${stem##*/}"; \
    output="$preview_dir/$stem.html"; \
    mkdir -p "$(dirname "$output")"; \
    css_href="$(realpath --relative-to="$(dirname "$output")" "$css_path")"; \
    pandoc -f gfm+tex_math_dollars -t html5 -s --mathjax -V body-class=markdown-body --metadata title="$title" --css "$css_href" "$input" -o "$output"; \
    "{{python_bin}}" -c 'from pathlib import Path; import re, sys; p = Path(sys.argv[1]); text = p.read_text(encoding="utf-8"); text = re.sub("href=\"((?![a-zA-Z][a-zA-Z0-9+.-]*:)[^\"]+)\\.md(#[^\"]*)?\"", "href=\"\\1.html\\2\"", text); p.write_text(text, encoding="utf-8")' "$output"; \
    echo "$output"

# Render one Markdown file and open the generated local preview in the browser.
docs-open file="README.md":
    @just docs-preview "{{file}}"
    @set -euo pipefail; \
    input="{{file}}"; \
    stem="${input#./}"; \
    stem="${stem%.*}"; \
    output="local/preview/$stem.html"; \
    xdg-open "$output"

# Render every tracked Markdown file to local/preview.
docs-preview-all:
    @set -euo pipefail; \
    git ls-files '*.md' | while IFS= read -r file; do \
        just docs-preview "$file" >/dev/null; \
    done; \
    count="$(git ls-files '*.md' | wc -l)"; \
    echo "Rendered $count Markdown files to local/preview"

# Render every tracked Markdown file and open the docs index preview.
docs-open-all:
    @just docs-preview-all
    @if [ -f local/preview/docs/index.html ]; then \
        xdg-open local/preview/docs/index.html; \
    else \
        xdg-open local/preview/README.html; \
    fi

# Launch the main F-Zero X app with the Python SQLite API.
fzerox:
    @./fzerox

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
