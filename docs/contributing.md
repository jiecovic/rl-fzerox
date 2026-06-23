# Contributing

`rl-fzerox` is currently a solo-maintained research project. There is not a
formal contribution process yet. If you are interested in contributing, please
open an issue first so we can align on scope before implementation.

The repo uses the `Justfile` for routine local checks.

```bash
PYTHON=.venv/bin/python just py-lint
PYTHON=.venv/bin/python just py-test
```

Use `just native` for the default Rust release build path and `just check` for a
full local gate when you want Rust, Python, and tests together.
