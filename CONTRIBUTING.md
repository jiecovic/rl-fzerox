# Contributing

This repo uses the `Justfile` for routine local checks.

```bash
PYTHON=.venv/bin/python just py-lint
PYTHON=.venv/bin/python just py-test
```

Use `just native` for the default Rust release build path and `just check` for a
full local gate when you want Rust, Python, and tests together.
