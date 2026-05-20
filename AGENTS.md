# AGENTS.md

## Commit Message Convention

Use Conventional Commits with a scope:

`<type>(<scope>): <summary>`

### Allowed Types

- `feat`
- `fix`
- `refactor`
- `perf`
- `docs`
- `test`
- `chore`
- `ci`
- `build`
- `style`
- `revert`

### Rules

- Use present tense, imperative mood.
- Keep summary lowercase and concise (target <= 72 chars).
- Do not end summary with a period.
- Keep each commit focused on one logical change.
- Add a body when needed to explain why, not what.
- Use footer for metadata:
  - `BREAKING CHANGE: ...`
  - `Closes #123`

### Examples

- `feat(ocr): retry once when llm ocr response is truncated`
- `fix(agent): prevent duplicate box id assignment in page merge`
- `chore(repo): ignore local agent guidance file`
- `docs(readme): clarify backend and frontend dev ports`

## Code Quality Principles

This repository should not become an AI-slop codebase. Favor code that a
strong human engineer would be comfortable owning, extending, and reviewing.

### Standards

- Prefer explicit, maintainable designs over fast, clever, or over-abstracted
  generated code.
- Keep modules small and cohesive; split files when doing so improves clarity.
- When splitting one subsystem into several related implementation files, prefer
  a small module directory with a facade module over many loose sibling files.
  For example, use `foo.rs` plus `foo/bar.rs` files when the `bar` pieces are
  owned by `foo`.
- Avoid ad-hoc suffix names such as `_state`, `_utils`, or `_helpers` when a
  clearer module boundary or nested module name can express the responsibility.
- Use clear names, straightforward control flow, and predictable data flow.
- Document the intent behind non-obvious behavior, invariants, and tradeoffs.
- Write code and comments for the next human maintainer, not for the model.
- Avoid speculative abstractions, premature generalization, and pattern cargo
  culting.
- Do not introduce large framework layers or indirection unless they solve a
  concrete problem already present in the codebase.
- Keep public interfaces narrow and well-defined.
- Prefer composition over inheritance when either would work.
- Make state transitions and side effects easy to trace.
- Avoid loose “constant soup” for related literal values when a proper data
  structure would be clearer. Group RAM offsets, protocol ids, wire values, and
  enum constants in small immutable data structures when reasonable.
- Prefer proper typing over escape hatches. Do not use `typing.cast` as a
  shortcut for “trust me”; tighten the real types with better stubs,
  `TypedDict`, protocols, helper functions, or runtime narrowing.
- Keep static typing pragmatic at third-party dynamic boundaries. Do not add
  large local `Protocol` layers just to compensate for weak or missing typing in
  libraries such as pygame, Stable-Baselines3, Gymnasium, or native extension
  modules. Prefer small typed wrappers, narrow helper functions, or explicit
  project-owned type aliases when they improve maintainability.
- Use the centralized aliases in `fzerox_emulator.arrays` for project-owned
  NumPy concepts such as RGB frames, observations, masks, and action vectors.
  Do not leave raw `np.ndarray` or direct `NDArray[...]` types on new
  project-owned interfaces when the dtype is known.
- Treat pyright warnings about missing parameter annotations as work to clean up
  on touched code, but avoid broad type-hardening refactors unless the task is
  explicitly about typing.
- Add or update tests for behavior changes when feasible; if skipped, state why.
- For Rust modules, keep test bodies out of implementation files. Prefer the
  nearest existing `tests/` folder, e.g. `foo.rs` plus
  `tests/foo_tests.rs`, with the implementation file containing only
  `#[cfg(test)] #[path = "..."] mod tests;`.
- Do not leave behind dead code, commented-out code, placeholder hooks, or vague
  TODOs.
- Before finishing meaningful changes, ensure the touched surface is readable
  end-to-end and remove obvious duplication.

### Documentation

- Keep README-level docs and developer guidance in sync with architectural
  changes.
- Add brief docstrings or comments where they materially reduce reader effort.
- For complex subsystems, document the core concepts, data flow, and extension
  points close to the code.

### Copyrighted Runtime Assets

- Never add ROM files or copyrighted game binaries to git. F-Zero X ROMs must
  stay local on the developer machine, normally under `local/roms/`.
- Tests may use fake temporary paths such as `tmp_path / "fzerox.n64"` only to
  satisfy path validation. Do not copy, generate, vendor, or commit real ROM
  contents for tests, fixtures, examples, releases, or checkpoints.
- Save states and generated run artifacts must not embed or require vendored
  ROM contents. If a local artifact is needed for development, keep it under an
  ignored local/cache/run directory unless it is legally safe to distribute.

## Local Tooling

- Use the repo `Justfile` for routine local quality checks when possible.
- Python recipes use the active interpreter by default. Override with
  `PYTHON=/path/to/python just <task>` when needed.
- `just native` should be the default local native build path and uses a
  Rust release build. Use `just native-dev` only when you explicitly want
  the Rust dev profile for debugging.
- Rust-focused tasks:
  - `just rust-fmt`
  - `just rust-lint`
  - `just rust-test`
  - `just audit`
- Python-focused tasks:
  - `just py-fmt`
  - `just py-lint`
  - `just py-test`
- Full repo checks:
  - `just fmt-check`
  - `just lint`
  - `just test`
  - `just check`

## Interview-Ready Mode

When interview prep or paired-coding context is active, optimize for code
readability, clear reasoning, and presentable delivery.

### Standards

- Prefer simple, explicit solutions over clever shortcuts.
- Keep changes small and logically scoped.
- Add brief code comments for tricky or non-obvious logic.
- Add a single-line file path comment as the first line in every source file.
- Aim to keep files under ~700 LOC; guideline only (not a hard guardrail).
- Split/refactor when it improves clarity or testability.
- Avoid dead code, commented-out code, and vague TODOs.
- Use descriptive names and clear control flow.
- Add or update tests for behavior changes; if skipped, state why.
- Run relevant lint, typecheck, and tests before finishing when feasible.
- Before commits or larger changes/refactors, run the relevant quality tools
  for the touched surface (for example `ruff`, `pyright`, `pytest`,
  frontend typecheck, frontend lint) and do not leave known failures
  unreported.
- In final responses, include what changed, why this approach, and tradeoffs.

### Quality Gate

Do not mark work complete if reasonable verification was feasible but not run.
Do not commit larger changes/refactors without first running the relevant lint,
typecheck, and test commands for that scope.
