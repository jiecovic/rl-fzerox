# AGENTS.md

These rules are repo-specific. Prefer existing local patterns and ownership
boundaries over generic cleanup advice.

## Commit Messages

Use Conventional Commits with a scope:

`<type>(<scope>): <summary>`

Allowed types:

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

Rules:

- Use present tense, imperative mood.
- Keep the summary lowercase and concise, target 72 characters or less.
- Do not end the summary with a period.
- Keep each commit focused on one logical change.
- Add a body when the reason is not obvious from the diff.
- Use footers for breaking changes or issue metadata when needed.

Examples:

- `feat(run-manager): add run comparison chart`
- `fix(emulator): reject unsupported rom revisions`
- `docs(readme): clarify local asset paths`
- `refactor(policy): split extractor network blocks`

## Architecture Boundaries

- Keep modules small and cohesive. Split files when doing so improves clarity,
  testability, or ownership.
- When one subsystem needs several implementation files, prefer an owned
  package with a small facade over many loose sibling files.
- Do not leave empty source directories behind after moves. Avoid one-file
  packages unless they are deliberate public facades or near-term grouping
  points.
- Avoid vague module suffixes such as `_state`, `_utils`, or `_helpers` when a
  clearer module boundary can express the responsibility.
- Keep public interfaces narrow and well-defined.
- Do not introduce framework layers or indirection unless they solve a concrete
  problem already present in the codebase.
- Before adding behavior to a large runtime file, first look for a cohesive
  split that keeps side effects and state transitions easier to trace.
- Add a single-line file path comment as the first line in every source file.

## Source of Truth

- For managed run-manager flows, SQLite is the source of truth for run specs
  and mutable runtime state.
- Treat filesystem manifests such as `train_manifest.yaml` as mirrors of the
  SQLite-managed run spec, not as fallbacks or sources of truth.
- Checkpoints, baselines, TensorBoard logs, and exported bundles are filesystem
  artifacts. They may be authoritative for their own payloads, but managed code
  must not infer run specs, lifecycle state, or mutable manager state from their
  presence or contents.
- Non-managed CLI paths may explicitly load a run directory as input, but that
  must stay separate from managed run-manager behavior.
- If a touched area has mixed ownership, multiple sources of truth, legacy
  compatibility paths, or other structural debt, call that out before extending
  it. Prefer fixing the boundary over adding behavior on top of it.
- Do not commit one-off DB or data migration scripts to repo source. Keep
  throwaway migration tooling under ignored `local/` paths, run it explicitly
  during the maintenance window, and remove temporary compatibility guards
  before committing the durable schema/code change.

## Runtime Lifecycle

- FSM/controller modules that own user-visible lifecycle must document states,
  transitions, and permitted side effects close to the code.
- Lifecycle boundaries such as success, failure, retry, and exit should be
  emitted as explicit domain events or signals.
- Worker modules should remain orchestration entrypoints for process IO,
  command draining, timing, and resource cleanup.
- Do not let workers become the source of truth for domain state transitions
  when a controller/FSM already owns that flow.
- Recording, logging, and persistence should consume explicit lifecycle signals
  from the domain owner. Do not infer live control boundaries from manager DB
  progress, telemetry byproducts, file existence, or frontend sync state.

## Implementation Standards

- Prefer explicit, maintainable designs over clever or over-abstracted code.
- Use clear names, straightforward control flow, and predictable data flow.
- Document intent behind non-obvious behavior, invariants, and tradeoffs.
- Keep comments useful for the next maintainer; do not restate obvious code.
- Avoid speculative abstractions, premature generalization, and copied patterns
  without a clear reason.
- Prefer composition over inheritance when either would work.
- Group related literal values in proper data structures when that clarifies
  RAM offsets, protocol ids, wire values, enum constants, or config tables.
- Do not leave dead code, commented-out code, placeholder hooks, or vague TODOs.
- Before finishing meaningful changes, reread the touched surface end to end and
  remove obvious duplication.

## Typing

- Prefer proper typing over escape hatches. Do not use `typing.cast` to bypass
  weak types; tighten the real types with better stubs, `TypedDict`,
  protocols, helper functions, or runtime narrowing.
- Keep static typing pragmatic at third-party dynamic boundaries. Avoid large
  local `Protocol` layers just to compensate for weak typing in pygame,
  Stable-Baselines3, Gymnasium, or native extension modules.
- Prefer small typed wrappers, narrow helper functions, or project-owned type
  aliases when they improve maintainability.
- Use centralized aliases in `fzerox_emulator.arrays` for project-owned NumPy
  concepts such as RGB frames, observations, masks, and action vectors.
- Do not leave raw `np.ndarray` or direct `NDArray[...]` types on new
  project-owned interfaces when the dtype is known.
- Treat pyright warnings about missing parameter annotations as work to clean up
  on touched code, but avoid broad type-hardening refactors unless the task is
  explicitly about typing.

## Frontend

- In the run-manager frontend, use configured path aliases such as `@/...` for
  project-local source and asset imports instead of relative `./` or `../`
  imports.
- Use Tailwind theme utilities and shared UI primitives for ordinary layout,
  spacing, borders, typography, and simple states.
- Keep CSS files for global tokens and complex visuals such as charts, sliders,
  minimaps, dense tables, and SVG diagrams when utility classes would make the
  component harder to read.

## Rust

- Keep Rust test bodies out of implementation files. Prefer the nearest
  existing `tests/` folder, e.g. `foo.rs` plus `tests/foo_tests.rs`, with the
  implementation file containing only `#[cfg(test)] #[path = "..."] mod tests;`.
- `just native` is the default native build path and uses a Rust release build.
  Use `just native-dev` only when explicitly debugging the Rust dev profile.

## Documentation

- Keep README-level docs and developer guidance in sync with architectural
  changes.
- Add brief docstrings or comments where they materially reduce reader effort.
- For complex subsystems, document core concepts, data flow, and extension
  points close to the code.
- Do not use a broad `concepts/` directory as a catch-all. Separate docs by
  reader need: user operation, theory, architecture, and development workflow.
- Keep user docs short and practical. Start with what the user wants to do, then
  list the necessary commands, settings, or UI steps.
- Do not try to fully document every subsystem in one pass. Establish the
  minimal map first, then expand one topic at a time.
- Keep theory docs separate from architecture. Theory explains why a design
  exists; architecture explains where state lives and how code paths connect.
- In user-facing docs, start from F-Zero X language before introducing project
  terms. Use concrete examples like "Mute City, GP Race, Master, Blue Falcon,
  ENG 50" before naming internal objects or schemas.
- In user-facing docs, avoid opening with internal nouns such as "course
  target", "materialization", "runtime projection", "backend", or "source of
  truth".
- Prefer tables, short rules, and examples over long narrative sections.
  Equations and schema fields belong only where they clarify a decision the
  reader must make.
- Avoid negative framing in final docs. Define what a concept is before
  mentioning what it is not.
- Do not carry conversational corrections, review back-and-forth, or planning
  residue into documentation prose.
- Use short `Note:` or `Non-goals:` sections only when an exclusion materially
  helps users understand an API boundary or workflow.
- Keep documentation standalone, neutral, and implementation-focused.

## Runtime Assets

- Never add ROM files or copyrighted game binaries to git. F-Zero X ROMs must
  stay local on the developer machine, normally under `local/roms/`.
- Tests may use fake temporary paths such as `tmp_path / "fzerox.n64"` only to
  satisfy path validation. Do not copy, generate, vendor, or commit real ROM
  contents for tests, fixtures, examples, releases, or checkpoints.
- Save states and generated run artifacts must not embed or require vendored
  ROM contents. Keep local development artifacts under ignored paths such as
  `local/runs/` or `local/cache/` unless they are legally safe to distribute.

## Quality Gate

- Use the repo `Justfile` for routine local quality checks when possible.
- Python recipes use the active interpreter by default. Override with
  `PYTHON=/path/to/python just <task>` when needed.
- Add or update tests for behavior changes when feasible. If skipped, state why.
- Run relevant lint, typecheck, and tests before finishing meaningful changes.
- Do not mark work complete if reasonable verification was feasible but not run.
- Do not commit larger changes or refactors without first running the relevant
  quality tools for the touched surface.
- If a check cannot be run, report that explicitly.

Useful checks:

- Rust: `just rust-fmt`, `just rust-lint`, `just rust-test`, `just audit`
- Python: `just py-fmt`, `just py-lint`, `just py-test`
- Python-only while reusing the installed native extension:
  `just py-test-no-native`
- Full repo: `just fmt-check`, `just lint`, `just test`, `just check`
