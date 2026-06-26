## Commit Messages

Use Conventional Commits with a scope: `<type>(<scope>): <summary>`.

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

Rules: present tense, imperative mood, lowercase summary, no trailing period,
focused logical change. Add body/footer only when useful.

Examples:

- `feat(run-manager): add run comparison chart`
- `fix(emulator): reject unsupported rom revisions`
- `docs(readme): clarify local asset paths`
- `refactor(policy): split extractor network blocks`

## Architecture Boundaries

- Keep modules small and cohesive. Around 700 LOC, actively consider a split;
  below about 500 LOC is a good target when the split stays natural.
- Prefer owned packages with small facades over many loose sibling files.
- Do not leave empty source directories behind after moves. Avoid one-file
  packages unless they are deliberate public facades or near-term grouping
  points.
- Avoid vague module suffixes such as `_state`, `_utils`, or `_helpers` when a
  clearer module boundary can express the responsibility.
- Keep public interfaces narrow and well-defined.
- Do not introduce framework layers or indirection unless they solve a concrete
  problem already present in the codebase.
- New helpers or files must pay for themselves immediately by reducing call
  paths, concepts, duplication, or test burden.
- Do not keep aliases, shims, fallback readers, or legacy paths unless they
  protect a shipped public contract or an explicit migration.
- Add config or env knobs only when existing defaults, UI, or schema cannot
  represent the behavior cleanly.
- In Python, prefer absolute imports from project packages. Relative imports are
  acceptable only for same-package local files, and should stay shallow
  (`from .foo import ...`, not parent-package hops).
- Add a single-line file path comment as the first line in every source file.

## Source of Truth

- In managed run-manager flows, SQLite owns run specs, lifecycle, and mutable
  manager state.
- `train_manifest.yaml` is only a mirror of the managed run spec. Do not use it
  as a fallback or source for managed behavior; write or verify it after reading
  from SQLite.
- Published checkpoints are catalog/install records that point at read-only
  archived run snapshots. Once installed, watch, fork, evaluation, save-game
  policy selection, and workspace views should use the linked run id instead of
  checkpoint-specific launch or workspace paths.
- If a touched area has mixed ownership, multiple sources of truth, legacy
  compatibility paths, or other structural debt, call that out before extending
  it. Prefer fixing the boundary over adding behavior on top of it.
- Do not commit one-off DB or data migration scripts to repo source. Keep
  throwaway migration tooling under ignored `local/` paths, run it explicitly
  during the maintenance window, and remove temporary compatibility guards
  before committing the durable schema/code change.

## Runtime Lifecycle

- Controllers/FSMs own user-visible lifecycle decisions. Document their states,
  transitions, permitted side effects, and lifecycle events close to the code.
- Workers own process IO, command draining, timing, and resource cleanup. They
  should consume controller signals, not become a second lifecycle owner.
- Recording, logging, and persistence should react to explicit lifecycle events,
  not infer boundaries from DB progress, telemetry byproducts, files, or
  frontend sync state.

## Implementation Standards

- Prefer explicit, maintainable designs over clever or over-abstracted code.
- Use clear names, straightforward control flow, and predictable data flow.
- Document intent behind non-obvious behavior, invariants, and tradeoffs.
- Avoid speculative abstractions, premature generalization, and copied patterns
  without a clear reason.
- Group related literal values in proper data structures when that clarifies
  RAM offsets, protocol ids, wire values, enum constants, or config tables.
- Do not leave dead code, commented-out code, placeholder hooks, or vague TODOs.
- Before finishing meaningful changes, reread the touched surface end to end and
  remove obvious duplication.

## Interfaces and Performance

- Prefer composition and small capability interfaces over deep inheritance. Use
  Rust traits or Python `Protocol` for project-owned behavior boundaries when
  they reduce coupling without hiding simple data flow.
- Keep hot loops allocation-light and data-oriented. Cache invariant lookups,
  avoid repeated schema/object construction, and keep logging out of per-frame
  paths unless explicitly sampled or gated.
- Avoid repeated discovery or polling in runtime hot paths. Prepare stable
  facts once and pass them forward.
- Put emulator-hot, frame-processing, and large array work in Rust, NumPy, or
  narrow helper functions instead of Python object-heavy loops.
- Prefer measuring the hot path before micro-optimizing. Preserve readability
  unless profiling or obvious algorithmic cost shows the code is hot.

## Typing

- Prefer proper typing over escape hatches. Do not use `typing.cast` to bypass
  weak types; tighten the real types with better stubs, `TypedDict`,
  protocols, helper functions, or runtime narrowing.
- Keep static typing pragmatic at third-party dynamic boundaries. Avoid large
  local `Protocol` layers just to compensate for weak typing in pygame,
  Stable-Baselines3, Gymnasium, or native extension modules.
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
- Keep user docs short and practical. Start with what the user wants to do, then
  list the necessary commands, settings, or UI steps.
- Do not try to fully document every subsystem in one pass. Establish the
  minimal map first, then expand one topic at a time.
- Keep theory docs separate from architecture. Theory explains why a design
  exists; architecture explains where state lives and how code paths connect.
- In user-facing docs, avoid opening with internal nouns such as "course
  target", "materialization", "runtime projection", "backend", or "source of
  truth".
- Prefer tables, short rules, and examples over long narrative sections.
- Avoid negative framing and conversational residue in final docs. Define what a
  concept is before mentioning what it is not.
- Use short `Note:` or `Non-goals:` sections only when an exclusion materially
  helps users understand an API boundary or workflow.

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
- Run relevant lint, typecheck, and tests before finishing or committing
  meaningful changes. If a feasible check cannot be run, report that explicitly.
- For reviews or verdicts, read the changed function or module plus relevant
  callers, callees, sibling code, and tests. Do not judge from diff-only
  context.
- For dependency-backed behavior, inspect upstream docs, source, or types before
  relying on API defaults, error behavior, or timing assumptions.

Useful checks:

- Rust: `just rust-fmt`, `just rust-lint`, `just rust-test`, `just audit`
- Python: `just py-fmt`, `just py-lint`, `just py-test`
- Python-only while reusing the installed native extension:
  `just py-test-no-native`
- Full repo: `just fmt-check`, `just lint`, `just test`, `just check`
