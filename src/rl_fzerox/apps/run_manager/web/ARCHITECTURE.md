# Run Manager Frontend Architecture

The run-manager frontend uses a layered, feature-oriented structure. Layers have
one dependency direction:

```text
app -> pages -> widgets -> features -> entities -> shared
```

Lower layers do not import higher layers. Cross-layer data flows through props,
explicit function arguments, or API contracts.

## Folder Layout

```text
src/
  app/
  pages/
  widgets/
  features/
  entities/
  shared/
  test/
```

## Layer Ownership

### `app/`

Owns application startup, providers, global workspace orchestration, top-level
route or tab wiring, and process-level state. This layer may compose all lower
layers.

### `pages/`

Owns top-level screens and tabs. A page chooses which widgets and features are
visible for one screen, but detailed domain mutation logic belongs below it.

### `widgets/`

Owns large composed UI regions such as a configurator panel, run workspace,
chart panel, save-game workspace, or live-runtime panel. A widget may coordinate
several features and entities for one visible region.

### `features/`

Owns user-triggered workflows with narrow public APIs. Examples include
launching watch, resuming training, editing a course setup, applying a bulk
default, resetting track stats, or forking a run.

### `entities/`

Owns domain concepts and their local model helpers, formatting, selectors, and
small reusable view fragments. Examples include run, draft, save game, track,
course, policy, artifact, and config section.

### `shared/`

Owns generic UI primitives, API transport/client contracts, browser helpers,
formatting utilities, test helpers, and global styles. Shared code stays
domain-agnostic unless it is part of the API contract layer.

### `test/`

Owns frontend tests. Tests should cover public behavior and important model
helpers instead of private layout details.

## Module Shape

Larger modules use small internal directories when that improves readability:

```text
someModule/
  api/
  lib/
  model/
  ui/
```

Use only the directories that the module actually needs. Public imports should
come from the module boundary instead of deep implementation files when a stable
boundary exists.

Project-local imports use the configured `@/...` alias.

## Styling Boundary

Ordinary layout, spacing, borders, typography, and simple states use Tailwind
theme utilities and shared UI primitives.

CSS files are reserved for global tokens and complex visuals where utility
classes would reduce readability, such as charts, sliders, minimaps, dense
tables, and SVG diagrams.

## References

- React: <https://legacy.reactjs.org/docs/faq-structure.html>
- Redux style guide: <https://redux.js.org/style-guide/>
- Feature-Sliced Design: <https://fsd.how/docs/get-started/overview/>
- bulletproof-react:
  <https://github.com/alan2207/bulletproof-react/blob/master/docs/project-structure.md>
