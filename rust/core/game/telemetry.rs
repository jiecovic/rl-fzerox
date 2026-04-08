// rust/core/game/telemetry.rs
//! F-Zero X telemetry decoding from live RDRAM snapshots.
//!
//! The module is split into:
//! - `layout`: reverse-engineered addresses, offsets, and flag enums
//! - `model`: typed telemetry structs exposed to the rest of the codebase
//! - `read`: RAM decoding helpers and the `read_snapshot(...)` entry point

mod layout;
mod model;
mod read;

pub use model::{PlayerTelemetry, TelemetrySnapshot};
pub use read::read_snapshot;

#[cfg(test)]
#[path = "tests/telemetry_tests.rs"]
mod tests;
