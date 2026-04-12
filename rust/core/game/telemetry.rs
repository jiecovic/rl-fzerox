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

pub(crate) use layout::{player_r_button_timer_offset, player_z_button_timer_offset};
pub(crate) use model::StepTelemetrySample;
pub use model::{PlayerTelemetry, TelemetrySnapshot};
pub use read::read_snapshot;
pub(crate) use read::read_step_sample;

#[cfg(test)]
#[path = "tests/telemetry_tests.rs"]
mod tests;
