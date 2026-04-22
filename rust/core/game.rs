// rust/core/game.rs
//! Game-specific native logic built on top of raw emulator memory, currently
//! focused on telemetry extraction.

pub(crate) mod memory;
pub mod race_start;
pub mod rng;
pub mod telemetry;
