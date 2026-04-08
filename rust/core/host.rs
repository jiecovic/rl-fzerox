// rust/core/host.rs
//! Native libretro host subsystem.
//!
//! This owns the loaded core instance, callback state, input mapping, video
//! processing, and runtime helpers used by the Python-facing bindings.

pub mod callbacks;
pub mod input;
pub mod options;
pub mod runtime;
pub mod stdio;
pub mod video;

pub use runtime::{Host, RepeatedStepConfig, StepSummary};
