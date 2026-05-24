// rust/core/host/runtime.rs
//! `Host` runtime facade over one loaded libretro core instance.
//!
//! The helper submodules split the implementation by responsibility:
//! bootstrap/lifecycle, baseline-state management, memory access, and the
//! main `Host` façade itself.

mod baseline;
mod bootstrap;
mod control;
mod host;
mod lean_slide;
mod memory;
mod race_start;
mod render;
mod repeat;
mod rng;
mod step;
mod step_accumulator;
mod telemetry;

pub use host::Host;
pub use step::{
    DisplayFrameBatch, ObservationRenderConfig, RepeatedStepConfig, StepCounters, StepStatus,
    StepSummary,
};

#[cfg(test)]
use bootstrap::resolve_display_aspect_ratio;

#[cfg(test)]
#[path = "tests/host_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests/step_accumulator_tests.rs"]
mod step_accumulator_tests;
