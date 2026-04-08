// rust/core/host/runtime.rs
//! `Host` runtime facade over one loaded libretro core instance.
//!
//! The helper submodules split the implementation by responsibility:
//! bootstrap/lifecycle, baseline-state management, memory access, and the
//! main `Host` façade itself.

mod baseline;
mod bootstrap;
mod host;
mod memory;

pub use host::Host;

#[cfg(test)]
use bootstrap::resolve_display_aspect_ratio;

#[cfg(test)]
#[path = "tests/host_tests.rs"]
mod tests;
