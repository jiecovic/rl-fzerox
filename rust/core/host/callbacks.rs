// rust/core/host/callbacks.rs
//! Libretro callback frontend wiring plus callback-owned state.
//!
//! The submodules split responsibilities between environment command handling,
//! callback entry points, thread-local activation, and the state container.

mod environment;
mod guard;
mod io;
mod stack;
mod state;
mod util;

pub use guard::CallbackGuard;
pub use io::{
    audio_sample_batch_callback, audio_sample_callback, environment_callback, input_device,
    input_poll_callback, input_state_callback, video_refresh_callback,
};
pub(crate) use stack::StackedObservationRequest;
pub use state::CallbackState;

#[cfg(test)]
use util::runtime_root_for_core;

#[cfg(test)]
#[path = "tests/callbacks_tests.rs"]
mod tests;
