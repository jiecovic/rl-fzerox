// rust/core.rs
//! Native-side core integration for libretro loading, frame processing, and
//! F-Zero X telemetry extraction.

pub mod error;
pub mod game;
pub mod host;
pub mod libretro;
pub mod minimap;
pub mod observation;
pub mod rom;

pub use game::telemetry;
pub use host::{callbacks, input, options, stdio, video};
pub use libretro::{api, info, probe};
