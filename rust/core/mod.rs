// rust/core/mod.rs
pub mod error;
pub mod game;
pub mod host;
pub mod libretro;

pub use game::telemetry;
pub use host::{callbacks, input, options, stdio, video};
pub use libretro::{api, info, probe};
