// rust/core/host/video.rs
//! Video/frame processing helpers used by the host runtime.
//!
//! The submodules separate raw capture, pixel conversion, render-plan
//! generation, and output processing.

mod capture;
mod convert;
mod plan;
mod process;
mod resize;
mod types;

pub use capture::{capture_raw_frame, capture_raw_frame_into};
pub use convert::decode_frame;
pub(crate) use convert::sample_rgb;
pub use plan::{
    ProcessedFramePlan, ProcessedFramePlanKey, build_processed_frame_plan, cropped_dimensions,
    display_size,
};
pub use process::{processed_frame, processed_frame_from_raw_into};
pub use resize::{resize_luma, resize_rgb};
pub use types::{PixelLayout, RawVideoFrame, VideoCrop, VideoFrame, VideoResizeFilter};

#[cfg(test)]
use convert::{convert_argb1555, convert_argb8888, convert_rgb565};
#[cfg(test)]
use process::processed_frame_from_raw;

#[cfg(test)]
#[path = "tests/video_tests.rs"]
mod tests;
