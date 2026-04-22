// rust/core/host/callbacks/video_state.rs
//! CallbackState video-frame capture and frame-buffer ownership.

use std::ffi::c_void;

use crate::core::host::hardware::HardwareRenderContext;
#[cfg(test)]
use crate::core::video::VideoFrame;
use crate::core::video::{capture_raw_frame, capture_raw_frame_into};

use super::CallbackState;

impl CallbackState {
    pub(in crate::core::host::callbacks) fn store_video_frame(
        &mut self,
        data: *const c_void,
        width: usize,
        height: usize,
        pitch: usize,
    ) {
        self.geometry = Some((width, height));
        if !self.capture_video {
            return;
        }

        if HardwareRenderContext::can_capture(data) {
            if let Some(hardware_render) = self.hardware_render.as_mut()
                && let Some(frame) = hardware_render.capture_frame(width, height)
            {
                self.raw_frame = None;
                self.frame = Some(frame);
                self.frame_serial += 1;
            }
            return;
        }

        let updated = match self.raw_frame.as_mut() {
            Some(raw_frame) => {
                capture_raw_frame_into(raw_frame, data, width, height, pitch, self.pixel_format)
            }
            None => false,
        };

        if updated {
            self.frame = None;
            self.frame_serial += 1;
            return;
        }

        if let Some(raw_frame) = capture_raw_frame(data, width, height, pitch, self.pixel_format) {
            self.raw_frame = Some(raw_frame);
            self.frame = None;
            self.frame_serial += 1;
        }
    }

    #[cfg(test)]
    pub(in crate::core::host::callbacks) fn set_frame_for_test_without_reset(
        &mut self,
        frame: VideoFrame,
    ) {
        self.geometry = Some((frame.width, frame.height));
        self.raw_frame = None;
        self.frame = Some(frame);
        self.frame_serial += 1;
    }
}
