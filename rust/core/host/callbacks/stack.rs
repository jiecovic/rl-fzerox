// rust/core/host/callbacks/stack.rs
//! Reusable stacked-observation buffering for callback-owned rendered frames.

use crate::core::error::CoreError;
use crate::core::minimap::MinimapLayerRequest;
use crate::core::observation::ObservationStackMode;
use crate::core::video::{ProcessedFramePlanKey, VideoCrop, VideoResizeFilter};

pub(crate) struct StackedObservationRequest {
    pub aspect_ratio: f64,
    pub target_width: usize,
    pub target_height: usize,
    pub rgb: bool,
    pub crop: VideoCrop,
    pub resize_filter: VideoResizeFilter,
    pub frame_stack: usize,
    pub stack_mode: ObservationStackMode,
    pub minimap_layer: Option<MinimapLayerRequest>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct StackedObservationKey {
    pub render_plan: ProcessedFramePlanKey,
    pub frame_stack: usize,
    pub stack_mode: ObservationStackMode,
    pub extra_channels_per_pixel: usize,
}

pub(super) struct StackedObservationBuffer {
    frame_len: usize,
    channels_per_pixel: usize,
    output_channels_per_pixel: usize,
    total_channels_per_pixel: usize,
    frame_stack: usize,
    stack_mode: ObservationStackMode,
    frames: Vec<u8>,
    bytes: Vec<u8>,
    next_slot: usize,
    last_frame_serial: Option<u64>,
}

impl StackedObservationBuffer {
    pub fn new(
        frame_len: usize,
        frame_stack: usize,
        channels_per_pixel: usize,
        stack_mode: ObservationStackMode,
        extra_channels_per_pixel: usize,
    ) -> Self {
        debug_assert!(frame_stack > 0);
        debug_assert!(channels_per_pixel > 0);
        debug_assert_eq!(frame_len % channels_per_pixel, 0);
        let pixel_count = frame_len / channels_per_pixel;
        let output_channels = stack_mode.stacked_channels(channels_per_pixel, frame_stack);
        let total_channels = output_channels + extra_channels_per_pixel;
        Self {
            frame_len,
            channels_per_pixel,
            output_channels_per_pixel: output_channels,
            total_channels_per_pixel: total_channels,
            frame_stack,
            stack_mode,
            frames: vec![0_u8; frame_len * frame_stack],
            bytes: vec![0_u8; pixel_count * total_channels],
            next_slot: 0,
            last_frame_serial: None,
        }
    }

    pub fn update(
        &mut self,
        frame: &[u8],
        frame_serial: u64,
        extra_frame: Option<&[u8]>,
    ) -> Result<(), CoreError> {
        if frame.len() != self.frame_len {
            return Err(CoreError::NoFrameAvailable);
        }
        if let Some(extra_frame) = extra_frame {
            let pixel_count = self.frame_len / self.channels_per_pixel;
            if extra_frame.len() != pixel_count {
                return Err(CoreError::NoFrameAvailable);
            }
            if self.total_channels_per_pixel != self.output_channels_per_pixel + 1 {
                return Err(CoreError::NoFrameAvailable);
            }
        }
        if self.last_frame_serial == Some(frame_serial) {
            return Ok(());
        }

        if self.last_frame_serial.is_none() {
            for slot_frame in self.frames.chunks_exact_mut(self.frame_len) {
                slot_frame.copy_from_slice(frame);
            }
        } else {
            let slot_start = self.next_slot * self.frame_len;
            self.frames[slot_start..slot_start + self.frame_len].copy_from_slice(frame);
            self.next_slot = (self.next_slot + 1) % self.frame_stack;
        }

        self.materialize(extra_frame);
        self.last_frame_serial = Some(frame_serial);
        Ok(())
    }

    pub fn clear(&mut self) {
        self.next_slot = 0;
        self.last_frame_serial = None;
    }

    pub fn as_slice(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    fn materialize(&mut self, extra_frame: Option<&[u8]>) {
        let pixel_count = self.frame_len / self.channels_per_pixel;
        let total_channels = self.total_channels_per_pixel;

        if self.stack_mode == ObservationStackMode::Rgb && self.channels_per_pixel == 3 {
            for pixel_index in 0..pixel_count {
                let pixel_src = pixel_index * 3;
                let pixel_dst = pixel_index * total_channels;
                for stack_index in 0..self.frame_stack {
                    let slot = (self.next_slot + stack_index) % self.frame_stack;
                    let src = (slot * self.frame_len) + pixel_src;
                    let dst = pixel_dst + (stack_index * 3);
                    self.bytes[dst] = self.frames[src];
                    self.bytes[dst + 1] = self.frames[src + 1];
                    self.bytes[dst + 2] = self.frames[src + 2];
                }
            }
            self.write_extra_frame(extra_frame, pixel_count);
            return;
        }

        if self.stack_mode == ObservationStackMode::RgbGray && self.channels_per_pixel == 3 {
            for pixel_index in 0..pixel_count {
                let pixel_src = pixel_index * 3;
                let pixel_dst = pixel_index * total_channels;
                for stack_index in 0..self.frame_stack {
                    let slot = (self.next_slot + stack_index) % self.frame_stack;
                    let src = (slot * self.frame_len) + pixel_src;
                    let dst = pixel_dst + stack_index;
                    if stack_index + 1 == self.frame_stack {
                        self.bytes[dst] = self.frames[src];
                        self.bytes[dst + 1] = self.frames[src + 1];
                        self.bytes[dst + 2] = self.frames[src + 2];
                    } else {
                        self.bytes[dst] = rgb_to_luma(
                            self.frames[src],
                            self.frames[src + 1],
                            self.frames[src + 2],
                        );
                    }
                }
            }
            self.write_extra_frame(extra_frame, pixel_count);
            return;
        }

        if self.stack_mode == ObservationStackMode::Gray && self.channels_per_pixel == 3 {
            for pixel_index in 0..pixel_count {
                let pixel_src = pixel_index * 3;
                let pixel_dst = pixel_index * total_channels;
                for stack_index in 0..self.frame_stack {
                    let slot = (self.next_slot + stack_index) % self.frame_stack;
                    let src = (slot * self.frame_len) + pixel_src;
                    self.bytes[pixel_dst + stack_index] =
                        rgb_to_luma(self.frames[src], self.frames[src + 1], self.frames[src + 2]);
                }
            }
            self.write_extra_frame(extra_frame, pixel_count);
            return;
        }

        if self.stack_mode == ObservationStackMode::LumaChroma && self.channels_per_pixel == 3 {
            for pixel_index in 0..pixel_count {
                let pixel_src = pixel_index * 3;
                let pixel_dst = pixel_index * total_channels;
                for stack_index in 0..self.frame_stack {
                    let slot = (self.next_slot + stack_index) % self.frame_stack;
                    let src = (slot * self.frame_len) + pixel_src;
                    let dst = pixel_dst + (stack_index * 2);
                    let red = self.frames[src];
                    let green = self.frames[src + 1];
                    let blue = self.frames[src + 2];
                    self.bytes[dst] = rgb_to_luma(red, green, blue);
                    self.bytes[dst + 1] = rgb_to_yellow_purple_chroma(red, green, blue);
                }
            }
            self.write_extra_frame(extra_frame, pixel_count);
            return;
        }

        for pixel_index in 0..pixel_count {
            let pixel_dst = pixel_index * total_channels;
            for stack_index in 0..self.frame_stack {
                let slot = (self.next_slot + stack_index) % self.frame_stack;
                let src = (slot * self.frame_len) + pixel_index;
                self.bytes[pixel_dst + stack_index] = self.frames[src];
            }
        }
        self.write_extra_frame(extra_frame, pixel_count);
    }

    fn write_extra_frame(&mut self, extra_frame: Option<&[u8]>, pixel_count: usize) {
        let Some(extra_frame) = extra_frame else {
            return;
        };
        for (pixel_index, value) in extra_frame.iter().enumerate().take(pixel_count) {
            let dst =
                (pixel_index * self.total_channels_per_pixel) + self.output_channels_per_pixel;
            self.bytes[dst] = *value;
        }
    }
}

fn rgb_to_luma(red: u8, green: u8, blue: u8) -> u8 {
    let weighted = (77 * u16::from(red)) + (150 * u16::from(green)) + (29 * u16::from(blue)) + 128;
    (weighted >> 8) as u8
}

fn rgb_to_yellow_purple_chroma(red: u8, green: u8, blue: u8) -> u8 {
    let opponent = (2 * i16::from(green)) - i16::from(red) - i16::from(blue);
    (128 + (opponent / 4)).clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use super::StackedObservationBuffer;
    use crate::core::observation::ObservationStackMode;

    #[test]
    fn minimap_extra_channel_is_appended_after_existing_stack_channels() {
        let mut stack = StackedObservationBuffer::new(6, 2, 3, ObservationStackMode::RgbGray, 1);

        stack
            .update(&[10, 20, 30, 40, 50, 60], 1, Some(&[90, 120]))
            .expect("initial stack should accept minimap layer");

        assert_eq!(stack.as_slice(), &[18, 10, 20, 30, 90, 48, 40, 50, 60, 120]);

        stack
            .update(&[70, 80, 90, 100, 110, 120], 2, Some(&[130, 160]))
            .expect("next stack should update minimap layer");

        assert_eq!(
            stack.as_slice(),
            &[18, 70, 80, 90, 130, 48, 100, 110, 120, 160]
        );
    }

    #[test]
    fn gray_stack_encodes_all_frames_as_luma() {
        let mut stack = StackedObservationBuffer::new(6, 2, 3, ObservationStackMode::Gray, 1);

        stack
            .update(&[10, 20, 30, 40, 50, 60], 1, Some(&[90, 120]))
            .expect("initial stack should accept minimap layer");

        assert_eq!(stack.as_slice(), &[18, 18, 90, 48, 48, 120]);

        stack
            .update(&[70, 80, 90, 100, 110, 120], 2, Some(&[130, 160]))
            .expect("next stack should update grayscale stack");

        assert_eq!(stack.as_slice(), &[18, 78, 130, 48, 108, 160]);
    }

    #[test]
    fn luma_chroma_stack_preserves_yellow_purple_cue() {
        let mut stack = StackedObservationBuffer::new(6, 2, 3, ObservationStackMode::LumaChroma, 1);

        stack
            .update(&[255, 255, 0, 255, 0, 255], 1, Some(&[90, 120]))
            .expect("initial stack should accept luma-chroma stack");

        assert_eq!(
            stack.as_slice(),
            &[226, 191, 226, 191, 90, 106, 1, 106, 1, 120]
        );
    }
}
