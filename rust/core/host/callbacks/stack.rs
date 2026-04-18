// rust/core/host/callbacks/stack.rs
//! Reusable stacked-observation buffering for callback-owned rendered frames.

use crate::core::error::CoreError;
use crate::core::observation::ObservationStackMode;
use crate::core::video::{ProcessedFramePlanKey, VideoCrop};

pub(crate) struct StackedObservationRequest {
    pub aspect_ratio: f64,
    pub target_width: usize,
    pub target_height: usize,
    pub rgb: bool,
    pub crop: VideoCrop,
    pub frame_stack: usize,
    pub stack_mode: ObservationStackMode,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct StackedObservationKey {
    pub render_plan: ProcessedFramePlanKey,
    pub frame_stack: usize,
    pub stack_mode: ObservationStackMode,
}

pub(super) struct StackedObservationBuffer {
    frame_len: usize,
    channels_per_pixel: usize,
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
    ) -> Self {
        debug_assert!(frame_stack > 0);
        debug_assert!(channels_per_pixel > 0);
        debug_assert_eq!(frame_len % channels_per_pixel, 0);
        let pixel_count = frame_len / channels_per_pixel;
        let output_channels = stack_mode.stacked_channels(channels_per_pixel, frame_stack);
        Self {
            frame_len,
            channels_per_pixel,
            frame_stack,
            stack_mode,
            frames: vec![0_u8; frame_len * frame_stack],
            bytes: vec![0_u8; pixel_count * output_channels],
            next_slot: 0,
            last_frame_serial: None,
        }
    }

    pub fn update(&mut self, frame: &[u8], frame_serial: u64) -> Result<(), CoreError> {
        if frame.len() != self.frame_len {
            return Err(CoreError::NoFrameAvailable);
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

        self.materialize();
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

    fn materialize(&mut self) {
        let pixel_count = self.frame_len / self.channels_per_pixel;

        if self.stack_mode == ObservationStackMode::Rgb && self.channels_per_pixel == 3 {
            for pixel_index in 0..pixel_count {
                let pixel_src = pixel_index * 3;
                let pixel_dst = pixel_index * 3 * self.frame_stack;
                for stack_index in 0..self.frame_stack {
                    let slot = (self.next_slot + stack_index) % self.frame_stack;
                    let src = (slot * self.frame_len) + pixel_src;
                    let dst = pixel_dst + (stack_index * 3);
                    self.bytes[dst] = self.frames[src];
                    self.bytes[dst + 1] = self.frames[src + 1];
                    self.bytes[dst + 2] = self.frames[src + 2];
                }
            }
            return;
        }

        if self.stack_mode == ObservationStackMode::RgbGray && self.channels_per_pixel == 3 {
            let output_channels = self.stack_mode.stacked_channels(3, self.frame_stack);
            for pixel_index in 0..pixel_count {
                let pixel_src = pixel_index * 3;
                let pixel_dst = pixel_index * output_channels;
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
            return;
        }

        for pixel_index in 0..pixel_count {
            let pixel_dst = pixel_index * self.frame_stack;
            for stack_index in 0..self.frame_stack {
                let slot = (self.next_slot + stack_index) % self.frame_stack;
                let src = (slot * self.frame_len) + pixel_index;
                self.bytes[pixel_dst + stack_index] = self.frames[src];
            }
        }
    }
}

fn rgb_to_luma(red: u8, green: u8, blue: u8) -> u8 {
    let weighted = (77 * u16::from(red)) + (150 * u16::from(green)) + (29 * u16::from(blue)) + 128;
    (weighted >> 8) as u8
}
