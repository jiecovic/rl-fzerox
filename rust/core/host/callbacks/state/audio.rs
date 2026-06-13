// rust/core/host/callbacks/state/audio.rs
//! CallbackState audio sample capture.

use std::slice;

use super::CallbackState;

impl CallbackState {
    pub(crate) fn set_capture_audio(&mut self, capture_audio: bool) {
        self.capture_audio = capture_audio;
        if !capture_audio {
            self.audio_samples.clear();
        }
    }

    pub(crate) fn clear_audio_samples(&mut self) {
        self.audio_samples.clear();
    }

    pub(crate) fn drain_audio_samples(&mut self) -> Vec<i16> {
        std::mem::take(&mut self.audio_samples)
    }

    pub(in crate::core::host::callbacks) fn store_audio_sample(&mut self, left: i16, right: i16) {
        if !self.capture_audio {
            return;
        }
        self.audio_samples.extend_from_slice(&[left, right]);
    }

    pub(in crate::core::host::callbacks) fn store_audio_sample_batch(
        &mut self,
        data: *const i16,
        frames: usize,
    ) -> usize {
        if !self.capture_audio {
            return frames;
        }
        let Some(sample_count) = frames.checked_mul(2) else {
            return 0;
        };
        if sample_count == 0 {
            return frames;
        }
        if data.is_null() {
            return 0;
        }
        let samples = unsafe { slice::from_raw_parts(data, sample_count) };
        self.audio_samples.extend_from_slice(samples);
        frames
    }
}
