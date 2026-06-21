// rust/core/host/runtime/control.rs
//! Controller and frame-stepping methods on the host runtime.

use crate::core::error::CoreError;
use crate::core::input::ControllerState;
use crate::core::stdio::with_silenced_stdio;

use super::host::Host;
use super::step::StepCounters;

impl Host {
    pub fn reset(&mut self) -> Result<(), CoreError> {
        self.ensure_open()?;
        self.restore_baseline()?;
        self.callbacks
            .set_controller_state(ControllerState::default());
        self.frame_index = 0;
        self.step_counters = StepCounters::default();
        self.spin_macro.reset();
        self.refresh_shape_from_frame();
        Ok(())
    }

    pub fn step_frames(&mut self, count: usize, capture_video: bool) -> Result<(), CoreError> {
        self.step_frames_impl(count, capture_video, false)
            .map(|_| ())
    }

    pub fn step_frames_with_audio(
        &mut self,
        count: usize,
        capture_video: bool,
    ) -> Result<Vec<i16>, CoreError> {
        self.step_frames_impl(count, capture_video, true)
    }

    fn step_frames_impl(
        &mut self,
        count: usize,
        capture_video: bool,
        capture_audio: bool,
    ) -> Result<Vec<i16>, CoreError> {
        self.ensure_open()?;
        self.callbacks.set_capture_video(capture_video);
        self.callbacks.set_capture_audio(capture_audio);
        self.callbacks.clear_audio_samples();
        let run_frames = || {
            for _ in 0..count {
                self.run_core_frame();
            }
        };
        with_silenced_stdio(run_frames);
        let audio_samples = self.callbacks.drain_audio_samples();
        self.callbacks.set_capture_audio(false);
        self.callbacks.set_capture_video(true);
        self.frame_index += count;
        self.refresh_shape_from_frame();
        if capture_video && !self.callbacks.has_frame() {
            return Err(CoreError::NoFrameAvailable);
        }
        Ok(audio_samples)
    }

    pub fn set_controller_state(
        &mut self,
        controller_state: ControllerState,
    ) -> Result<(), CoreError> {
        self.ensure_open()?;
        self.spin_macro.reset();
        self.callbacks.set_controller_state(controller_state);
        Ok(())
    }
}
