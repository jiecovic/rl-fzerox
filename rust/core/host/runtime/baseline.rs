// rust/core/host/runtime/baseline.rs
//! Baseline and save-state helpers for fast episode resets.

use std::path::Path;

use super::host::{BaselineKind, FRAME_WAIT_LIMIT, Host};
use crate::core::error::CoreError;

impl Host {
    pub(super) fn capture_startup_baseline(&mut self) -> Result<(), CoreError> {
        // Capture one canonical startup baseline up front. Every reset then
        // restores this snapshot instead of replaying startup frames again.
        self.initialize_running_state()?;
        self.capture_current_baseline(BaselineKind::Startup)
    }

    pub(super) fn load_baseline_from_file(
        &mut self,
        baseline_state_path: &Path,
    ) -> Result<(), CoreError> {
        // A savestate restore does not always advance the video callback on its
        // own, so we explicitly wait for a fresh frame before freezing the
        // custom baseline metadata around it.
        self.initialize_running_state()?;
        let baseline_state =
            std::fs::read(baseline_state_path).map_err(|error| CoreError::ReadFile {
                path: baseline_state_path.to_path_buf(),
                message: error.to_string(),
            })?;
        let baseline_serial = self.callbacks.frame_serial();
        self.restore_state_bytes(&baseline_state)?;
        if self.callbacks.frame_serial() == baseline_serial {
            self.run_until_frame(baseline_serial, FRAME_WAIT_LIMIT)?;
        }
        self.capture_current_baseline(BaselineKind::Custom)
    }

    fn initialize_running_state(&mut self) -> Result<(), CoreError> {
        let baseline_serial = self.callbacks.frame_serial();
        self.call_core(|core| unsafe {
            core.reset();
        });
        self.run_until_frame(baseline_serial, FRAME_WAIT_LIMIT)
    }

    fn capture_current_baseline(&mut self, baseline_kind: BaselineKind) -> Result<(), CoreError> {
        self.refresh_av_info();
        self.refresh_shape_from_frame();
        self.baseline_state = self.serialize_state()?;
        self.baseline_frame = Some(
            self.callbacks
                .frame()
                .cloned()
                .ok_or(CoreError::NoFrameAvailable)?,
        );
        self.baseline_kind = baseline_kind;
        Ok(())
    }

    pub(super) fn restore_baseline(&mut self) -> Result<(), CoreError> {
        let baseline_frame = self
            .baseline_frame
            .clone()
            .ok_or(CoreError::NoFrameAvailable)?;
        let baseline_state = self.baseline_state.clone();
        self.restore_state_bytes(&baseline_state)?;
        self.callbacks.set_frame(baseline_frame);
        Ok(())
    }

    fn restore_state_bytes(&mut self, state: &[u8]) -> Result<(), CoreError> {
        let restored =
            self.call_core(|core| unsafe { core.unserialize(state.as_ptr().cast(), state.len()) });
        if restored {
            Ok(())
        } else {
            Err(CoreError::UnserializeFailed)
        }
    }

    fn serialize_state(&mut self) -> Result<Vec<u8>, CoreError> {
        let size = self.call_core(|core| unsafe { core.serialize_size() });
        if size == 0 {
            return Err(CoreError::UnsupportedSaveState);
        }

        let mut state = vec![0_u8; size];
        let serialized = self
            .call_core(|core| unsafe { core.serialize(state.as_mut_ptr().cast(), state.len()) });
        if !serialized {
            return Err(CoreError::SerializeFailed);
        }
        Ok(state)
    }

    fn write_state_bytes(&self, path: &Path, state: &[u8]) -> Result<(), CoreError> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(|error| CoreError::CreateDirectory {
                path: parent.to_path_buf(),
                message: error.to_string(),
            })?;
        }

        std::fs::write(path, state).map_err(|error| CoreError::WriteFile {
            path: path.to_path_buf(),
            message: error.to_string(),
        })
    }

    pub(super) fn save_state_to_path(&mut self, path: &Path) -> Result<(), CoreError> {
        let state = self.serialize_state()?;
        self.write_state_bytes(path, &state)
    }

    pub(super) fn capture_current_as_baseline_to_path(
        &mut self,
        save_path: Option<&Path>,
    ) -> Result<(), CoreError> {
        // This powers the watch UI's "save current as baseline" flow and keeps
        // the in-memory reset snapshot in sync with the optional on-disk file.
        let state = self.serialize_state()?;
        if let Some(path) = save_path {
            self.write_state_bytes(path, &state)?;
        }
        self.baseline_state = state;
        self.baseline_frame = Some(
            self.callbacks
                .frame()
                .cloned()
                .ok_or(CoreError::NoFrameAvailable)?,
        );
        self.baseline_kind = BaselineKind::Custom;
        self.refresh_av_info();
        self.refresh_shape_from_frame();
        Ok(())
    }
}
