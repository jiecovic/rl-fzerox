// rust/core/host/callbacks/guard.rs
//! Thread-local activation of the current callback state while a libretro core
//! call is in flight.

use std::cell::Cell;
use std::ptr;

use crate::core::callbacks::CallbackState;

thread_local! {
    static ACTIVE_STATE: Cell<*mut CallbackState> = const { Cell::new(ptr::null_mut()) };
}

pub struct CallbackGuard {
    previous: *mut CallbackState,
}

impl CallbackGuard {
    /// Install the active callback state for the duration of one core call.
    pub fn activate(state: *mut CallbackState) -> Self {
        let previous = ACTIVE_STATE.with(|slot| {
            let previous = slot.get();
            slot.set(state);
            previous
        });
        Self { previous }
    }
}

impl Drop for CallbackGuard {
    fn drop(&mut self) {
        ACTIVE_STATE.with(|slot| slot.set(self.previous));
    }
}

pub(super) fn with_state_mut<R>(callback: impl FnOnce(&mut CallbackState) -> R) -> Option<R> {
    ACTIVE_STATE.with(|slot| {
        let state = slot.get();
        if state.is_null() {
            return None;
        }

        // SAFETY: The active callback state pointer is set by CallbackGuard for the
        // duration of a libretro call into this thread. The pointed-to state lives in
        // the owning Host and outlives the callback invocation.
        Some(callback(unsafe { &mut *state }))
    })
}
