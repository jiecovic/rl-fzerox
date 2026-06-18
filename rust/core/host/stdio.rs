// rust/core/host/stdio.rs
//! Helpers for temporarily silencing noisy core stdout/stderr during libretro
//! init/load paths.

use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::os::raw::c_int;
use std::sync::{Mutex, MutexGuard, OnceLock};

static STDIO_SILENCE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Run one operation while temporarily redirecting stdout/stderr to `/dev/null`.
pub fn with_silenced_stdio<T>(operation: impl FnOnce() -> T) -> T {
    let _guard = StdioSilencer::acquire();
    operation()
}

struct StdioSilencer {
    _lock: MutexGuard<'static, ()>,
    saved_stdout: c_int,
    saved_stderr: c_int,
}

impl StdioSilencer {
    fn acquire() -> Self {
        let lock = STDIO_SILENCE_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // SAFETY: Passing a null stream asks libc to flush all open output streams.
        let _ = unsafe { libc::fflush(std::ptr::null_mut()) };

        let Ok(dev_null) = OpenOptions::new().write(true).open("/dev/null") else {
            return Self {
                _lock: lock,
                saved_stdout: -1,
                saved_stderr: -1,
            };
        };

        // SAFETY: Duplicating process stdio file descriptors is safe; failures
        // are reported as negative fds and handled below.
        let saved_stdout = unsafe { libc::dup(libc::STDOUT_FILENO) };
        // SAFETY: Same as stdout duplication above.
        let saved_stderr = unsafe { libc::dup(libc::STDERR_FILENO) };

        if saved_stdout >= 0 {
            // SAFETY: `dev_null` is an open fd and stdout is a valid process fd.
            let _ = unsafe { libc::dup2(dev_null.as_raw_fd(), libc::STDOUT_FILENO) };
        }
        if saved_stderr >= 0 {
            // SAFETY: `dev_null` is an open fd and stderr is a valid process fd.
            let _ = unsafe { libc::dup2(dev_null.as_raw_fd(), libc::STDERR_FILENO) };
        }

        Self {
            _lock: lock,
            saved_stdout,
            saved_stderr,
        }
    }
}

impl Drop for StdioSilencer {
    fn drop(&mut self) {
        // SAFETY: Passing a null stream asks libc to flush all open output streams.
        let _ = unsafe { libc::fflush(std::ptr::null_mut()) };
        if self.saved_stdout >= 0 {
            // SAFETY: `saved_stdout` was returned by `dup` and is restored once.
            let _ = unsafe { libc::dup2(self.saved_stdout, libc::STDOUT_FILENO) };
            // SAFETY: `saved_stdout` is no longer needed after restore.
            let _ = unsafe { libc::close(self.saved_stdout) };
        }
        if self.saved_stderr >= 0 {
            // SAFETY: `saved_stderr` was returned by `dup` and is restored once.
            let _ = unsafe { libc::dup2(self.saved_stderr, libc::STDERR_FILENO) };
            // SAFETY: `saved_stderr` is no longer needed after restore.
            let _ = unsafe { libc::close(self.saved_stderr) };
        }
    }
}
