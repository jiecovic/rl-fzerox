// rust/core/host/stdio.rs
use std::fs::OpenOptions;
use std::os::fd::AsRawFd;
use std::os::raw::c_int;
use std::sync::{Mutex, MutexGuard, OnceLock};

static STDIO_SILENCE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

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

        let _ = unsafe { libc::fflush(std::ptr::null_mut()) };

        let Ok(dev_null) = OpenOptions::new().write(true).open("/dev/null") else {
            return Self {
                _lock: lock,
                saved_stdout: -1,
                saved_stderr: -1,
            };
        };

        let saved_stdout = unsafe { libc::dup(libc::STDOUT_FILENO) };
        let saved_stderr = unsafe { libc::dup(libc::STDERR_FILENO) };

        if saved_stdout >= 0 {
            let _ = unsafe { libc::dup2(dev_null.as_raw_fd(), libc::STDOUT_FILENO) };
        }
        if saved_stderr >= 0 {
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
        let _ = unsafe { libc::fflush(std::ptr::null_mut()) };
        if self.saved_stdout >= 0 {
            let _ = unsafe { libc::dup2(self.saved_stdout, libc::STDOUT_FILENO) };
            let _ = unsafe { libc::close(self.saved_stdout) };
        }
        if self.saved_stderr >= 0 {
            let _ = unsafe { libc::dup2(self.saved_stderr, libc::STDERR_FILENO) };
            let _ = unsafe { libc::close(self.saved_stderr) };
        }
    }
}
