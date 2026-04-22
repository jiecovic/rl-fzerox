// rust/core/error.rs
//! Error types shared across the native libretro host, video pipeline, and
//! telemetry readers.

use std::fmt::{Display, Formatter};
use std::path::PathBuf;

/// Domain-specific failures raised by the native host and exposed to Python.
#[derive(Debug)]
pub enum CoreError {
    MissingCore(PathBuf),
    MissingRom(PathBuf),
    AlreadyClosed,
    UnsupportedSaveState,
    MemoryUnavailable {
        memory_id: u32,
    },
    MemoryOutOfRange {
        memory_id: u32,
        offset: usize,
        length: usize,
        available: usize,
    },
    SerializeFailed,
    UnserializeFailed,
    CreateDirectory {
        path: PathBuf,
        message: String,
    },
    ReadFile {
        path: PathBuf,
        message: String,
    },
    InvalidRomHeader {
        path: PathBuf,
    },
    UnsupportedRom {
        path: PathBuf,
        title: String,
        game_code: String,
        revision: u8,
        expected_title: &'static str,
        expected_game_code: &'static str,
        expected_revision: u8,
    },
    WriteFile {
        path: PathBuf,
        message: String,
    },
    InvalidPath {
        path: PathBuf,
    },
    LoadLibrary {
        path: PathBuf,
        message: String,
    },
    MissingSymbol {
        symbol: String,
        message: String,
    },
    UnsupportedApiVersion {
        actual: u32,
        expected: u32,
    },
    #[allow(dead_code)]
    InvalidStepRepeatCount {
        count: usize,
    },
    LoadGameFailed {
        path: PathBuf,
    },
    HardwareRenderFailed {
        message: String,
    },
    InvalidObservationPreset {
        name: String,
    },
    InvalidResizeFilter {
        name: String,
    },
    ResizeFailed {
        message: String,
    },
    InvalidVideoCrop {
        frame_width: usize,
        frame_height: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    },
    NoFrameAvailable,
}

impl Display for CoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingCore(path) => {
                write!(formatter, "Libretro core not found: {}", path.display())
            }
            Self::MissingRom(path) => {
                write!(formatter, "ROM not found: {}", path.display())
            }
            Self::AlreadyClosed => {
                write!(formatter, "The emulator host is already closed")
            }
            Self::UnsupportedSaveState => {
                write!(
                    formatter,
                    "The libretro core does not expose save-state support"
                )
            }
            Self::MemoryUnavailable { memory_id } => {
                write!(
                    formatter,
                    "The libretro core does not expose memory id {memory_id}"
                )
            }
            Self::MemoryOutOfRange {
                memory_id,
                offset,
                length,
                available,
            } => {
                write!(
                    formatter,
                    "Memory read is out of range for memory id {memory_id}: offset={offset}, length={length}, available={available}"
                )
            }
            Self::SerializeFailed => {
                write!(
                    formatter,
                    "The libretro core failed to serialize its current state"
                )
            }
            Self::UnserializeFailed => {
                write!(
                    formatter,
                    "The libretro core failed to restore a saved state"
                )
            }
            Self::CreateDirectory { path, message } => {
                write!(
                    formatter,
                    "Could not create directory '{}': {message}",
                    path.display()
                )
            }
            Self::ReadFile { path, message } => {
                write!(formatter, "Could not read '{}': {message}", path.display())
            }
            Self::InvalidRomHeader { path } => {
                write!(
                    formatter,
                    "Unsupported ROM '{}': not a recognized N64 ROM image",
                    path.display()
                )
            }
            Self::UnsupportedRom {
                path,
                title,
                game_code,
                revision,
                expected_title,
                expected_game_code,
                expected_revision,
            } => {
                write!(
                    formatter,
                    "Unsupported ROM '{}': detected title='{title}', game_code='{game_code}', revision={revision}; expected title='{expected_title}', game_code='{expected_game_code}', revision={expected_revision}. This project uses fixed RAM offsets for the US F-Zero X ROM.",
                    path.display()
                )
            }
            Self::WriteFile { path, message } => {
                write!(formatter, "Could not write '{}': {message}", path.display())
            }
            Self::InvalidPath { path } => {
                write!(
                    formatter,
                    "Path contains unsupported NUL byte: {}",
                    path.display()
                )
            }
            Self::LoadLibrary { path, message } => {
                write!(
                    formatter,
                    "Could not load libretro core '{}': {message}",
                    path.display()
                )
            }
            Self::MissingSymbol { symbol, message } => {
                write!(formatter, "Missing libretro symbol '{symbol}': {message}")
            }
            Self::UnsupportedApiVersion { actual, expected } => {
                write!(
                    formatter,
                    "Unsupported libretro API version {actual} (expected {expected})"
                )
            }
            Self::InvalidStepRepeatCount { count } => {
                write!(formatter, "Invalid repeated env-step frame count {count}")
            }
            Self::LoadGameFailed { path } => {
                write!(
                    formatter,
                    "Libretro core could not load ROM '{}'",
                    path.display()
                )
            }
            Self::HardwareRenderFailed { message } => {
                write!(formatter, "Hardware renderer setup failed: {message}")
            }
            Self::InvalidObservationPreset { name } => {
                write!(formatter, "Unknown observation preset '{name}'")
            }
            Self::InvalidResizeFilter { name } => {
                write!(formatter, "Unknown video resize filter '{name}'")
            }
            Self::ResizeFailed { message } => {
                write!(formatter, "Video resize failed: {message}")
            }
            Self::InvalidVideoCrop {
                frame_width,
                frame_height,
                top,
                bottom,
                left,
                right,
            } => {
                write!(
                    formatter,
                    "Invalid video crop for frame {frame_width}x{frame_height}: top={top}, bottom={bottom}, left={left}, right={right}"
                )
            }
            Self::NoFrameAvailable => {
                write!(formatter, "The emulator did not produce a video frame")
            }
        }
    }
}

impl std::error::Error for CoreError {}
