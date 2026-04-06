// rust/core/error.rs
use std::fmt::{Display, Formatter};
use std::path::PathBuf;

#[derive(Debug)]
pub enum CoreError {
    MissingCore(PathBuf),
    MissingRom(PathBuf),
    AlreadyClosed,
    ReadFile { path: PathBuf, message: String },
    InvalidPath { path: PathBuf },
    LoadLibrary { path: PathBuf, message: String },
    MissingSymbol { symbol: String, message: String },
    UnsupportedApiVersion { actual: u32, expected: u32 },
    LoadGameFailed { path: PathBuf },
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
            Self::ReadFile { path, message } => {
                write!(formatter, "Could not read '{}': {message}", path.display())
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
            Self::LoadGameFailed { path } => {
                write!(
                    formatter,
                    "Libretro core could not load ROM '{}'",
                    path.display()
                )
            }
            Self::NoFrameAvailable => {
                write!(formatter, "The emulator did not produce a video frame")
            }
        }
    }
}

impl std::error::Error for CoreError {}
