// rust/core/error.rs
//! Error types shared across the native libretro host, video pipeline, and
//! telemetry readers.

use std::path::PathBuf;

use thiserror::Error;

/// Domain-specific failures raised by the native host and exposed to Python.
#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Libretro core not found: {0}")]
    MissingCore(PathBuf),
    #[error("ROM not found: {0}")]
    MissingRom(PathBuf),
    #[error("The emulator host is already closed")]
    AlreadyClosed,
    #[error("The libretro core does not expose save-state support")]
    UnsupportedSaveState,
    #[error("The libretro core does not expose memory id {memory_id}")]
    MemoryUnavailable { memory_id: u32 },
    #[error(
        "Memory range is out of bounds for memory id {memory_id}: offset={offset}, length={length}, available={available}"
    )]
    MemoryOutOfRange {
        memory_id: u32,
        offset: usize,
        length: usize,
        available: usize,
    },
    #[error("Save RAM buffer has wrong size: expected {expected} bytes, got {actual}")]
    InvalidSaveRamSize { expected: usize, actual: usize },
    #[error("The libretro core failed to serialize its current state")]
    SerializeFailed,
    #[error("The libretro core failed to restore a saved state")]
    UnserializeFailed,
    #[error("Could not create directory '{path}': {source}")]
    CreateDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Could not read '{path}': {source}")]
    ReadFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Unsupported ROM '{path}': not a recognized N64 ROM image")]
    InvalidRomHeader { path: PathBuf },
    #[error(
        "Unsupported ROM '{path}': detected title='{title}', game_code='{game_code}', revision={revision}; expected title='{expected_title}', game_code='{expected_game_code}', revision={expected_revision}. This project uses fixed RAM offsets for the US F-Zero X ROM."
    )]
    UnsupportedRom {
        path: PathBuf,
        title: String,
        game_code: String,
        revision: u8,
        expected_title: &'static str,
        expected_game_code: &'static str,
        expected_revision: u8,
    },
    #[error("Could not write '{path}': {source}")]
    WriteFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Path contains unsupported NUL byte: {path}")]
    InvalidPath { path: PathBuf },
    #[error("Could not load libretro core '{path}': {source}")]
    LoadLibrary {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },
    #[error("Missing libretro symbol '{symbol}': {source}")]
    MissingSymbol {
        symbol: String,
        #[source]
        source: libloading::Error,
    },
    #[error("Unsupported libretro API version {actual} (expected {expected})")]
    UnsupportedApiVersion { actual: u32, expected: u32 },
    #[error("Invalid repeated env-step frame count {count}")]
    InvalidStepRepeatCount { count: usize },
    #[error("Libretro core could not load ROM '{path}'")]
    LoadGameFailed { path: PathBuf },
    #[error("Hardware renderer setup failed: {0}")]
    HardwareRenderFailed(#[from] HardwareRenderError),
    #[error("Unknown observation preset '{name}'")]
    InvalidObservationPreset { name: String },
    #[error("Unknown video resize filter '{name}'")]
    InvalidResizeFilter { name: String },
    #[error("Video resize failed: {message}")]
    ResizeFailed { message: String },
    #[error(
        "Invalid video crop for frame {frame_width}x{frame_height}: top={top}, bottom={bottom}, left={left}, right={right}"
    )]
    InvalidVideoCrop {
        frame_width: usize,
        frame_height: usize,
        top: usize,
        bottom: usize,
        left: usize,
        right: usize,
    },
    #[error(
        "Video buffer range is out of bounds: offset={offset}, length={length}, available={available}"
    )]
    InvalidVideoBuffer {
        offset: usize,
        length: usize,
        available: usize,
    },
    #[error("Invalid F-Zero X race-start setup: {message}")]
    InvalidRaceStartSetup { message: String },
    #[error("The emulator did not produce a video frame")]
    NoFrameAvailable,
}

/// Failures specific to the native OpenGL/EGL frontend used by hardware renderers.
#[derive(Clone, Debug, Error)]
pub enum HardwareRenderError {
    #[error("hardware context {context_type} is not supported by this host yet")]
    UnsupportedContext { context_type: String },
    #[error("SET_HW_RENDER received null callback")]
    NullCallback,
    #[error("could not load {library}: {message}")]
    LoadLibrary {
        library: &'static str,
        message: String,
    },
    #[error("missing {library} symbol '{symbol}': {message}")]
    MissingSymbol {
        library: &'static str,
        symbol: String,
        message: String,
    },
    #[error("invalid C symbol name '{symbol}': {message}")]
    InvalidSymbolName { symbol: String, message: String },
    #[error("missing GL symbol '{symbol}'")]
    MissingGlSymbol { symbol: String },
    #[error("{call} returned null")]
    NullHandle { call: &'static str },
    #[error("{call} returned no usable EGL config")]
    NoConfig { call: &'static str },
    #[error("{call} failed with {egl_error}")]
    EglCallFailed {
        call: &'static str,
        egl_error: String,
    },
}
