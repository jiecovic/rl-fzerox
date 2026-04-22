// rust/core/host/video/types.rs
//! Shared video/frame data structures.

/// Fully decoded RGB frame owned by the frontend.
#[derive(Clone, Debug)]
pub struct VideoFrame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// Raw framebuffer bytes captured from the libretro video callback.
#[derive(Clone, Debug)]
pub struct RawVideoFrame {
    pub width: usize,
    pub height: usize,
    pub pitch: usize,
    pub pixel_layout: PixelLayout,
    pub bytes: Vec<u8>,
}

/// Pixel layouts the libretro core may hand to the frontend.
#[derive(Clone, Copy, Debug)]
pub enum PixelLayout {
    Argb1555,
    Argb8888,
    Rgb565,
}

/// Resize filter used by policy observation rendering.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum VideoResizeFilter {
    Nearest,
    Bilinear,
}

impl VideoResizeFilter {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "nearest" => Some(Self::Nearest),
            "bilinear" => Some(Self::Bilinear),
            _ => None,
        }
    }
}

/// Number of pixels trimmed from each side before observation/display resize.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub struct VideoCrop {
    pub top: usize,
    pub bottom: usize,
    pub left: usize,
    pub right: usize,
}
