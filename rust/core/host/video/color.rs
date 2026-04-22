// rust/core/host/video/color.rs
//! Color-channel transforms used by policy-frame preprocessing.

#[inline]
pub(crate) fn rgb_to_luma_vec(rgb: &[u8]) -> Vec<u8> {
    let mut luma = vec![0_u8; rgb.len() / 3];
    rgb_to_luma_into(rgb, &mut luma);
    luma
}

#[inline]
pub(crate) fn rgb_to_luma_into(rgb: &[u8], luma: &mut [u8]) {
    debug_assert_eq!(rgb.len(), luma.len() * 3);
    for (pixel, output) in rgb.chunks_exact(3).zip(luma.iter_mut()) {
        *output = rgb_to_luma(pixel[0], pixel[1], pixel[2]);
    }
}

#[inline]
pub(crate) fn rgb_to_luma_in_place(rgb: &mut Vec<u8>) {
    let pixel_count = rgb.len() / 3;
    for pixel_index in 0..pixel_count {
        let rgb_index = pixel_index * 3;
        rgb[pixel_index] = rgb_to_luma(rgb[rgb_index], rgb[rgb_index + 1], rgb[rgb_index + 2]);
    }
    rgb.truncate(pixel_count);
}

#[inline(always)]
pub(crate) fn rgb_to_luma(red: u8, green: u8, blue: u8) -> u8 {
    let weighted = (77 * u16::from(red)) + (150 * u16::from(green)) + (29 * u16::from(blue)) + 128;
    (weighted >> 8) as u8
}

#[inline(always)]
pub(crate) fn rgb_to_yellow_purple_chroma(red: u8, green: u8, blue: u8) -> u8 {
    let opponent = (2 * i16::from(green)) - i16::from(red) - i16::from(blue);
    (128 + (opponent / 4)).clamp(0, 255) as u8
}
