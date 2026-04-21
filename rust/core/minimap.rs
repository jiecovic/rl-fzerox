// rust/core/minimap.rs
//! Optional minimap observation layer rendered from raw emulator frames.

use crate::core::error::CoreError;
use crate::core::observation::ObservationCropProfile;
use crate::core::video::{RawVideoFrame, VideoFrame, sample_rgb};

const COURSE_COUNT: usize = 24;
const MARKER_HOLD_SAMPLES: u8 = 4;
const PLAYER_MARKER_LUMA: u8 = 192;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MinimapRoi {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MinimapMaskSet {
    roi: MinimapRoi,
    masks: [&'static [u8]; COURSE_COUNT],
}

const GLIDEN64_ROI: MinimapRoi = MinimapRoi {
    x: 235,
    y: 133,
    width: 58,
    height: 61,
};

const ANGRYLION_ROI: MinimapRoi = MinimapRoi {
    x: 470,
    y: 134,
    width: 116,
    height: 60,
};

const GLIDEN64_MASKS: [&[u8]; COURSE_COUNT] = [
    include_bytes!("minimap/masks/gliden64/00_jack_mute_city.bin"),
    include_bytes!("minimap/masks/gliden64/01_jack_silence.bin"),
    include_bytes!("minimap/masks/gliden64/02_jack_sand_ocean.bin"),
    include_bytes!("minimap/masks/gliden64/03_jack_devils_forest.bin"),
    include_bytes!("minimap/masks/gliden64/04_jack_big_blue.bin"),
    include_bytes!("minimap/masks/gliden64/05_jack_port_town.bin"),
    include_bytes!("minimap/masks/gliden64/06_queen_sector_alpha.bin"),
    include_bytes!("minimap/masks/gliden64/07_queen_red_canyon.bin"),
    include_bytes!("minimap/masks/gliden64/08_queen_devils_forest_2.bin"),
    include_bytes!("minimap/masks/gliden64/09_queen_mute_city_2.bin"),
    include_bytes!("minimap/masks/gliden64/10_queen_big_blue_2.bin"),
    include_bytes!("minimap/masks/gliden64/11_queen_white_land.bin"),
    include_bytes!("minimap/masks/gliden64/12_king_fire_field.bin"),
    include_bytes!("minimap/masks/gliden64/13_king_silence_2.bin"),
    include_bytes!("minimap/masks/gliden64/14_king_sector_beta.bin"),
    include_bytes!("minimap/masks/gliden64/15_king_red_canyon_2.bin"),
    include_bytes!("minimap/masks/gliden64/16_king_white_land_2.bin"),
    include_bytes!("minimap/masks/gliden64/17_king_mute_city_3.bin"),
    include_bytes!("minimap/masks/gliden64/18_joker_rainbow_road.bin"),
    include_bytes!("minimap/masks/gliden64/19_joker_devils_forest_3.bin"),
    include_bytes!("minimap/masks/gliden64/20_joker_space_plant.bin"),
    include_bytes!("minimap/masks/gliden64/21_joker_sand_ocean_2.bin"),
    include_bytes!("minimap/masks/gliden64/22_joker_port_town_2.bin"),
    include_bytes!("minimap/masks/gliden64/23_joker_big_hand.bin"),
];

const ANGRYLION_MASKS: [&[u8]; COURSE_COUNT] = [
    include_bytes!("minimap/masks/angrylion/00_jack_mute_city.bin"),
    include_bytes!("minimap/masks/angrylion/01_jack_silence.bin"),
    include_bytes!("minimap/masks/angrylion/02_jack_sand_ocean.bin"),
    include_bytes!("minimap/masks/angrylion/03_jack_devils_forest.bin"),
    include_bytes!("minimap/masks/angrylion/04_jack_big_blue.bin"),
    include_bytes!("minimap/masks/angrylion/05_jack_port_town.bin"),
    include_bytes!("minimap/masks/angrylion/06_queen_sector_alpha.bin"),
    include_bytes!("minimap/masks/angrylion/07_queen_red_canyon.bin"),
    include_bytes!("minimap/masks/angrylion/08_queen_devils_forest_2.bin"),
    include_bytes!("minimap/masks/angrylion/09_queen_mute_city_2.bin"),
    include_bytes!("minimap/masks/angrylion/10_queen_big_blue_2.bin"),
    include_bytes!("minimap/masks/angrylion/11_queen_white_land.bin"),
    include_bytes!("minimap/masks/angrylion/12_king_fire_field.bin"),
    include_bytes!("minimap/masks/angrylion/13_king_silence_2.bin"),
    include_bytes!("minimap/masks/angrylion/14_king_sector_beta.bin"),
    include_bytes!("minimap/masks/angrylion/15_king_red_canyon_2.bin"),
    include_bytes!("minimap/masks/angrylion/16_king_white_land_2.bin"),
    include_bytes!("minimap/masks/angrylion/17_king_mute_city_3.bin"),
    include_bytes!("minimap/masks/angrylion/18_joker_rainbow_road.bin"),
    include_bytes!("minimap/masks/angrylion/19_joker_devils_forest_3.bin"),
    include_bytes!("minimap/masks/angrylion/20_joker_space_plant.bin"),
    include_bytes!("minimap/masks/angrylion/21_joker_sand_ocean_2.bin"),
    include_bytes!("minimap/masks/angrylion/22_joker_port_town_2.bin"),
    include_bytes!("minimap/masks/angrylion/23_joker_big_hand.bin"),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MinimapLayerRequest {
    pub crop_profile: ObservationCropProfile,
    pub course_index: usize,
    pub target_width: usize,
    pub target_height: usize,
}

#[derive(Debug, Default)]
pub(crate) struct MinimapLayerRenderer {
    marker_hold: MinimapMarkerHold,
    marker_scratch: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MinimapLayerKey {
    crop_profile: ObservationCropProfile,
    course_index: usize,
    target_width: usize,
    target_height: usize,
}

#[derive(Debug, Default)]
struct MinimapMarkerHold {
    key: Option<MinimapLayerKey>,
    marker_layer: Vec<u8>,
    missing_samples: u8,
}

impl MinimapLayerRenderer {
    pub fn render_from_raw_into(
        &mut self,
        frame: &RawVideoFrame,
        request: MinimapLayerRequest,
        output: &mut Vec<u8>,
    ) -> Result<(), CoreError> {
        let Some(mask_set) = mask_set(request.crop_profile) else {
            self.clear();
            return write_zero_layer(request, output);
        };
        let Some(mask) = mask_set.masks.get(request.course_index).copied() else {
            self.clear();
            return write_zero_layer(request, output);
        };
        self.render_with_marker_hold(request, mask_set.roi, mask, output, |x, y| {
            sample_rgb(frame, x, y)
        })
    }

    pub fn render_from_frame_into(
        &mut self,
        frame: &VideoFrame,
        request: MinimapLayerRequest,
        output: &mut Vec<u8>,
    ) -> Result<(), CoreError> {
        let Some(mask_set) = mask_set(request.crop_profile) else {
            self.clear();
            return write_zero_layer(request, output);
        };
        let Some(mask) = mask_set.masks.get(request.course_index).copied() else {
            self.clear();
            return write_zero_layer(request, output);
        };
        self.render_with_marker_hold(request, mask_set.roi, mask, output, |x, y| {
            let index = y.checked_mul(frame.width)?.checked_add(x)?.checked_mul(3)?;
            let pixel = frame.rgb.get(index..index + 3)?;
            Some([pixel[0], pixel[1], pixel[2]])
        })
    }

    pub fn clear(&mut self) {
        self.marker_hold.clear();
        self.marker_scratch.clear();
    }

    fn render_with_marker_hold(
        &mut self,
        request: MinimapLayerRequest,
        roi: MinimapRoi,
        mask: &[u8],
        output: &mut Vec<u8>,
        sample: impl FnMut(usize, usize) -> Option<[u8; 3]>,
    ) -> Result<(), CoreError> {
        self.marker_hold.ensure_key(request);
        let marker_count = render_layer_into(
            request,
            roi,
            mask,
            output,
            Some(&mut self.marker_scratch),
            sample,
        )?;
        self.marker_hold.update(marker_count, &self.marker_scratch);
        self.marker_hold.overlay(output);
        Ok(())
    }
}

impl MinimapMarkerHold {
    fn ensure_key(&mut self, request: MinimapLayerRequest) {
        let key = MinimapLayerKey {
            crop_profile: request.crop_profile,
            course_index: request.course_index,
            target_width: request.target_width,
            target_height: request.target_height,
        };
        if self.key != Some(key) {
            self.key = Some(key);
            self.marker_layer.clear();
            self.missing_samples = MARKER_HOLD_SAMPLES;
        }
    }

    fn update(&mut self, marker_count: usize, current_marker: &[u8]) {
        if marker_count > 0 {
            self.marker_layer.clear();
            self.marker_layer.extend_from_slice(current_marker);
            self.missing_samples = 0;
            return;
        }
        if self.marker_layer.is_empty() {
            return;
        }
        self.missing_samples = self.missing_samples.saturating_add(1);
        if self.missing_samples > MARKER_HOLD_SAMPLES {
            self.marker_layer.clear();
        }
    }

    fn overlay(&self, output: &mut [u8]) {
        if self.marker_layer.len() != output.len() {
            return;
        }
        for (value, marker) in output.iter_mut().zip(self.marker_layer.iter()) {
            if *marker != 0 {
                *value = PLAYER_MARKER_LUMA;
            }
        }
    }

    fn clear(&mut self) {
        self.key = None;
        self.marker_layer.clear();
        self.missing_samples = MARKER_HOLD_SAMPLES;
    }
}

fn mask_set(crop_profile: ObservationCropProfile) -> Option<MinimapMaskSet> {
    match crop_profile {
        ObservationCropProfile::Gliden64 => Some(MinimapMaskSet {
            roi: GLIDEN64_ROI,
            masks: GLIDEN64_MASKS,
        }),
        ObservationCropProfile::Angrylion => Some(MinimapMaskSet {
            roi: ANGRYLION_ROI,
            masks: ANGRYLION_MASKS,
        }),
    }
}

fn render_layer_into(
    request: MinimapLayerRequest,
    roi: MinimapRoi,
    mask: &[u8],
    output: &mut Vec<u8>,
    marker_layer: Option<&mut Vec<u8>>,
    mut sample: impl FnMut(usize, usize) -> Option<[u8; 3]>,
) -> Result<usize, CoreError> {
    let output_len = request
        .target_width
        .checked_mul(request.target_height)
        .ok_or(CoreError::NoFrameAvailable)?;
    if mask.len() != roi.width * roi.height {
        return Err(CoreError::NoFrameAvailable);
    }

    output.resize(output_len, 0);
    let mut marker_count = 0_usize;
    let mut marker_layer = marker_layer;
    if let Some(marker_layer) = marker_layer.as_deref_mut() {
        marker_layer.clear();
        marker_layer.resize(output_len, 0);
    }

    for output_y in 0..request.target_height {
        let mask_y = scale_axis(output_y, request.target_height, roi.height);
        for output_x in 0..request.target_width {
            let mask_x = scale_axis(output_x, request.target_width, roi.width);
            let mask_index = mask_y * roi.width + mask_x;
            let output_index = output_y * request.target_width + output_x;
            let [red, green, blue] =
                sample(roi.x + mask_x, roi.y + mask_y).ok_or(CoreError::NoFrameAvailable)?;
            if is_player_marker_color(red, green, blue)
                && let Some(marker_layer) = marker_layer.as_deref_mut()
            {
                marker_layer[output_index] = 1;
                marker_count += 1;
            }
            if mask[mask_index] == 0 {
                output[output_index] = 0;
                continue;
            }
            output[output_index] = rgb_to_luma(red, green, blue);
        }
    }
    Ok(marker_count)
}

fn write_zero_layer(request: MinimapLayerRequest, output: &mut Vec<u8>) -> Result<(), CoreError> {
    let output_len = request
        .target_width
        .checked_mul(request.target_height)
        .ok_or(CoreError::NoFrameAvailable)?;
    output.clear();
    output.resize(output_len, 0);
    Ok(())
}

fn scale_axis(output_index: usize, output_size: usize, input_size: usize) -> usize {
    if output_size <= 1 || input_size <= 1 {
        return 0;
    }
    let scale = (input_size - 1) as f64 / (output_size - 1) as f64;
    ((output_index as f64) * scale).round() as usize
}

fn rgb_to_luma(red: u8, green: u8, blue: u8) -> u8 {
    let weighted = (77 * u16::from(red)) + (150 * u16::from(green)) + (29 * u16::from(blue)) + 128;
    (weighted >> 8) as u8
}

fn is_player_marker_color(red: u8, green: u8, blue: u8) -> bool {
    red <= 80
        && green >= 140
        && blue >= 140
        && green.saturating_sub(red) >= 60
        && blue.saturating_sub(red) >= 60
        && green.abs_diff(blue) <= 96
}

#[cfg(test)]
mod tests {
    use super::{
        ANGRYLION_ROI, COURSE_COUNT, GLIDEN64_MASKS, GLIDEN64_ROI, MinimapLayerRenderer,
        MinimapLayerRequest, MinimapRoi, PLAYER_MARKER_LUMA,
    };
    use crate::core::observation::ObservationCropProfile;

    #[test]
    fn embedded_gliden64_masks_match_roi_area() {
        assert_eq!(GLIDEN64_MASKS.len(), COURSE_COUNT);
        assert!(
            GLIDEN64_MASKS
                .iter()
                .all(|mask| mask.len() == GLIDEN64_ROI.width * GLIDEN64_ROI.height)
        );
    }

    #[test]
    fn embedded_angrylion_roi_is_double_width_renderer_variant() {
        assert_eq!(ANGRYLION_ROI.width, GLIDEN64_ROI.width * 2);
    }

    #[test]
    fn marker_hold_reuses_last_marker_when_marker_blinks_off() {
        let request = test_request();
        let roi = MinimapRoi {
            x: 0,
            y: 0,
            width: 2,
            height: 1,
        };
        let mask = [1_u8, 1_u8];
        let mut renderer = MinimapLayerRenderer::default();
        let mut output = Vec::new();

        renderer
            .render_with_marker_hold(request, roi, &mask, &mut output, |x, _| {
                Some(if x == 0 {
                    [0, 255, 255]
                } else {
                    [255, 255, 255]
                })
            })
            .expect("visible marker frame should render");
        assert_eq!(output[0], PLAYER_MARKER_LUMA);

        renderer
            .render_with_marker_hold(request, roi, &mask, &mut output, |x, _| {
                Some(if x == 0 { [0, 0, 0] } else { [255, 255, 255] })
            })
            .expect("blink-off marker frame should render");
        assert_eq!(output[0], PLAYER_MARKER_LUMA);
    }

    #[test]
    fn visible_marker_replaces_held_marker_without_trail() {
        let request = test_request();
        let roi = MinimapRoi {
            x: 0,
            y: 0,
            width: 2,
            height: 1,
        };
        let mask = [1_u8, 1_u8];
        let mut renderer = MinimapLayerRenderer::default();
        let mut output = Vec::new();

        renderer
            .render_with_marker_hold(request, roi, &mask, &mut output, |x, _| {
                Some(if x == 0 { [0, 255, 255] } else { [0, 0, 0] })
            })
            .expect("first marker frame should render");
        renderer
            .render_with_marker_hold(request, roi, &mask, &mut output, |x, _| {
                Some(if x == 1 { [0, 255, 255] } else { [0, 0, 0] })
            })
            .expect("second marker frame should render");

        assert_eq!(output, [0, PLAYER_MARKER_LUMA]);
    }

    fn test_request() -> MinimapLayerRequest {
        MinimapLayerRequest {
            crop_profile: ObservationCropProfile::Gliden64,
            course_index: 0,
            target_width: 2,
            target_height: 1,
        }
    }
}
