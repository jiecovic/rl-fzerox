// rust/core/minimap.rs
//! Optional minimap observation layer rendered from raw emulator frames.

use crate::core::error::CoreError;
use crate::core::observation::ObservationCropProfile;
use crate::core::video::{RawVideoFrame, VideoFrame, VideoResizeFilter, resize_luma, sample_rgb};

const COURSE_COUNT: usize = 24;
const MARKER_HOLD_SAMPLES: u8 = 16;
const PARTIAL_MARKER_REPLACE_SAMPLES: u8 = 3;
const TRACK_LUMA: u8 = 128;
const PLAYER_MARKER_LUMA: u8 = 255;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MinimapTransform {
    Identity,
    Rotate90Clockwise,
}

const COURSE_MINIMAP_TRANSFORMS: [MinimapTransform; COURSE_COUNT] = [
    MinimapTransform::Identity,          // Mute City
    MinimapTransform::Rotate90Clockwise, // Silence
    MinimapTransform::Identity,          // Sand Ocean
    MinimapTransform::Rotate90Clockwise, // Devil's Forest
    MinimapTransform::Rotate90Clockwise, // Big Blue
    MinimapTransform::Identity,          // Port Town
    MinimapTransform::Identity,          // Sector Alpha
    MinimapTransform::Identity,          // Red Canyon
    MinimapTransform::Identity,          // Devil's Forest 2
    MinimapTransform::Identity,          // Mute City 2
    MinimapTransform::Identity,          // Big Blue 2
    MinimapTransform::Identity,          // White Land
    MinimapTransform::Rotate90Clockwise, // Fire Field
    MinimapTransform::Identity,          // Silence 2
    MinimapTransform::Identity,          // Sector Beta
    MinimapTransform::Identity,          // Red Canyon 2
    MinimapTransform::Identity,          // White Land 2
    MinimapTransform::Identity,          // Mute City 3
    MinimapTransform::Rotate90Clockwise, // Rainbow Road
    MinimapTransform::Identity,          // Devil's Forest 3
    MinimapTransform::Identity,          // Space Plant
    MinimapTransform::Identity,          // Sand Ocean 2
    MinimapTransform::Rotate90Clockwise, // Port Town 2
    MinimapTransform::Identity,          // Big Hand
];

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
    pub resize_filter: VideoResizeFilter,
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
    resize_filter: VideoResizeFilter,
}

#[derive(Debug, Default)]
struct MinimapMarkerHold {
    key: Option<MinimapLayerKey>,
    marker_layer: Vec<u8>,
    marker_pixels: usize,
    missing_samples: u8,
    partial_samples: u8,
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
            course_minimap_transform(request.course_index),
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
            resize_filter: request.resize_filter,
        };
        if self.key != Some(key) {
            self.key = Some(key);
            self.marker_layer.clear();
            self.marker_pixels = 0;
            self.missing_samples = MARKER_HOLD_SAMPLES;
            self.partial_samples = 0;
        }
    }

    fn update(&mut self, _marker_count: usize, current_marker: &[u8]) {
        let current_pixels = marker_pixel_count(current_marker);
        if current_pixels > 0 {
            if self.should_replace_marker(current_marker, current_pixels) {
                self.replace_marker(current_marker, current_pixels);
            } else {
                self.partial_samples = self.partial_samples.saturating_add(1);
            }
            self.missing_samples = 0;
            return;
        }
        if self.marker_layer.is_empty() {
            return;
        }
        self.missing_samples = self.missing_samples.saturating_add(1);
        if self.missing_samples > MARKER_HOLD_SAMPLES {
            self.marker_layer.clear();
            self.marker_pixels = 0;
            self.partial_samples = 0;
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
        self.marker_pixels = 0;
        self.missing_samples = MARKER_HOLD_SAMPLES;
        self.partial_samples = 0;
    }

    fn should_replace_marker(&self, current_marker: &[u8], current_pixels: usize) -> bool {
        if self.marker_layer.is_empty() {
            return true;
        }
        if current_pixels >= self.marker_pixels {
            return true;
        }
        if marker_is_subset(current_marker, &self.marker_layer) {
            return false;
        }
        self.partial_samples >= PARTIAL_MARKER_REPLACE_SAMPLES
    }

    fn replace_marker(&mut self, current_marker: &[u8], current_pixels: usize) {
        self.marker_layer.clear();
        self.marker_layer.extend_from_slice(current_marker);
        self.marker_pixels = current_pixels;
        self.partial_samples = 0;
    }
}

fn marker_pixel_count(marker_layer: &[u8]) -> usize {
    marker_layer.iter().filter(|value| **value != 0).count()
}

fn marker_is_subset(candidate: &[u8], reference: &[u8]) -> bool {
    candidate.len() == reference.len()
        && candidate
            .iter()
            .zip(reference.iter())
            .all(|(candidate_value, reference_value)| {
                *candidate_value == 0 || *reference_value != 0
            })
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

fn course_minimap_transform(course_index: usize) -> MinimapTransform {
    COURSE_MINIMAP_TRANSFORMS
        .get(course_index)
        .copied()
        .unwrap_or(MinimapTransform::Identity)
}

fn render_layer_into(
    request: MinimapLayerRequest,
    roi: MinimapRoi,
    mask: &[u8],
    transform: MinimapTransform,
    output: &mut Vec<u8>,
    marker_layer: Option<&mut Vec<u8>>,
    mut sample: impl FnMut(usize, usize) -> Option<[u8; 3]>,
) -> Result<usize, CoreError> {
    let output_len = request
        .target_width
        .checked_mul(request.target_height)
        .ok_or(CoreError::NoFrameAvailable)?;
    let roi_len = roi
        .width
        .checked_mul(roi.height)
        .ok_or(CoreError::NoFrameAvailable)?;
    if mask.len() != roi.width * roi.height {
        return Err(CoreError::NoFrameAvailable);
    }

    let mut roi_output = vec![0_u8; roi_len];
    let mut marker_count = 0_usize;
    let mut marker_layer = marker_layer;
    if let Some(marker_layer) = marker_layer.as_deref_mut() {
        marker_layer.clear();
        marker_layer.resize(roi_len, 0);
    }

    for roi_y in 0..roi.height {
        for roi_x in 0..roi.width {
            let roi_index = roi_y * roi.width + roi_x;
            let [red, green, blue] =
                sample(roi.x + roi_x, roi.y + roi_y).ok_or(CoreError::NoFrameAvailable)?;
            if is_player_marker_color(red, green, blue)
                && let Some(marker_layer) = marker_layer.as_deref_mut()
            {
                marker_layer[roi_index] = 1;
                marker_count += 1;
            }
            if mask[roi_index] == 0 {
                roi_output[roi_index] = 0;
                continue;
            }
            roi_output[roi_index] = TRACK_LUMA;
        }
    }

    let (roi_output, layer_width, layer_height) =
        transform_luma_layer(&roi_output, roi.width, roi.height, transform);
    if let Some(marker_layer) = marker_layer.as_deref_mut() {
        let (transformed_marker, _, _) =
            transform_luma_layer(marker_layer, roi.width, roi.height, transform);
        marker_layer.clear();
        marker_layer.extend_from_slice(&transformed_marker);
    }

    let resized_output = resize_luma(
        &roi_output,
        layer_width,
        layer_height,
        request.target_width,
        request.target_height,
        request.resize_filter,
    )?;
    output.clear();
    output.extend_from_slice(&resized_output);
    if let Some(marker_layer) = marker_layer.as_deref_mut() {
        let resized_marker = resize_luma(
            marker_layer,
            layer_width,
            layer_height,
            request.target_width,
            request.target_height,
            VideoResizeFilter::Nearest,
        )?;
        marker_layer.clear();
        marker_layer.extend_from_slice(&resized_marker);
    }
    debug_assert_eq!(output.len(), output_len);
    Ok(marker_count)
}

fn transform_luma_layer(
    layer: &[u8],
    width: usize,
    height: usize,
    transform: MinimapTransform,
) -> (Vec<u8>, usize, usize) {
    match transform {
        MinimapTransform::Identity => (layer.to_vec(), width, height),
        MinimapTransform::Rotate90Clockwise => rotate_luma_90_clockwise(layer, width, height),
    }
}

fn rotate_luma_90_clockwise(layer: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    let output_width = height;
    let output_height = width;
    let mut output = vec![0_u8; layer.len()];
    for y in 0..height {
        for x in 0..width {
            let source_index = y * width + x;
            let target_x = height - 1 - y;
            let target_y = x;
            output[target_y * output_width + target_x] = layer[source_index];
        }
    }
    (output, output_width, output_height)
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
        ANGRYLION_ROI, COURSE_COUNT, GLIDEN64_MASKS, GLIDEN64_ROI, MARKER_HOLD_SAMPLES,
        MinimapLayerRenderer, MinimapLayerRequest, MinimapRoi, PARTIAL_MARKER_REPLACE_SAMPLES,
        PLAYER_MARKER_LUMA, TRACK_LUMA,
    };
    use crate::core::observation::ObservationCropProfile;
    use crate::core::video::VideoResizeFilter;

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
    fn silence_minimap_rotates_before_resize() {
        let request = test_request_for_course_and_size(1, 3, 2);
        let roi = MinimapRoi {
            x: 0,
            y: 0,
            width: 2,
            height: 3,
        };
        let mask = [
            1_u8, 0_u8, //
            0_u8, 1_u8, //
            1_u8, 0_u8,
        ];
        let mut renderer = MinimapLayerRenderer::default();
        let mut output = Vec::new();

        renderer
            .render_with_marker_hold(request, roi, &mask, &mut output, |_, _| Some([0, 0, 0]))
            .expect("rotated minimap layer should render");

        assert_eq!(output, [TRACK_LUMA, 0, TRACK_LUMA, 0, TRACK_LUMA, 0,]);
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
        assert_eq!(output[1], TRACK_LUMA);

        renderer
            .render_with_marker_hold(request, roi, &mask, &mut output, |x, _| {
                Some(if x == 0 { [0, 0, 0] } else { [255, 255, 255] })
            })
            .expect("blink-off marker frame should render");
        assert_eq!(output[0], PLAYER_MARKER_LUMA);
        assert_eq!(output[1], TRACK_LUMA);
    }

    #[test]
    fn marker_hold_bridges_multi_frame_blink_off_at_action_repeat_one() {
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
            .expect("visible marker frame should render");
        assert_eq!(output[0], PLAYER_MARKER_LUMA);

        for _ in 0..MARKER_HOLD_SAMPLES {
            renderer
                .render_with_marker_hold(request, roi, &mask, &mut output, |_, _| Some([0, 0, 0]))
                .expect("blink-off marker frame should render");
            assert_eq!(output[0], PLAYER_MARKER_LUMA);
        }

        renderer
            .render_with_marker_hold(request, roi, &mask, &mut output, |_, _| Some([0, 0, 0]))
            .expect("expired blink-off marker frame should render");
        assert_eq!(output[0], TRACK_LUMA);
    }

    #[test]
    fn partial_marker_detection_does_not_replace_full_held_marker() {
        let request = test_request_for_width(3);
        let roi = MinimapRoi {
            x: 0,
            y: 0,
            width: 3,
            height: 1,
        };
        let mask = [1_u8, 1_u8, 1_u8];
        let mut renderer = MinimapLayerRenderer::default();
        let mut output = Vec::new();

        renderer
            .render_with_marker_hold(request, roi, &mask, &mut output, |x, _| {
                Some(if x <= 1 { [0, 255, 255] } else { [0, 0, 0] })
            })
            .expect("full marker frame should render");
        assert_eq!(output, [PLAYER_MARKER_LUMA, PLAYER_MARKER_LUMA, TRACK_LUMA]);

        renderer
            .render_with_marker_hold(request, roi, &mask, &mut output, |x, _| {
                Some(if x == 0 { [0, 255, 255] } else { [0, 0, 0] })
            })
            .expect("partial marker frame should render");
        assert_eq!(output, [PLAYER_MARKER_LUMA, PLAYER_MARKER_LUMA, TRACK_LUMA]);
    }

    #[test]
    fn persistent_smaller_marker_eventually_replaces_held_marker() {
        let request = test_request_for_width(3);
        let roi = MinimapRoi {
            x: 0,
            y: 0,
            width: 3,
            height: 1,
        };
        let mask = [1_u8, 1_u8, 1_u8];
        let mut renderer = MinimapLayerRenderer::default();
        let mut output = Vec::new();

        renderer
            .render_with_marker_hold(request, roi, &mask, &mut output, |x, _| {
                Some(if x <= 1 { [0, 255, 255] } else { [0, 0, 0] })
            })
            .expect("full marker frame should render");

        for _ in 0..=PARTIAL_MARKER_REPLACE_SAMPLES {
            renderer
                .render_with_marker_hold(request, roi, &mask, &mut output, |x, _| {
                    Some(if x == 2 { [0, 255, 255] } else { [0, 0, 0] })
                })
                .expect("persistent smaller marker frame should render");
        }

        assert_eq!(output, [TRACK_LUMA, TRACK_LUMA, PLAYER_MARKER_LUMA]);
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

        assert_eq!(output, [TRACK_LUMA, PLAYER_MARKER_LUMA]);
    }

    fn test_request() -> MinimapLayerRequest {
        test_request_for_width(2)
    }

    fn test_request_for_width(width: usize) -> MinimapLayerRequest {
        test_request_for_course_and_size(0, width, 1)
    }

    fn test_request_for_course_and_size(
        course_index: usize,
        target_width: usize,
        target_height: usize,
    ) -> MinimapLayerRequest {
        MinimapLayerRequest {
            crop_profile: ObservationCropProfile::Gliden64,
            course_index,
            target_width,
            target_height,
            resize_filter: VideoResizeFilter::Nearest,
        }
    }
}
