// rust/core/minimap/tests.rs
use super::catalog::{
    ANGRYLION_ROI, COURSE_COUNT, COURSE_MINIMAP_CATALOG, GLIDEN64_ROI, MinimapRoi,
};
use super::marker::{
    MARKER_HOLD_SAMPLES, PARTIAL_MARKER_REPLACE_SAMPLES, PLAYER_MARKER_LUMA, TRACK_LUMA,
};
use super::{MinimapLayerRenderer, MinimapLayerRequest};
use crate::core::observation::ObservationCropProfile;
use crate::core::video::VideoResizeFilter;

#[test]
fn embedded_gliden64_masks_match_roi_area() {
    assert_eq!(COURSE_MINIMAP_CATALOG.len(), COURSE_COUNT);
    assert!(
        COURSE_MINIMAP_CATALOG
            .iter()
            .all(|entry| entry.gliden64_mask.len() == GLIDEN64_ROI.width * GLIDEN64_ROI.height)
    );
}

#[test]
fn embedded_course_catalog_matches_game_course_order() {
    let course_ids: Vec<_> = COURSE_MINIMAP_CATALOG
        .iter()
        .map(|entry| entry.id)
        .collect();

    assert_eq!(
        course_ids,
        [
            "jack/mute_city",
            "jack/silence",
            "jack/sand_ocean",
            "jack/devils_forest",
            "jack/big_blue",
            "jack/port_town",
            "queen/sector_alpha",
            "queen/red_canyon",
            "queen/devils_forest_2",
            "queen/mute_city_2",
            "queen/big_blue_2",
            "queen/white_land",
            "king/fire_field",
            "king/silence_2",
            "king/sector_beta",
            "king/red_canyon_2",
            "king/white_land_2",
            "king/mute_city_3",
            "joker/rainbow_road",
            "joker/devils_forest_3",
            "joker/space_plant",
            "joker/sand_ocean_2",
            "joker/port_town_2",
            "joker/big_hand",
        ]
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
fn marker_detection_is_clipped_to_course_mask() {
    let request = test_request_for_width(3);
    let roi = MinimapRoi {
        x: 0,
        y: 0,
        width: 3,
        height: 1,
    };
    let mask = [1_u8, 0_u8, 1_u8];
    let mut renderer = MinimapLayerRenderer::default();
    let mut output = Vec::new();

    renderer
        .render_with_marker_hold(request, roi, &mask, &mut output, |x, _| {
            Some(if x == 1 { [0, 255, 255] } else { [0, 0, 0] })
        })
        .expect("marker-colored noise outside the mask should render");
    assert_eq!(output, [TRACK_LUMA, 0, TRACK_LUMA]);

    renderer
        .render_with_marker_hold(request, roi, &mask, &mut output, |_, _| Some([0, 0, 0]))
        .expect("masked marker-colored noise should not enter marker hold");
    assert_eq!(output, [TRACK_LUMA, 0, TRACK_LUMA]);
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
