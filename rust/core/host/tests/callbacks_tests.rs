// Covers callback utility helpers and callback-owned observation stacking.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{CallbackState, StackedObservationRequest, runtime_root_for_core};
use crate::core::observation::ObservationStackMode;
use crate::core::video::{VideoCrop, VideoFrame};

static NEXT_RUNTIME_DIR_ID: AtomicUsize = AtomicUsize::new(0);

#[test]
fn runtime_root_uses_repo_local_root_for_libretro_layout() {
    let runtime_root = runtime_root_for_core(Path::new(
        "/repo/local/libretro/mupen64plus_next_libretro.so",
    ))
    .expect("runtime root should resolve");

    assert_eq!(runtime_root, PathBuf::from("/repo/local"));
}

#[test]
fn runtime_root_uses_core_directory_for_generic_layout() {
    let runtime_root = runtime_root_for_core(Path::new("/opt/cores/mupen64plus_next_libretro.so"))
        .expect("runtime root should resolve");

    assert_eq!(runtime_root, PathBuf::from("/opt/cores"));
}

#[test]
fn stacked_observation_repeats_the_first_frame_across_the_full_stack() {
    let mut state = callback_state();
    state.set_frame(rgb_frame([10, 11, 12]));

    let stacked = stacked_observation_bytes(&mut state);

    assert_eq!(
        stacked,
        vec![10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 12]
    );
}

#[test]
fn stacked_observation_appends_each_new_frame_once() {
    let mut state = callback_state();
    state.set_frame(rgb_frame([10, 11, 12]));
    assert_eq!(
        stacked_observation_bytes(&mut state),
        vec![10, 11, 12, 10, 11, 12, 10, 11, 12, 10, 11, 12]
    );

    state.set_frame_for_test_without_reset(rgb_frame([20, 21, 22]));
    assert_eq!(
        stacked_observation_bytes(&mut state),
        vec![10, 11, 12, 10, 11, 12, 10, 11, 12, 20, 21, 22]
    );

    assert_eq!(
        stacked_observation_bytes(&mut state),
        vec![10, 11, 12, 10, 11, 12, 10, 11, 12, 20, 21, 22]
    );

    state.set_frame_for_test_without_reset(rgb_frame([30, 31, 32]));
    assert_eq!(
        stacked_observation_bytes(&mut state),
        vec![10, 11, 12, 10, 11, 12, 20, 21, 22, 30, 31, 32]
    );
}

#[test]
fn set_frame_clears_existing_stacked_observations_for_the_next_episode() {
    let mut state = callback_state();
    state.set_frame(rgb_frame([10, 11, 12]));
    let _ = stacked_observation_bytes(&mut state);

    state.set_frame_for_test_without_reset(rgb_frame([20, 21, 22]));
    let _ = stacked_observation_bytes(&mut state);

    state.set_frame(rgb_frame([40, 41, 42]));
    let stacked = stacked_observation_bytes(&mut state);

    assert_eq!(
        stacked,
        vec![40, 41, 42, 40, 41, 42, 40, 41, 42, 40, 41, 42]
    );
}

#[test]
fn stacked_observation_interleaves_frames_per_pixel_for_multi_pixel_images() {
    let mut state = callback_state();
    state.set_frame(rgb_row_frame([[1, 2, 3], [4, 5, 6]]));

    let stacked = state
        .stacked_observation_frame(stacked_observation_request(
            2,
            1,
            2,
            ObservationStackMode::Rgb,
        ))
        .expect("stacked observation should render")
        .to_vec();

    assert_eq!(stacked, vec![1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,]);

    state.set_frame_for_test_without_reset(rgb_row_frame([[10, 11, 12], [13, 14, 15]]));

    let stacked = state
        .stacked_observation_frame(stacked_observation_request(
            2,
            1,
            2,
            ObservationStackMode::Rgb,
        ))
        .expect("stacked observation should render")
        .to_vec();

    assert_eq!(stacked, vec![1, 2, 3, 10, 11, 12, 4, 5, 6, 13, 14, 15,]);
}

#[test]
fn rgb_gray_stack_encodes_history_as_luma_and_latest_as_rgb() {
    let mut state = callback_state();
    state.set_frame(rgb_row_frame([[10, 20, 30], [40, 50, 60]]));
    let _ = state
        .stacked_observation_frame(stacked_observation_request(
            2,
            1,
            4,
            ObservationStackMode::RgbGray,
        ))
        .expect("initial stacked observation should render");

    state.set_frame_for_test_without_reset(rgb_row_frame([[70, 80, 90], [100, 110, 120]]));
    let stacked = state
        .stacked_observation_frame(stacked_observation_request(
            2,
            1,
            4,
            ObservationStackMode::RgbGray,
        ))
        .expect("stacked observation should render")
        .to_vec();

    assert_eq!(
        stacked,
        vec![18, 18, 18, 70, 80, 90, 48, 48, 48, 100, 110, 120,]
    );
}

fn callback_state() -> CallbackState {
    let runtime_dir = std::env::temp_dir().join(format!(
        "rl_fzerox_callbacks_tests_{}_{}",
        std::process::id(),
        NEXT_RUNTIME_DIR_ID.fetch_add(1, Ordering::Relaxed),
    ));
    std::fs::create_dir_all(&runtime_dir).expect("runtime dir should be creatable");
    CallbackState::new(
        Path::new("/tmp/mupen64plus_next_libretro.so"),
        Some(runtime_dir.as_path()),
        "angrylion",
    )
    .expect("callback state should initialize")
}

fn rgb_frame(rgb: [u8; 3]) -> VideoFrame {
    VideoFrame {
        width: 1,
        height: 1,
        rgb: rgb.to_vec(),
    }
}

fn rgb_row_frame(pixels: [[u8; 3]; 2]) -> VideoFrame {
    VideoFrame {
        width: 2,
        height: 1,
        rgb: pixels.into_iter().flatten().collect(),
    }
}

fn stacked_observation_bytes(state: &mut CallbackState) -> Vec<u8> {
    state
        .stacked_observation_frame(stacked_observation_request(
            1,
            1,
            4,
            ObservationStackMode::Rgb,
        ))
        .expect("stacked observation should render")
        .to_vec()
}

fn stacked_observation_request(
    target_width: usize,
    target_height: usize,
    frame_stack: usize,
    stack_mode: ObservationStackMode,
) -> StackedObservationRequest {
    StackedObservationRequest {
        aspect_ratio: 0.0,
        target_width,
        target_height,
        rgb: true,
        crop: VideoCrop::default(),
        frame_stack,
        stack_mode,
    }
}
