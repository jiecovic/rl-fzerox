// rust/core/minimap/marker.rs
//! Player-marker detection and blink hold for minimap observation layers.

use super::MinimapLayerRequest;

pub(super) const MARKER_HOLD_SAMPLES: u8 = 16;
pub(super) const PARTIAL_MARKER_REPLACE_SAMPLES: u8 = 3;
pub(super) const TRACK_LUMA: u8 = 128;
pub(super) const PLAYER_MARKER_LUMA: u8 = 255;

#[derive(Debug, Default)]
pub(super) struct MinimapMarkerHold {
    key: Option<MinimapLayerRequest>,
    marker_layer: Vec<u8>,
    marker_pixels: usize,
    missing_samples: u8,
    partial_samples: u8,
}

impl MinimapMarkerHold {
    pub(super) fn ensure_key(&mut self, request: MinimapLayerRequest) {
        if self.key != Some(request) {
            self.key = Some(request);
            self.marker_layer.clear();
            self.marker_pixels = 0;
            self.missing_samples = MARKER_HOLD_SAMPLES;
            self.partial_samples = 0;
        }
    }

    pub(super) fn update(&mut self, _marker_count: usize, current_marker: &[u8]) {
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

    pub(super) fn overlay(&self, output: &mut [u8]) {
        if self.marker_layer.len() != output.len() {
            return;
        }
        for (value, marker) in output.iter_mut().zip(self.marker_layer.iter()) {
            if *marker != 0 {
                *value = PLAYER_MARKER_LUMA;
            }
        }
    }

    pub(super) fn clear(&mut self) {
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

#[inline(always)]
pub(super) fn is_player_marker_color(red: u8, green: u8, blue: u8) -> bool {
    red <= 80
        && green >= 140
        && blue >= 140
        && green.saturating_sub(red) >= 60
        && blue.saturating_sub(red) >= 60
        && green.abs_diff(blue) <= 96
}
