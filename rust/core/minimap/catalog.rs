// rust/core/minimap/catalog.rs
//! Static renderer/course metadata for minimap observation masks.

use crate::core::observation::ObservationCropProfile;

pub(super) const COURSE_COUNT: usize = 24;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MinimapRoi {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MinimapMaskSet {
    pub roi: MinimapRoi,
    pub masks: [&'static [u8]; COURSE_COUNT],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MinimapTransform {
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

pub(super) const GLIDEN64_ROI: MinimapRoi = MinimapRoi {
    x: 235,
    y: 133,
    width: 58,
    height: 61,
};

pub(super) const ANGRYLION_ROI: MinimapRoi = MinimapRoi {
    x: 470,
    y: 134,
    width: 116,
    height: 60,
};

pub(super) const GLIDEN64_MASKS: [&[u8]; COURSE_COUNT] = [
    include_bytes!("masks/gliden64/00_jack_mute_city.bin"),
    include_bytes!("masks/gliden64/01_jack_silence.bin"),
    include_bytes!("masks/gliden64/02_jack_sand_ocean.bin"),
    include_bytes!("masks/gliden64/03_jack_devils_forest.bin"),
    include_bytes!("masks/gliden64/04_jack_big_blue.bin"),
    include_bytes!("masks/gliden64/05_jack_port_town.bin"),
    include_bytes!("masks/gliden64/06_queen_sector_alpha.bin"),
    include_bytes!("masks/gliden64/07_queen_red_canyon.bin"),
    include_bytes!("masks/gliden64/08_queen_devils_forest_2.bin"),
    include_bytes!("masks/gliden64/09_queen_mute_city_2.bin"),
    include_bytes!("masks/gliden64/10_queen_big_blue_2.bin"),
    include_bytes!("masks/gliden64/11_queen_white_land.bin"),
    include_bytes!("masks/gliden64/12_king_fire_field.bin"),
    include_bytes!("masks/gliden64/13_king_silence_2.bin"),
    include_bytes!("masks/gliden64/14_king_sector_beta.bin"),
    include_bytes!("masks/gliden64/15_king_red_canyon_2.bin"),
    include_bytes!("masks/gliden64/16_king_white_land_2.bin"),
    include_bytes!("masks/gliden64/17_king_mute_city_3.bin"),
    include_bytes!("masks/gliden64/18_joker_rainbow_road.bin"),
    include_bytes!("masks/gliden64/19_joker_devils_forest_3.bin"),
    include_bytes!("masks/gliden64/20_joker_space_plant.bin"),
    include_bytes!("masks/gliden64/21_joker_sand_ocean_2.bin"),
    include_bytes!("masks/gliden64/22_joker_port_town_2.bin"),
    include_bytes!("masks/gliden64/23_joker_big_hand.bin"),
];

pub(super) const ANGRYLION_MASKS: [&[u8]; COURSE_COUNT] = [
    include_bytes!("masks/angrylion/00_jack_mute_city.bin"),
    include_bytes!("masks/angrylion/01_jack_silence.bin"),
    include_bytes!("masks/angrylion/02_jack_sand_ocean.bin"),
    include_bytes!("masks/angrylion/03_jack_devils_forest.bin"),
    include_bytes!("masks/angrylion/04_jack_big_blue.bin"),
    include_bytes!("masks/angrylion/05_jack_port_town.bin"),
    include_bytes!("masks/angrylion/06_queen_sector_alpha.bin"),
    include_bytes!("masks/angrylion/07_queen_red_canyon.bin"),
    include_bytes!("masks/angrylion/08_queen_devils_forest_2.bin"),
    include_bytes!("masks/angrylion/09_queen_mute_city_2.bin"),
    include_bytes!("masks/angrylion/10_queen_big_blue_2.bin"),
    include_bytes!("masks/angrylion/11_queen_white_land.bin"),
    include_bytes!("masks/angrylion/12_king_fire_field.bin"),
    include_bytes!("masks/angrylion/13_king_silence_2.bin"),
    include_bytes!("masks/angrylion/14_king_sector_beta.bin"),
    include_bytes!("masks/angrylion/15_king_red_canyon_2.bin"),
    include_bytes!("masks/angrylion/16_king_white_land_2.bin"),
    include_bytes!("masks/angrylion/17_king_mute_city_3.bin"),
    include_bytes!("masks/angrylion/18_joker_rainbow_road.bin"),
    include_bytes!("masks/angrylion/19_joker_devils_forest_3.bin"),
    include_bytes!("masks/angrylion/20_joker_space_plant.bin"),
    include_bytes!("masks/angrylion/21_joker_sand_ocean_2.bin"),
    include_bytes!("masks/angrylion/22_joker_port_town_2.bin"),
    include_bytes!("masks/angrylion/23_joker_big_hand.bin"),
];

pub(super) fn mask_set(crop_profile: ObservationCropProfile) -> Option<MinimapMaskSet> {
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

pub(super) fn course_minimap_transform(course_index: usize) -> MinimapTransform {
    COURSE_MINIMAP_TRANSFORMS
        .get(course_index)
        .copied()
        .unwrap_or(MinimapTransform::Identity)
}
