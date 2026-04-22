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
pub(super) enum MinimapTransform {
    Identity,
    Rotate90Clockwise,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct MinimapCourseEntry {
    pub id: &'static str,
    pub transform: MinimapTransform,
    pub gliden64_mask: &'static [u8],
    pub angrylion_mask: &'static [u8],
}

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

pub(super) const COURSE_MINIMAP_CATALOG: [MinimapCourseEntry; COURSE_COUNT] = [
    entry(
        "jack/mute_city",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/00_jack_mute_city.bin"),
        include_bytes!("masks/angrylion/00_jack_mute_city.bin"),
    ),
    entry(
        "jack/silence",
        MinimapTransform::Rotate90Clockwise,
        include_bytes!("masks/gliden64/01_jack_silence.bin"),
        include_bytes!("masks/angrylion/01_jack_silence.bin"),
    ),
    entry(
        "jack/sand_ocean",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/02_jack_sand_ocean.bin"),
        include_bytes!("masks/angrylion/02_jack_sand_ocean.bin"),
    ),
    entry(
        "jack/devils_forest",
        MinimapTransform::Rotate90Clockwise,
        include_bytes!("masks/gliden64/03_jack_devils_forest.bin"),
        include_bytes!("masks/angrylion/03_jack_devils_forest.bin"),
    ),
    entry(
        "jack/big_blue",
        MinimapTransform::Rotate90Clockwise,
        include_bytes!("masks/gliden64/04_jack_big_blue.bin"),
        include_bytes!("masks/angrylion/04_jack_big_blue.bin"),
    ),
    entry(
        "jack/port_town",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/05_jack_port_town.bin"),
        include_bytes!("masks/angrylion/05_jack_port_town.bin"),
    ),
    entry(
        "queen/sector_alpha",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/06_queen_sector_alpha.bin"),
        include_bytes!("masks/angrylion/06_queen_sector_alpha.bin"),
    ),
    entry(
        "queen/red_canyon",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/07_queen_red_canyon.bin"),
        include_bytes!("masks/angrylion/07_queen_red_canyon.bin"),
    ),
    entry(
        "queen/devils_forest_2",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/08_queen_devils_forest_2.bin"),
        include_bytes!("masks/angrylion/08_queen_devils_forest_2.bin"),
    ),
    entry(
        "queen/mute_city_2",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/09_queen_mute_city_2.bin"),
        include_bytes!("masks/angrylion/09_queen_mute_city_2.bin"),
    ),
    entry(
        "queen/big_blue_2",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/10_queen_big_blue_2.bin"),
        include_bytes!("masks/angrylion/10_queen_big_blue_2.bin"),
    ),
    entry(
        "queen/white_land",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/11_queen_white_land.bin"),
        include_bytes!("masks/angrylion/11_queen_white_land.bin"),
    ),
    entry(
        "king/fire_field",
        MinimapTransform::Rotate90Clockwise,
        include_bytes!("masks/gliden64/12_king_fire_field.bin"),
        include_bytes!("masks/angrylion/12_king_fire_field.bin"),
    ),
    entry(
        "king/silence_2",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/13_king_silence_2.bin"),
        include_bytes!("masks/angrylion/13_king_silence_2.bin"),
    ),
    entry(
        "king/sector_beta",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/14_king_sector_beta.bin"),
        include_bytes!("masks/angrylion/14_king_sector_beta.bin"),
    ),
    entry(
        "king/red_canyon_2",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/15_king_red_canyon_2.bin"),
        include_bytes!("masks/angrylion/15_king_red_canyon_2.bin"),
    ),
    entry(
        "king/white_land_2",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/16_king_white_land_2.bin"),
        include_bytes!("masks/angrylion/16_king_white_land_2.bin"),
    ),
    entry(
        "king/mute_city_3",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/17_king_mute_city_3.bin"),
        include_bytes!("masks/angrylion/17_king_mute_city_3.bin"),
    ),
    entry(
        "joker/rainbow_road",
        MinimapTransform::Rotate90Clockwise,
        include_bytes!("masks/gliden64/18_joker_rainbow_road.bin"),
        include_bytes!("masks/angrylion/18_joker_rainbow_road.bin"),
    ),
    entry(
        "joker/devils_forest_3",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/19_joker_devils_forest_3.bin"),
        include_bytes!("masks/angrylion/19_joker_devils_forest_3.bin"),
    ),
    entry(
        "joker/space_plant",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/20_joker_space_plant.bin"),
        include_bytes!("masks/angrylion/20_joker_space_plant.bin"),
    ),
    entry(
        "joker/sand_ocean_2",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/21_joker_sand_ocean_2.bin"),
        include_bytes!("masks/angrylion/21_joker_sand_ocean_2.bin"),
    ),
    entry(
        "joker/port_town_2",
        MinimapTransform::Rotate90Clockwise,
        include_bytes!("masks/gliden64/22_joker_port_town_2.bin"),
        include_bytes!("masks/angrylion/22_joker_port_town_2.bin"),
    ),
    entry(
        "joker/big_hand",
        MinimapTransform::Identity,
        include_bytes!("masks/gliden64/23_joker_big_hand.bin"),
        include_bytes!("masks/angrylion/23_joker_big_hand.bin"),
    ),
];

const fn entry(
    id: &'static str,
    transform: MinimapTransform,
    gliden64_mask: &'static [u8],
    angrylion_mask: &'static [u8],
) -> MinimapCourseEntry {
    MinimapCourseEntry {
        id,
        transform,
        gliden64_mask,
        angrylion_mask,
    }
}

pub(super) fn minimap_roi(crop_profile: ObservationCropProfile) -> Option<MinimapRoi> {
    match crop_profile {
        ObservationCropProfile::Gliden64 => Some(GLIDEN64_ROI),
        ObservationCropProfile::Angrylion => Some(ANGRYLION_ROI),
    }
}

pub(super) fn course_minimap_mask(
    crop_profile: ObservationCropProfile,
    course_index: usize,
) -> Option<&'static [u8]> {
    let entry = COURSE_MINIMAP_CATALOG.get(course_index)?;
    Some(match crop_profile {
        ObservationCropProfile::Gliden64 => entry.gliden64_mask,
        ObservationCropProfile::Angrylion => entry.angrylion_mask,
    })
}

pub(super) fn course_minimap_transform(course_index: usize) -> MinimapTransform {
    COURSE_MINIMAP_CATALOG
        .get(course_index)
        .map(|entry| entry.transform)
        .unwrap_or(MinimapTransform::Identity)
}
