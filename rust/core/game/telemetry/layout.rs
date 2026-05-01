// rust/core/game/telemetry/layout.rs
//! Reverse-engineered F-Zero X telemetry layout details.

/// Scalar decoder settings that conceptually belong together.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TelemetryConfig {
    pub kseg0_base: usize,
    pub system_ram_size_min: usize,
    pub player_racer_index: usize,
    pub speed_to_kph: f32,
}

pub(crate) const TELEMETRY_CONFIG: TelemetryConfig = TelemetryConfig {
    kseg0_base: 0x8000_0000,
    system_ram_size_min: 0x0030_0000,
    player_racer_index: 0,
    speed_to_kph: 21.6,
};

#[derive(Clone, Copy)]
pub(crate) struct GlobalOffsets {
    pub num_players: usize,
    pub total_lap_count: usize,
    pub difficulty: usize,
    pub race_intro_timer: usize,
    pub selected_mode: usize,
    pub game_mode_change_state: usize,
    pub current_ghost_type: usize,
    pub game_mode: usize,
    pub queued_game_mode: usize,
    pub total_racers: usize,
    pub current_course_info: usize,
    pub course_index: usize,
    pub cameras: usize,
    pub damage_rumble_counters: usize,
    pub character_last_engine: usize,
    pub player_characters: usize,
    pub player_machine_skins: usize,
    pub player_engine: usize,
    pub reverse_timers: usize,
    pub racers: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct MachineTableOffsets {
    pub machines: usize,
    pub machine_count: usize,
    pub machine_size: usize,
    pub body_stat: usize,
    pub boost_stat: usize,
    pub grip_stat: usize,
    pub weight: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct CameraOffsets {
    pub race_setting: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct RacerOffsets {
    pub size: usize,
    pub state_flags: usize,
    pub segment_position_info: usize,
    pub local_velocity: usize,
    pub velocity: usize,
    pub acceleration: usize,
    pub speed: usize,
    pub height_above_ground: usize,
    pub acceleration_force: usize,
    pub drift_attack_force: usize,
    pub colliding_strength: usize,
    pub engine_curve: usize,
    pub boost_timer: usize,
    pub recoil_tilt: usize,
    pub energy: usize,
    pub max_energy: usize,
    pub race_distance: usize,
    pub lap_distance: usize,
    pub segment_basis: usize,
    pub current_radius_left: usize,
    pub current_radius_right: usize,
    pub z_button_timer: usize,
    pub r_button_timer: usize,
    pub race_time: usize,
    pub character: usize,
    pub machine_skin_index: usize,
    pub lap: usize,
    pub laps_completed: usize,
    pub position: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct RacerSegmentPositionInfoOffsets {
    pub course_segment: usize,
    pub segment_t_value: usize,
    pub segment_length_proportion: usize,
    pub segment_displacement: usize,
    pub distance_from_segment: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct CourseSegmentOffsets {
    pub segment_index: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct CourseInfoOffsets {
    pub segment_count: usize,
    pub length: usize,
}

// Global RDRAM addresses derived from the F-Zero X USA decomp / symbol dumps.
// We keep them grouped here so the reverse-engineered memory layout is easy to
// audit and update as field semantics are validated.
pub(crate) const GLOBALS: GlobalOffsets = GlobalOffsets {
    num_players: rdram_offset(0x800C_D000),
    total_lap_count: rdram_offset(0x800C_D00C),
    difficulty: rdram_offset(0x800C_D008),
    race_intro_timer: rdram_offset(0x800F_5E98),
    selected_mode: rdram_offset(0x800C_D380),
    game_mode_change_state: rdram_offset(0x800C_D046),
    current_ghost_type: rdram_offset(0x800C_D3CC),
    game_mode: rdram_offset(0x800DCE44),
    queued_game_mode: rdram_offset(0x800DCE48),
    total_racers: rdram_offset(0x800E5EC0),
    current_course_info: rdram_offset(0x800F8510),
    course_index: rdram_offset(0x800F8514),
    cameras: rdram_offset(0x800E5220),
    damage_rumble_counters: rdram_offset(0x800E5F20),
    character_last_engine: rdram_offset(0x800E40F0),
    player_characters: rdram_offset(0x800E5EE0),
    player_machine_skins: rdram_offset(0x800E5EE8),
    player_engine: rdram_offset(0x800E5EF0),
    reverse_timers: rdram_offset(0x800F_80A8),
    racers: rdram_offset(0x802C4920),
};

// Byte offsets within the live `gMachines` table. Single-byte table fields are
// read with word-swapped addressing because Mupen exposes raw N64 RDRAM bytes.
pub(crate) const MACHINE_TABLE: MachineTableOffsets = MachineTableOffsets {
    machines: rdram_offset(0x800F80C8),
    machine_count: 30,
    machine_size: 0x16,
    body_stat: 0x11,
    boost_stat: 0x12,
    grip_stat: 0x13,
    weight: 0x14,
};

// Byte offsets within `struct Camera`, derived from the decomp's
// `include/fzx_camera.h`.
pub(crate) const CAMERA: CameraOffsets = CameraOffsets {
    race_setting: 0x008,
};

// Byte offsets within `struct Racer`, derived from the decomp's
// `include/unk_structs.h`.
pub(crate) const RACER: RacerOffsets = RacerOffsets {
    size: 0x3A8,
    state_flags: 0x004,
    segment_position_info: 0x00C,
    local_velocity: 0x05C,
    velocity: 0x074,
    acceleration: 0x08C,
    speed: 0x098,
    height_above_ground: 0x0A0,
    acceleration_force: 0x1D4,
    drift_attack_force: 0x1D8,
    colliding_strength: 0x1F4,
    engine_curve: 0x1A8,
    boost_timer: 0x218,
    recoil_tilt: 0x118,
    energy: 0x228,
    max_energy: 0x22C,
    race_distance: 0x23C,
    lap_distance: 0x244,
    segment_basis: 0x24C,
    current_radius_left: 0x270,
    current_radius_right: 0x274,
    z_button_timer: 0x278,
    r_button_timer: 0x27A,
    race_time: 0x2A0,
    character: 0x2C8,
    machine_skin_index: 0x2CC,
    lap: 0x2A8,
    laps_completed: 0x2AA,
    position: 0x2AC,
};

pub(crate) const RACER_SEGMENT_POSITION_INFO: RacerSegmentPositionInfoOffsets =
    RacerSegmentPositionInfoOffsets {
        course_segment: 0x000,
        segment_t_value: 0x004,
        segment_length_proportion: 0x008,
        segment_displacement: 0x028,
        distance_from_segment: 0x040,
    };

pub(crate) const COURSE_SEGMENT: CourseSegmentOffsets = CourseSegmentOffsets {
    segment_index: 0x030,
};

// Byte offsets within `struct CourseInfo`, derived from the decomp's
// `include/unk_structs.h`. `length` is the summed spline length for one lap.
pub(crate) const COURSE_INFO: CourseInfoOffsets = CourseInfoOffsets {
    segment_count: 0x008,
    length: 0x00C,
};

pub(crate) const fn player_z_button_timer_offset() -> usize {
    player_racer_field_offset(RACER.z_button_timer)
}

pub(crate) const fn player_r_button_timer_offset() -> usize {
    player_racer_field_offset(RACER.r_button_timer)
}

const fn player_racer_field_offset(field_offset: usize) -> usize {
    GLOBALS.racers + (TELEMETRY_CONFIG.player_racer_index * RACER.size) + field_offset
}

// F-Zero X game-mode ids derived from the decomp's `include/fzx_game.h`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GameMode {
    Title = 0x00,
    GpRace = 0x01,
    Practice = 0x02,
    Vs2p = 0x03,
    Vs3p = 0x04,
    Vs4p = 0x05,
    Records = 0x06,
    MainMenu = 0x07,
    MachineSelect = 0x08,
    MachineSettings = 0x09,
    CourseSelect = 0x0A,
    SkippableCredits = 0x0B,
    UnskippableCredits = 0x0C,
    CourseEdit = 0x0D,
    TimeAttack = 0x0E,
    GpRaceNextCourse = 0x0F,
    CreateMachine = 0x10,
    GpEndCutscene = 0x11,
    GpRaceNextMachineSettings = 0x12,
    RecordsCourseSelect = 0x13,
    OptionsMenu = 0x14,
    DeathRace = 0x15,
    EadDemo = 0x16,
}

impl GameMode {
    pub(super) const fn wire_name(self) -> &'static str {
        match self {
            Self::Title => "title",
            Self::GpRace => "gp_race",
            Self::Practice => "practice",
            Self::Vs2p => "vs_2p",
            Self::Vs3p => "vs_3p",
            Self::Vs4p => "vs_4p",
            Self::Records => "records",
            Self::MainMenu => "main_menu",
            Self::MachineSelect => "machine_select",
            Self::MachineSettings => "machine_settings",
            Self::CourseSelect => "course_select",
            Self::SkippableCredits => "skippable_credits",
            Self::UnskippableCredits => "unskippable_credits",
            Self::CourseEdit => "course_edit",
            Self::TimeAttack => "time_attack",
            Self::GpRaceNextCourse => "gp_race_next_course",
            Self::CreateMachine => "create_machine",
            Self::GpEndCutscene => "gp_end_cutscene",
            Self::GpRaceNextMachineSettings => "gp_race_next_machine_settings",
            Self::RecordsCourseSelect => "records_course_select",
            Self::OptionsMenu => "options_menu",
            Self::DeathRace => "death_race",
            Self::EadDemo => "ead_demo",
        }
    }

    pub(super) const fn is_race(self) -> bool {
        matches!(
            self,
            Self::GpRace
                | Self::Practice
                | Self::Vs2p
                | Self::Vs3p
                | Self::Vs4p
                | Self::TimeAttack
                | Self::DeathRace
        )
    }
}

impl TryFrom<u32> for GameMode {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            x if x == Self::Title as u32 => Ok(Self::Title),
            x if x == Self::GpRace as u32 => Ok(Self::GpRace),
            x if x == Self::Practice as u32 => Ok(Self::Practice),
            x if x == Self::Vs2p as u32 => Ok(Self::Vs2p),
            x if x == Self::Vs3p as u32 => Ok(Self::Vs3p),
            x if x == Self::Vs4p as u32 => Ok(Self::Vs4p),
            x if x == Self::Records as u32 => Ok(Self::Records),
            x if x == Self::MainMenu as u32 => Ok(Self::MainMenu),
            x if x == Self::MachineSelect as u32 => Ok(Self::MachineSelect),
            x if x == Self::MachineSettings as u32 => Ok(Self::MachineSettings),
            x if x == Self::CourseSelect as u32 => Ok(Self::CourseSelect),
            x if x == Self::SkippableCredits as u32 => Ok(Self::SkippableCredits),
            x if x == Self::UnskippableCredits as u32 => Ok(Self::UnskippableCredits),
            x if x == Self::CourseEdit as u32 => Ok(Self::CourseEdit),
            x if x == Self::TimeAttack as u32 => Ok(Self::TimeAttack),
            x if x == Self::GpRaceNextCourse as u32 => Ok(Self::GpRaceNextCourse),
            x if x == Self::CreateMachine as u32 => Ok(Self::CreateMachine),
            x if x == Self::GpEndCutscene as u32 => Ok(Self::GpEndCutscene),
            x if x == Self::GpRaceNextMachineSettings as u32 => Ok(Self::GpRaceNextMachineSettings),
            x if x == Self::RecordsCourseSelect as u32 => Ok(Self::RecordsCourseSelect),
            x if x == Self::OptionsMenu as u32 => Ok(Self::OptionsMenu),
            x if x == Self::DeathRace as u32 => Ok(Self::DeathRace),
            x if x == Self::EadDemo as u32 => Ok(Self::EadDemo),
            _ => Err(()),
        }
    }
}

// F-Zero X race difficulty ids derived from the decomp's `include/fzx_game.h`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RaceDifficulty {
    Novice = 0x00,
    Standard = 0x01,
    Expert = 0x02,
    Master = 0x03,
}

impl RaceDifficulty {
    pub(super) const fn wire_name(self) -> &'static str {
        match self {
            Self::Novice => "novice",
            Self::Standard => "standard",
            Self::Expert => "expert",
            Self::Master => "master",
        }
    }
}

impl TryFrom<i32> for RaceDifficulty {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            x if x == Self::Novice as i32 => Ok(Self::Novice),
            x if x == Self::Standard as i32 => Ok(Self::Standard),
            x if x == Self::Expert as i32 => Ok(Self::Expert),
            x if x == Self::Master as i32 => Ok(Self::Master),
            _ => Err(()),
        }
    }
}

// F-Zero X race camera settings derived from the decomp's `include/fzx_camera.h`.
#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CameraRaceSetting {
    Overhead = 0x00,
    CloseBehind = 0x01,
    Regular = 0x02,
    Wide = 0x03,
}

impl CameraRaceSetting {
    pub(super) const fn wire_name(self) -> &'static str {
        match self {
            Self::Overhead => "overhead",
            Self::CloseBehind => "close_behind",
            Self::Regular => "regular",
            Self::Wide => "wide",
        }
    }
}

impl TryFrom<i32> for CameraRaceSetting {
    type Error = ();

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            x if x == Self::Overhead as i32 => Ok(Self::Overhead),
            x if x == Self::CloseBehind as i32 => Ok(Self::CloseBehind),
            x if x == Self::Regular as i32 => Ok(Self::Regular),
            x if x == Self::Wide as i32 => Ok(Self::Wide),
            _ => Err(()),
        }
    }
}

pub(super) const fn rdram_offset(vram_address: usize) -> usize {
    vram_address - TELEMETRY_CONFIG.kseg0_base
}
