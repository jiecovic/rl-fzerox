// rust/core/game/telemetry/layout.rs
//! Reverse-engineered F-Zero X telemetry layout details.

/// Scalar decoder settings that conceptually belong together.
#[derive(Clone, Copy, Debug)]
pub(super) struct TelemetryConfig {
    pub kseg0_base: usize,
    pub system_ram_size_min: usize,
    pub player_racer_index: usize,
    pub speed_to_kph: f32,
}

pub(super) const TELEMETRY_CONFIG: TelemetryConfig = TelemetryConfig {
    kseg0_base: 0x8000_0000,
    system_ram_size_min: 0x0030_0000,
    player_racer_index: 0,
    speed_to_kph: 21.6,
};

#[derive(Clone, Copy)]
pub(super) struct GlobalOffsets {
    pub total_lap_count: usize,
    pub game_mode: usize,
    pub total_racers: usize,
    pub course_index: usize,
    pub racers: usize,
}

#[derive(Clone, Copy)]
pub(super) struct RacerOffsets {
    pub size: usize,
    pub state_flags: usize,
    pub speed: usize,
    pub boost_timer: usize,
    pub energy: usize,
    pub max_energy: usize,
    pub race_distance: usize,
    pub lap_distance: usize,
    pub race_time: usize,
    pub lap: usize,
    pub laps_completed: usize,
    pub position: usize,
}

// Global RDRAM addresses derived from the F-Zero X USA decomp / symbol dumps.
// We keep them grouped here so the reverse-engineered memory layout is easy to
// audit and update as field semantics are validated.
pub(super) const GLOBALS: GlobalOffsets = GlobalOffsets {
    total_lap_count: rdram_offset(0x800C_D00C),
    game_mode: rdram_offset(0x800DCE44),
    total_racers: rdram_offset(0x800E5EC0),
    course_index: rdram_offset(0x800F8514),
    racers: rdram_offset(0x802C4920),
};

// Byte offsets within `struct Racer`, derived from the decomp's
// `include/unk_structs.h`.
pub(super) const RACER: RacerOffsets = RacerOffsets {
    size: 0x3A8,
    state_flags: 0x004,
    speed: 0x098,
    boost_timer: 0x21C,
    energy: 0x228,
    max_energy: 0x22C,
    race_distance: 0x23C,
    lap_distance: 0x244,
    race_time: 0x2A0,
    lap: 0x2A8,
    laps_completed: 0x2AA,
    position: 0x2AC,
};

// F-Zero X game-mode ids derived from the decomp's `include/fzx_game.h`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GameMode {
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

pub(super) const fn rdram_offset(vram_address: usize) -> usize {
    vram_address - TELEMETRY_CONFIG.kseg0_base
}
