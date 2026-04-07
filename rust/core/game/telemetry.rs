// rust/core/game/telemetry.rs
use std::mem::size_of;

use libretro_sys::MEMORY_SYSTEM_RAM;

use crate::core::error::CoreError;

const KSEG0_BASE: usize = 0x8000_0000;
const SYSTEM_RAM_SIZE_MIN: usize = 0x0030_0000;
const PLAYER_RACER_INDEX: usize = 0;
const SPEED_TO_KPH: f32 = 21.6;
const GAME_MODE_MASK: u32 = 0x1F;

#[derive(Clone, Copy)]
struct GlobalOffsets {
    game_mode: usize,
    game_frame_count: usize,
    course_index: usize,
    racers: usize,
}

#[derive(Clone, Copy)]
struct RacerOffsets {
    size: usize,
    state_flags: usize,
    speed: usize,
    boost_timer: usize,
    energy: usize,
    max_energy: usize,
    race_distance: usize,
    laps_completed_distance: usize,
    lap_distance: usize,
    race_distance_position: usize,
    race_time: usize,
    lap: usize,
    laps_completed: usize,
    position: usize,
    character: usize,
    machine_index: usize,
}

// Global RDRAM addresses derived from the F-Zero X USA decomp / symbol dumps.
// We keep them grouped here so the reverse-engineered memory layout is easy to
// audit and update as field semantics are validated.
const GLOBALS: GlobalOffsets = GlobalOffsets {
    game_mode: rdram_offset(0x800DCE44),
    game_frame_count: rdram_offset(0x800CCFE0),
    course_index: rdram_offset(0x800F8514),
    racers: rdram_offset(0x802C4920),
};

// Byte offsets within `struct Racer`, derived from the decomp's
// `include/unk_structs.h`.
const RACER: RacerOffsets = RacerOffsets {
    size: 0x3A8,
    state_flags: 0x004,
    speed: 0x098,
    boost_timer: 0x21C,
    energy: 0x228,
    max_energy: 0x22C,
    race_distance: 0x23C,
    laps_completed_distance: 0x240,
    lap_distance: 0x244,
    race_distance_position: 0x248,
    race_time: 0x2A0,
    lap: 0x2A8,
    laps_completed: 0x2AA,
    position: 0x2AC,
    character: 0x2C8,
    machine_index: 0x2C9,
};

// F-Zero X game-mode ids derived from the decomp's `include/fzx_game.h`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GameMode {
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
    const fn name(self) -> &'static str {
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

    const fn is_race(self) -> bool {
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

// Racer state bits derived from the decomp's racer state flags and validated
// against live telemetry/HUD behavior in-game.
#[repr(u32)]
#[derive(Clone, Copy)]
enum RacerStateFlag {
    CollisionRecoil = 1 << 13,
    SpinningOut = 1 << 14,
    Retired = 1 << 18,
    FallingOffTrack = 1 << 19,
    CanBoost = 1 << 20,
    CpuControlled = 1 << 23,
    DashPadBoost = 1 << 24,
    Finished = 1 << 25,
    Airborne = 1 << 26,
    Crashed = 1 << 27,
    Active = 1 << 30,
}

#[derive(Clone, Debug)]
pub struct PlayerTelemetry {
    pub state_flags: u32,
    pub state_labels: Vec<&'static str>,
    pub speed_raw: f32,
    pub speed_kph: f32,
    pub energy: f32,
    pub max_energy: f32,
    pub boost_timer: i32,
    pub race_distance: f32,
    pub laps_completed_distance: f32,
    pub lap_distance: f32,
    pub race_distance_position: f32,
    pub race_time_ms: i32,
    pub lap: i16,
    pub laps_completed: i16,
    pub position: i32,
    pub character: u8,
    pub machine_index: u8,
}

#[derive(Clone, Debug)]
pub struct TelemetrySnapshot {
    pub system_ram_size: usize,
    pub game_frame_count: u32,
    pub game_mode_raw: u32,
    pub game_mode_name: String,
    pub course_index: u32,
    pub in_race_mode: bool,
    pub player: PlayerTelemetry,
}

pub fn read_snapshot(system_ram: &[u8]) -> Result<TelemetrySnapshot, CoreError> {
    if system_ram.len() < SYSTEM_RAM_SIZE_MIN {
        return Err(CoreError::MemoryOutOfRange {
            memory_id: MEMORY_SYSTEM_RAM,
            offset: 0,
            length: SYSTEM_RAM_SIZE_MIN,
            available: system_ram.len(),
        });
    }

    let player_base = GLOBALS.racers + (PLAYER_RACER_INDEX * RACER.size);
    let player_end = player_base + RACER.machine_index + 1;
    if player_end > system_ram.len() {
        return Err(CoreError::MemoryOutOfRange {
            memory_id: MEMORY_SYSTEM_RAM,
            offset: player_base,
            length: player_end - player_base,
            available: system_ram.len(),
        });
    }

    let game_mode_raw = read_u32(system_ram, GLOBALS.game_mode)?;
    let mode_id = game_mode_raw & GAME_MODE_MASK;
    let game_mode = GameMode::try_from(mode_id).ok();
    let game_mode_name = game_mode
        .map(GameMode::name)
        .unwrap_or("unknown")
        .to_owned();
    let player_state_flags = read_u32(system_ram, player_base + RACER.state_flags)?;
    let player_speed_raw = read_f32(system_ram, player_base + RACER.speed)?;
    let player = PlayerTelemetry {
        state_flags: player_state_flags,
        state_labels: decode_racer_flags(player_state_flags),
        speed_raw: player_speed_raw,
        speed_kph: player_speed_raw * SPEED_TO_KPH,
        energy: read_f32(system_ram, player_base + RACER.energy)?,
        max_energy: read_f32(system_ram, player_base + RACER.max_energy)?,
        boost_timer: read_i32(system_ram, player_base + RACER.boost_timer)?,
        race_distance: read_f32(system_ram, player_base + RACER.race_distance)?,
        laps_completed_distance: read_f32(system_ram, player_base + RACER.laps_completed_distance)?,
        lap_distance: read_f32(system_ram, player_base + RACER.lap_distance)?,
        race_distance_position: read_f32(system_ram, player_base + RACER.race_distance_position)?,
        race_time_ms: read_i32(system_ram, player_base + RACER.race_time)?,
        lap: read_i16(system_ram, player_base + RACER.lap)?,
        laps_completed: read_i16(system_ram, player_base + RACER.laps_completed)?,
        position: read_i32(system_ram, player_base + RACER.position)?,
        character: read_u8(system_ram, player_base + RACER.character)?,
        machine_index: read_u8(system_ram, player_base + RACER.machine_index)?,
    };

    Ok(TelemetrySnapshot {
        system_ram_size: system_ram.len(),
        game_frame_count: read_u32(system_ram, GLOBALS.game_frame_count)?,
        game_mode_raw,
        game_mode_name,
        course_index: read_u32(system_ram, GLOBALS.course_index)?,
        in_race_mode: game_mode.is_some_and(GameMode::is_race),
        player,
    })
}

const fn rdram_offset(vram_address: usize) -> usize {
    vram_address - KSEG0_BASE
}

fn decode_racer_flags(state_flags: u32) -> Vec<&'static str> {
    const RACER_FLAG_LABELS: &[(RacerStateFlag, &str)] = &[
        (RacerStateFlag::CollisionRecoil, "collision_recoil"),
        (RacerStateFlag::SpinningOut, "spinning_out"),
        (RacerStateFlag::Retired, "retired"),
        (RacerStateFlag::FallingOffTrack, "falling_off_track"),
        (RacerStateFlag::CanBoost, "can_boost"),
        (RacerStateFlag::CpuControlled, "cpu_controlled"),
        (RacerStateFlag::DashPadBoost, "dash_pad_boost"),
        (RacerStateFlag::Finished, "finished"),
        (RacerStateFlag::Airborne, "airborne"),
        (RacerStateFlag::Crashed, "crashed"),
        (RacerStateFlag::Active, "active"),
    ];

    RACER_FLAG_LABELS
        .iter()
        .filter_map(|(flag, label)| (state_flags & (*flag as u32) != 0).then_some(*label))
        .collect()
}

fn read_u8(memory: &[u8], offset: usize) -> Result<u8, CoreError> {
    let value = *memory
        .get(offset)
        .ok_or_else(|| memory_error(offset, size_of::<u8>(), memory.len()))?;
    Ok(value)
}

fn read_i16(memory: &[u8], offset: usize) -> Result<i16, CoreError> {
    Ok(i16::from_le_bytes(read_array(memory, offset)?))
}

fn read_i32(memory: &[u8], offset: usize) -> Result<i32, CoreError> {
    Ok(i32::from_le_bytes(read_array(memory, offset)?))
}

fn read_u32(memory: &[u8], offset: usize) -> Result<u32, CoreError> {
    Ok(u32::from_le_bytes(read_array(memory, offset)?))
}

fn read_f32(memory: &[u8], offset: usize) -> Result<f32, CoreError> {
    Ok(f32::from_le_bytes(read_array(memory, offset)?))
}

fn read_array<const N: usize>(memory: &[u8], offset: usize) -> Result<[u8; N], CoreError> {
    let end = offset + N;
    let bytes = memory
        .get(offset..end)
        .ok_or_else(|| memory_error(offset, N, memory.len()))?;
    let mut array = [0_u8; N];
    array.copy_from_slice(bytes);
    Ok(array)
}

fn memory_error(offset: usize, length: usize, available: usize) -> CoreError {
    CoreError::MemoryOutOfRange {
        memory_id: MEMORY_SYSTEM_RAM,
        offset,
        length,
        available,
    }
}

#[cfg(test)]
#[path = "tests/telemetry_tests.rs"]
mod tests;
