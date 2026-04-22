// rust/core/host/callbacks/environment.rs
//! Libretro environment-command ids and small routing helpers.

use bitflags::bitflags;
use libretro_sys::{
    ENVIRONMENT_GET_CAN_DUPE, ENVIRONMENT_GET_CURRENT_SOFTWARE_FRAMEBUFFER,
    ENVIRONMENT_GET_LIBRETRO_PATH, ENVIRONMENT_GET_LOG_INTERFACE, ENVIRONMENT_GET_OVERSCAN,
    ENVIRONMENT_GET_SAVE_DIRECTORY, ENVIRONMENT_GET_SYSTEM_DIRECTORY, ENVIRONMENT_GET_VARIABLE,
    ENVIRONMENT_GET_VARIABLE_UPDATE, ENVIRONMENT_SET_CONTROLLER_INFO, ENVIRONMENT_SET_GEOMETRY,
    ENVIRONMENT_SET_HW_RENDER, ENVIRONMENT_SET_INPUT_DESCRIPTORS, ENVIRONMENT_SET_MEMORY_MAPS,
    ENVIRONMENT_SET_PIXEL_FORMAT, ENVIRONMENT_SET_PROC_ADDRESS_CALLBACK,
    ENVIRONMENT_SET_SUBSYSTEM_INFO, ENVIRONMENT_SET_VARIABLES,
};

const ENVIRONMENT_EXPERIMENTAL: u32 = 0x10000;

// These frontend environment command ids come from libretro.h. The
// "experimental" ones are tagged by OR-ing RETRO_ENVIRONMENT_EXPERIMENTAL
// from libretro.h.
#[repr(u32)]
pub(super) enum EnvironmentCmd {
    GetAudioVideoEnable = 47 | ENVIRONMENT_EXPERIMENTAL,
    GetFastForwarding = 49 | ENVIRONMENT_EXPERIMENTAL,
    GetTargetRefreshRate = 50 | ENVIRONMENT_EXPERIMENTAL,
    GetInputBitmasks = 51 | ENVIRONMENT_EXPERIMENTAL,
    GetCoreOptionsVersion = 52 | ENVIRONMENT_EXPERIMENTAL,
    SetCoreOptionsDisplay = 55 | ENVIRONMENT_EXPERIMENTAL,
    SetCoreOptionsUpdateDisplayCallback = 69,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum EnvironmentCommand {
    SetVariables,
    GetVariable,
    GetVariableUpdate,
    GetSystemDirectory,
    GetSaveDirectory,
    GetLibretroPath,
    GetLogInterface,
    SetPixelFormat,
    GetOverscan,
    GetCanDupe,
    GetInputBitmasks,
    GetFastForwarding,
    GetTargetRefreshRate,
    GetAudioVideoEnable,
    GetCoreOptionsVersion,
    SetGeometry,
    GetCurrentSoftwareFramebuffer,
    SetHardwareRender,
    Passthrough,
    Unknown,
}

bitflags! {
    // Flags returned for ENVIRONMENT_GET_AUDIO_VIDEO_ENABLE from libretro.h.
    pub(super) struct AudioVideoEnable: u32 {
        const VIDEO = 1 << 0;
        const AUDIO = 1 << 1;
    }
}

impl EnvironmentCmd {
    pub(super) const fn code(self) -> u32 {
        self as u32
    }
}

impl EnvironmentCommand {
    pub(super) fn from_raw(cmd: u32) -> Self {
        match cmd {
            ENVIRONMENT_SET_VARIABLES => Self::SetVariables,
            ENVIRONMENT_GET_VARIABLE => Self::GetVariable,
            ENVIRONMENT_GET_VARIABLE_UPDATE => Self::GetVariableUpdate,
            ENVIRONMENT_GET_SYSTEM_DIRECTORY => Self::GetSystemDirectory,
            ENVIRONMENT_GET_SAVE_DIRECTORY => Self::GetSaveDirectory,
            ENVIRONMENT_GET_LIBRETRO_PATH => Self::GetLibretroPath,
            ENVIRONMENT_GET_LOG_INTERFACE => Self::GetLogInterface,
            ENVIRONMENT_SET_PIXEL_FORMAT => Self::SetPixelFormat,
            ENVIRONMENT_GET_OVERSCAN => Self::GetOverscan,
            ENVIRONMENT_GET_CAN_DUPE => Self::GetCanDupe,
            ENVIRONMENT_SET_GEOMETRY => Self::SetGeometry,
            ENVIRONMENT_GET_CURRENT_SOFTWARE_FRAMEBUFFER => Self::GetCurrentSoftwareFramebuffer,
            ENVIRONMENT_SET_HW_RENDER => Self::SetHardwareRender,
            value if value == EnvironmentCmd::GetInputBitmasks.code() => Self::GetInputBitmasks,
            value if value == EnvironmentCmd::GetFastForwarding.code() => Self::GetFastForwarding,
            value if value == EnvironmentCmd::GetTargetRefreshRate.code() => {
                Self::GetTargetRefreshRate
            }
            value if value == EnvironmentCmd::GetAudioVideoEnable.code() => {
                Self::GetAudioVideoEnable
            }
            value if value == EnvironmentCmd::GetCoreOptionsVersion.code() => {
                Self::GetCoreOptionsVersion
            }
            value if is_passthrough_environment_cmd(value) => Self::Passthrough,
            _ => Self::Unknown,
        }
    }
}

pub(super) fn is_passthrough_environment_cmd(cmd: u32) -> bool {
    // These commands only advertise optional frontend capabilities; they do not
    // require any real state mutation in this host.
    matches!(
        cmd,
        ENVIRONMENT_SET_INPUT_DESCRIPTORS
            | ENVIRONMENT_SET_CONTROLLER_INFO
            | ENVIRONMENT_SET_MEMORY_MAPS
            | ENVIRONMENT_SET_PROC_ADDRESS_CALLBACK
            | ENVIRONMENT_SET_SUBSYSTEM_INFO
    ) || cmd == EnvironmentCmd::SetCoreOptionsDisplay.code()
        || cmd == EnvironmentCmd::SetCoreOptionsUpdateDisplayCallback.code()
}
