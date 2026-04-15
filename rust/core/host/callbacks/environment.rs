// rust/core/host/callbacks/environment.rs
//! Libretro environment-command ids and small routing helpers.

use bitflags::bitflags;
use libretro_sys::{
    ENVIRONMENT_SET_CONTROLLER_INFO, ENVIRONMENT_SET_INPUT_DESCRIPTORS,
    ENVIRONMENT_SET_MEMORY_MAPS, ENVIRONMENT_SET_PROC_ADDRESS_CALLBACK,
    ENVIRONMENT_SET_SUBSYSTEM_INFO,
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
    SetHwRenderContextNegotiationInterface = 43 | ENVIRONMENT_EXPERIMENTAL,
    GetHwRenderContextNegotiationInterfaceSupport = 73 | ENVIRONMENT_EXPERIMENTAL,
    SetCoreOptionsUpdateDisplayCallback = 69,
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
