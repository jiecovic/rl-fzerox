// rust/core/host/runtime/race_start.rs
//! Race-start patching and validation methods on the host runtime.

use crate::core::error::CoreError;
use crate::core::game::race_start::{RaceStartMode, RaceStartSetup, VehicleSetupInfo};

use super::host::Host;

impl Host {
    pub fn patch_time_attack_menu_mode(&mut self) -> Result<(), CoreError> {
        let system_ram = self.system_ram_slice_mut()?;
        crate::core::game::race_start::write_time_attack_menu_mode(system_ram)
    }

    pub fn patch_race_start_setup(
        &mut self,
        mode: RaceStartMode,
        setup: RaceStartSetup,
    ) -> Result<(), CoreError> {
        let system_ram = self.system_ram_slice_mut()?;
        crate::core::game::race_start::write_race_setup(system_ram, mode, setup)
    }

    pub fn patch_machine_settings(
        &mut self,
        mode: RaceStartMode,
        setup: RaceStartSetup,
    ) -> Result<(), CoreError> {
        let system_ram = self.system_ram_slice_mut()?;
        crate::core::game::race_start::write_machine_settings(system_ram, mode, setup)
    }

    pub fn patch_engine_settings(
        &mut self,
        mode: RaceStartMode,
        engine_setting_raw_value: i32,
    ) -> Result<(), CoreError> {
        let system_ram = self.system_ram_slice_mut()?;
        crate::core::game::race_start::write_engine_settings(
            system_ram,
            mode,
            engine_setting_raw_value,
        )
    }

    pub fn force_race_reinit(&mut self, mode: RaceStartMode) -> Result<(), CoreError> {
        let system_ram = self.system_ram_slice_mut()?;
        crate::core::game::race_start::force_race_reinit(system_ram, mode)
    }

    pub fn validate_race_start_setup(
        &mut self,
        mode: RaceStartMode,
        setup: RaceStartSetup,
    ) -> Result<(), CoreError> {
        let system_ram = self.system_ram_slice()?;
        crate::core::game::race_start::validate_race_setup(system_ram, mode, setup)
    }

    pub fn vehicle_setup_info(&mut self) -> Result<VehicleSetupInfo, CoreError> {
        let system_ram = self.system_ram_slice()?;
        crate::core::game::race_start::vehicle_setup_info(system_ram)
    }
}
