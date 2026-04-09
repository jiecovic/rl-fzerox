// rust/core/host/runtime/rng.rs
//! Host helpers for inspecting and patching F-Zero X game RNG state.

use super::host::Host;
use crate::core::error::CoreError;
use crate::core::game::rng::{GameRngState, read_state, state_from_seed, write_state};

impl Host {
    pub fn game_rng_state(&mut self) -> Result<GameRngState, CoreError> {
        let system_ram = self.system_ram_slice()?;
        read_state(system_ram)
    }

    pub fn randomize_game_rng(&mut self, seed: u64) -> Result<GameRngState, CoreError> {
        let state = state_from_seed(seed);
        let system_ram = self.system_ram_slice_mut()?;
        write_state(system_ram, state)?;
        Ok(state)
    }
}
