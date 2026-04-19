// rust/core/game/rng.rs
//! F-Zero X game RNG state accessors.

use libretro_sys::MEMORY_SYSTEM_RAM;

use crate::core::error::CoreError;

const KSEG0_BASE: usize = 0x8000_0000;

#[derive(Clone, Copy, Debug)]
struct GameRngRamLayout {
    seed1: usize,
    mask1: usize,
    seed2: usize,
    mask2: usize,
}

const RNG_RAM: GameRngRamLayout = GameRngRamLayout {
    seed1: rdram_offset(0x800C_D170),
    mask1: rdram_offset(0x800C_D174),
    seed2: rdram_offset(0x800C_D178),
    mask2: rdram_offset(0x800C_D17C),
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GameRngState {
    pub seed1: u32,
    pub mask1: u32,
    pub seed2: u32,
    pub mask2: u32,
}

impl GameRngState {
    pub const fn as_tuple(self) -> (u32, u32, u32, u32) {
        (self.seed1, self.mask1, self.seed2, self.mask2)
    }
}

pub fn state_from_seed(seed: u64) -> GameRngState {
    let mut stream = seed;
    GameRngState {
        seed1: splitmix64_next(&mut stream) as u32,
        mask1: splitmix64_next(&mut stream) as u32,
        seed2: splitmix64_next(&mut stream) as u32,
        mask2: splitmix64_next(&mut stream) as u32,
    }
}

pub fn read_state(memory: &[u8]) -> Result<GameRngState, CoreError> {
    Ok(GameRngState {
        seed1: read_u32(memory, RNG_RAM.seed1)?,
        mask1: read_u32(memory, RNG_RAM.mask1)?,
        seed2: read_u32(memory, RNG_RAM.seed2)?,
        mask2: read_u32(memory, RNG_RAM.mask2)?,
    })
}

pub fn write_state(memory: &mut [u8], state: GameRngState) -> Result<(), CoreError> {
    write_u32(memory, RNG_RAM.seed1, state.seed1)?;
    write_u32(memory, RNG_RAM.mask1, state.mask1)?;
    write_u32(memory, RNG_RAM.seed2, state.seed2)?;
    write_u32(memory, RNG_RAM.mask2, state.mask2)?;
    Ok(())
}

const fn rdram_offset(vram_address: usize) -> usize {
    vram_address - KSEG0_BASE
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn read_u32(memory: &[u8], offset: usize) -> Result<u32, CoreError> {
    Ok(u32::from_le_bytes(read_array(memory, offset)?))
}

fn write_u32(memory: &mut [u8], offset: usize, value: u32) -> Result<(), CoreError> {
    let available = memory.len();
    let end = offset
        .checked_add(4)
        .ok_or_else(|| memory_error(offset, 4, available))?;
    let bytes = memory
        .get_mut(offset..end)
        .ok_or_else(|| memory_error(offset, 4, available))?;
    bytes.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn read_array<const N: usize>(memory: &[u8], offset: usize) -> Result<[u8; N], CoreError> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| memory_error(offset, N, memory.len()))?;
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
mod tests {
    use super::*;

    #[test]
    fn rng_state_round_trips_through_us_ram_offsets() {
        let mut memory = vec![0_u8; 0x0030_0000];
        let state = GameRngState {
            seed1: 0x1111_2222,
            mask1: 0x3333_4444,
            seed2: 0x5555_6666,
            mask2: 0x7777_8888,
        };

        write_state(&mut memory, state).expect("state should fit mapped US RDRAM offsets");

        assert_eq!(read_state(&memory).expect("state should read back"), state);
    }

    #[test]
    fn state_from_seed_is_deterministic_and_seed_sensitive() {
        assert_eq!(state_from_seed(123), state_from_seed(123));
        assert_ne!(state_from_seed(123), state_from_seed(124));
    }
}
