// rust/core/game/rng/tests.rs

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
