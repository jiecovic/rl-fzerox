// rust/core/game/race_start/engine_physics.rs
//! Recompute live racer engine physics after reset-time engine RAM patches.

use crate::core::error::CoreError;
use crate::core::game::memory::{read_f32, read_word_swapped_i8, read_word_swapped_i16, write_f32};
use crate::core::game::race_start::invalid_setup;
use crate::core::game::telemetry::layout::{MACHINE_TABLE, RACER, RACER_ENGINE};

const ENGINE_CURVE_MAGIC: f32 = 0.689_999_8;
const ENGINE_SMOOTHING_BASE: f32 = 0.1;
const BOOST_STAT_ACCELERATION: [f32; 5] = [0.210, 0.207, 0.204, 0.201, 0.198];
const GRIP_PRIMARY_BASE: [f32; 5] = [1.277, 1.237, 1.197, 1.157, 1.117];
const GRIP_SECONDARY_BASE: [f32; 5] = [0.397, 0.367, 0.337, 0.307, 0.277];

#[derive(Clone, Copy)]
struct MachinePhysicsContext {
    weight: f32,
    boost_stat: usize,
    grip_stat: usize,
}

#[derive(Clone, Copy)]
struct WeightRange {
    light: f32,
    heavy: f32,
}

#[derive(Clone, Copy)]
struct EngineCurveRange {
    low_engine: WeightRange,
    high_engine: WeightRange,
    midpoint_bias: f32,
}

#[derive(Clone, Copy)]
struct EnginePhysicsFormula {
    acceleration_curve_high: EngineCurveRange,
    acceleration_curve_low: EngineCurveRange,
    acceleration_transition_speed: EngineCurveRange,
    boost_reserve: EngineCurveRange,
    boost_decay_denominator: EngineCurveRange,
    boost_base: EngineCurveRange,
}

#[derive(Clone, Copy)]
struct EnginePhysicsFields {
    acceleration_curve_high: f32,
    acceleration_curve_low: f32,
    acceleration_target: f32,
    acceleration_transition_speed: f32,
    boost_multiplier: f32,
    dash_multiplier: f32,
    boost_reserve: f32,
    boost_decay: f32,
    acceleration_smoothing_floor: f32,
    grip_primary: f32,
    grip_secondary: f32,
    dash_multiplier_offset: f32,
    engine_curve_bias: f32,
    acceleration_transition_scale: f32,
    acceleration_smoothing_inverse: f32,
}

// Constants are the USA decomp's ROM_READ-derived endpoints from
// `Racer_InitMachineStats`. The formula still uses the live machine table for
// weight, boost stat, and grip stat, so vehicle stats remain vehicle-specific.
const ENGINE_PHYSICS_FORMULA: EnginePhysicsFormula = EnginePhysicsFormula {
    acceleration_curve_high: EngineCurveRange {
        low_engine: WeightRange {
            light: 2.204,
            heavy: 1.700,
        },
        high_engine: WeightRange {
            light: 0.135,
            heavy: 0.131,
        },
        midpoint_bias: -0.2,
    },
    acceleration_curve_low: EngineCurveRange {
        low_engine: WeightRange {
            light: 1.702,
            heavy: 1.700,
        },
        high_engine: WeightRange {
            light: 0.135,
            heavy: 0.131,
        },
        midpoint_bias: -0.2,
    },
    acceleration_transition_speed: EngineCurveRange {
        low_engine: WeightRange {
            light: 33.0,
            heavy: 33.0,
        },
        high_engine: WeightRange {
            light: 39.0,
            heavy: 39.0,
        },
        midpoint_bias: 0.0,
    },
    boost_reserve: EngineCurveRange {
        low_engine: WeightRange {
            light: 0.066,
            heavy: 0.065,
        },
        high_engine: WeightRange {
            light: 0.490,
            heavy: 0.480,
        },
        midpoint_bias: 0.2,
    },
    boost_decay_denominator: EngineCurveRange {
        low_engine: WeightRange {
            light: 15.0,
            heavy: 10.0,
        },
        high_engine: WeightRange {
            light: 60.0,
            heavy: 50.0,
        },
        midpoint_bias: 0.2,
    },
    boost_base: EngineCurveRange {
        low_engine: WeightRange {
            light: 0.002,
            heavy: 0.000,
        },
        high_engine: WeightRange {
            light: 0.003,
            heavy: 0.001,
        },
        midpoint_bias: 0.0,
    },
};

pub(super) fn write_live_engine_fields(
    system_ram: &mut [u8],
    character_index: i16,
    engine_value: f32,
    player_base: usize,
) -> Result<(), CoreError> {
    let engine_curve = engine_to_curve_value(engine_value);
    let machine = read_machine_physics_context(system_ram, character_index, player_base)?;
    let fields = compute_engine_physics_fields(machine, engine_curve);

    write_f32(system_ram, player_base + RACER.engine_curve, engine_curve)?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.acceleration_curve_high,
        fields.acceleration_curve_high,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.acceleration_curve_low,
        fields.acceleration_curve_low,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.acceleration_target,
        fields.acceleration_target,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.acceleration_transition_speed,
        fields.acceleration_transition_speed,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.boost_multiplier,
        fields.boost_multiplier,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.dash_multiplier,
        fields.dash_multiplier,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.boost_reserve,
        fields.boost_reserve,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.boost_decay,
        fields.boost_decay,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.acceleration_smoothing_floor,
        fields.acceleration_smoothing_floor,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.grip_primary,
        fields.grip_primary,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.grip_secondary,
        fields.grip_secondary,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.dash_multiplier_offset,
        fields.dash_multiplier_offset,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.engine_curve_bias,
        fields.engine_curve_bias,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.acceleration_transition_scale,
        fields.acceleration_transition_scale,
    )?;
    write_f32(
        system_ram,
        player_base + RACER_ENGINE.acceleration_smoothing_inverse,
        fields.acceleration_smoothing_inverse,
    )
}

pub(super) fn engine_to_curve_value(engine_value: f32) -> f32 {
    if engine_value == 0.0 {
        0.0
    } else {
        1.0 / (((1.0 + ENGINE_CURVE_MAGIC) / engine_value) - ENGINE_CURVE_MAGIC)
    }
}

fn read_machine_physics_context(
    system_ram: &[u8],
    character_index: i16,
    player_base: usize,
) -> Result<MachinePhysicsContext, CoreError> {
    let machine_base =
        MACHINE_TABLE.machines + ((character_index as usize) * MACHINE_TABLE.machine_size);
    let boost_stat = read_machine_stat(system_ram, machine_base + MACHINE_TABLE.boost_stat)?;
    let grip_stat = read_machine_stat(system_ram, machine_base + MACHINE_TABLE.grip_stat)?;
    let table_weight =
        read_word_swapped_i16(system_ram, machine_base + MACHINE_TABLE.weight)? as f32;
    let live_weight = read_f32(system_ram, player_base + RACER_ENGINE.machine_weight)?;
    Ok(MachinePhysicsContext {
        weight: if table_weight > 0.0 {
            table_weight
        } else {
            live_weight
        },
        boost_stat,
        grip_stat,
    })
}

fn read_machine_stat(system_ram: &[u8], offset: usize) -> Result<usize, CoreError> {
    let stat = read_word_swapped_i8(system_ram, offset)?;
    if (0..BOOST_STAT_ACCELERATION.len() as i8).contains(&stat) {
        return Ok(stat as usize);
    }
    Err(invalid_setup(format!(
        "machine stat must be in [0, {}), got {stat}",
        BOOST_STAT_ACCELERATION.len()
    )))
}

fn compute_engine_physics_fields(
    machine: MachinePhysicsContext,
    engine_curve: f32,
) -> EnginePhysicsFields {
    let weight_ratio = (machine.weight - 780.0) / 1560.0;
    let acceleration_curve_high = ENGINE_PHYSICS_FORMULA
        .acceleration_curve_high
        .value(weight_ratio, engine_curve);
    let acceleration_curve_low = ENGINE_PHYSICS_FORMULA
        .acceleration_curve_low
        .value(weight_ratio, engine_curve);
    let acceleration_target = acceleration_target_value(weight_ratio, engine_curve);
    let acceleration_transition_speed = ENGINE_PHYSICS_FORMULA
        .acceleration_transition_speed
        .value(weight_ratio, engine_curve);
    let boost_reserve = ENGINE_PHYSICS_FORMULA
        .boost_reserve
        .value(weight_ratio, engine_curve);
    let boost_decay_denominator = ENGINE_PHYSICS_FORMULA
        .boost_decay_denominator
        .value(weight_ratio, engine_curve);
    let boost_base = ENGINE_PHYSICS_FORMULA
        .boost_base
        .value(weight_ratio, engine_curve);
    let boost_multiplier =
        (BOOST_STAT_ACCELERATION[machine.boost_stat] + boost_base) / acceleration_target;
    let dash_multiplier = (BOOST_STAT_ACCELERATION[2] + boost_base) / acceleration_target;
    let acceleration_smoothing_curve = engine_curve_to_setting_value(
        engine_curve_to_setting_value(engine_curve_to_setting_value(engine_curve)),
    );
    let acceleration_smoothing_floor =
        ((ENGINE_SMOOTHING_BASE - 1.0) * acceleration_smoothing_curve) + 1.0;

    EnginePhysicsFields {
        acceleration_curve_high,
        acceleration_curve_low,
        acceleration_target,
        acceleration_transition_speed,
        boost_multiplier,
        dash_multiplier,
        boost_reserve,
        boost_decay: boost_reserve / boost_decay_denominator,
        acceleration_smoothing_floor,
        grip_primary: grip_primary_value(
            GRIP_PRIMARY_BASE[machine.grip_stat],
            weight_ratio,
            engine_curve,
        ),
        grip_secondary: grip_secondary_value(
            GRIP_SECONDARY_BASE[machine.grip_stat],
            weight_ratio,
            engine_curve,
        ),
        dash_multiplier_offset: dash_multiplier - 1.0,
        engine_curve_bias: (0.5 - engine_curve) * 0.5,
        acceleration_transition_scale: 1.0 / (2.0 * acceleration_transition_speed),
        acceleration_smoothing_inverse: 1.0 - acceleration_smoothing_floor,
    }
}

fn engine_curve_to_setting_value(engine_curve: f32) -> f32 {
    if engine_curve == 0.0 {
        0.0
    } else {
        (1.0 + ENGINE_CURVE_MAGIC) / ((1.0 / engine_curve) + ENGINE_CURVE_MAGIC)
    }
}

fn acceleration_target_value(weight_ratio: f32, engine_curve: f32) -> f32 {
    // Match `Racer_InitMachineStats`: these values are produced by integer
    // ROM reads multiplied by `0.001f`, not by pre-rounded decimal literals.
    let low_engine = ((milli(102) - milli(100)) * weight_ratio) + milli(100);
    let high_engine = ((milli(131) - milli(129)) * weight_ratio) + milli(129);
    ((high_engine - low_engine) * engine_curve) + low_engine
}

fn grip_primary_value(base: f32, weight_ratio: f32, engine_curve: f32) -> f32 {
    let low_engine = ((0.0 - centi(2)) * weight_ratio) + centi(2);
    let high_engine = ((centi(2) - centi(4)) * weight_ratio) + centi(4);
    (base + ((high_engine - low_engine) * engine_curve)) + low_engine
}

fn grip_secondary_value(base: f32, weight_ratio: f32, engine_curve: f32) -> f32 {
    let low_engine = ((0.0 - milli(15)) * weight_ratio) + milli(15);
    let high_engine = ((milli(15) - centi(3)) * weight_ratio) + centi(3);
    (base + ((high_engine - low_engine) * engine_curve)) + low_engine
}

fn centi(value: u16) -> f32 {
    (value as f32) * 0.01
}

fn milli(value: u16) -> f32 {
    (value as f32) * 0.001
}

impl WeightRange {
    fn value(self, weight_ratio: f32) -> f32 {
        ((self.heavy - self.light) * weight_ratio) + self.light
    }
}

impl EngineCurveRange {
    fn value(self, weight_ratio: f32, engine_curve: f32) -> f32 {
        let low_engine = self.low_engine.value(weight_ratio);
        let high_engine = self.high_engine.value(weight_ratio);
        let linear = ((high_engine - low_engine) * engine_curve) + low_engine;
        let midpoint_distance = (2.0 * engine_curve) - 1.0;
        let midpoint_curve = 1.0 - (midpoint_distance * midpoint_distance);
        linear + (self.midpoint_bias * (high_engine - low_engine) * midpoint_curve)
    }
}
