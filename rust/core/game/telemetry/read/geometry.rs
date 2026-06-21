// rust/core/game/telemetry/read/geometry.rs
//! Racer geometry decoding and future-local segment sampling.

use std::mem::size_of;

use crate::core::error::CoreError;
use crate::core::game::memory::{read_f32, read_i32, read_u32};
use crate::core::telemetry::layout::{COURSE_SEGMENT, RACER, RACER_SEGMENT_POSITION_INFO};
use crate::core::telemetry::model::{RacerGeometryTelemetry, outside_track_bounds};

use super::scalars::{Vec3, dot_vec3, kseg0_pointer_to_offset, read_vec3, read_vec3_magnitude};

#[derive(Clone, Copy, Debug)]
struct FutureLocalSegmentScanConfig {
    future_segment_count: usize,
    samples_per_segment: usize,
}

#[derive(Clone, Copy, Debug)]
struct NearestFutureSegment {
    index: i32,
    distance: f32,
}

const FUTURE_LOCAL_SEGMENT_SCAN: FutureLocalSegmentScanConfig = FutureLocalSegmentScanConfig {
    future_segment_count: 3,
    samples_per_segment: 5,
};

pub(super) fn read_racer_geometry(
    system_ram: &[u8],
    player_base: usize,
) -> Result<RacerGeometryTelemetry, CoreError> {
    let segment_info_base = player_base + RACER.segment_position_info;
    let current_segment_offset = read_current_segment_offset(system_ram, segment_info_base)?;
    let segment_index = read_segment_index(system_ram, current_segment_offset)?;
    let world_pos = read_vec3(
        system_ram,
        segment_info_base + RACER_SEGMENT_POSITION_INFO.pos,
    )?;
    let segment_center = read_vec3(
        system_ram,
        segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_pos,
    )?;
    let signed_lateral_offset =
        read_signed_lateral_offset(system_ram, player_base, segment_info_base)?;
    let current_radius_left = read_f32(system_ram, player_base + RACER.current_radius_left)?;
    let current_radius_right = read_f32(system_ram, player_base + RACER.current_radius_right)?;
    let future_local_segment = future_local_nearest_segment(
        system_ram,
        current_segment_offset,
        world_pos,
        signed_lateral_offset,
        current_radius_left,
        current_radius_right,
    )?;
    Ok(RacerGeometryTelemetry {
        segment_index,
        segment_t: read_f32(
            system_ram,
            segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_t_value,
        )?,
        segment_length_proportion: read_f32(
            system_ram,
            segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_length_proportion,
        )?,
        world_pos_x: world_pos.x,
        world_pos_y: world_pos.y,
        world_pos_z: world_pos.z,
        segment_center_x: segment_center.x,
        segment_center_y: segment_center.y,
        segment_center_z: segment_center.z,
        local_lateral_velocity: read_f32(system_ram, player_base + RACER.local_velocity)?,
        signed_lateral_offset,
        lateral_distance: read_f32(
            system_ram,
            segment_info_base + RACER_SEGMENT_POSITION_INFO.distance_from_segment,
        )?,
        lateral_displacement_magnitude: read_vec3_magnitude(
            system_ram,
            segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_displacement,
        )?,
        current_radius_left,
        current_radius_right,
        height_above_ground: read_f32(system_ram, player_base + RACER.height_above_ground)?,
        future_local_nearest_segment_index: future_local_segment.map(|segment| segment.index),
        future_local_nearest_segment_distance: future_local_segment
            .map_or(0.0, |segment| segment.distance),
        velocity_magnitude: read_vec3_magnitude(system_ram, player_base + RACER.velocity)?,
        acceleration_magnitude: read_vec3_magnitude(system_ram, player_base + RACER.acceleration)?,
        acceleration_force: read_f32(system_ram, player_base + RACER.acceleration_force)?,
        drift_attack_force: read_f32(system_ram, player_base + RACER.drift_attack_force)?,
        collision_mass: read_f32(system_ram, player_base + RACER.colliding_strength)?,
    })
}

pub(super) fn read_signed_lateral_offset(
    system_ram: &[u8],
    player_base: usize,
    segment_info_base: usize,
) -> Result<f32, CoreError> {
    // The decomp's edge checks project segment displacement onto segmentBasis.z.
    // Positive is left of the spline centerline; negative is right.
    dot_vec3(
        system_ram,
        segment_info_base + RACER_SEGMENT_POSITION_INFO.segment_displacement,
        player_base + RACER.segment_basis + 0x18,
    )
}

fn future_local_nearest_segment(
    system_ram: &[u8],
    current_segment_offset: Option<usize>,
    world_pos: Vec3,
    signed_lateral_offset: f32,
    current_radius_left: f32,
    current_radius_right: f32,
) -> Result<Option<NearestFutureSegment>, CoreError> {
    if !outside_track_bounds(
        signed_lateral_offset,
        current_radius_left,
        current_radius_right,
    ) {
        return Ok(None);
    }

    let Some(mut segment_offset) = current_segment_offset else {
        return Ok(None);
    };
    let mut best: Option<NearestFutureSegment> = None;
    for _ in 0..=FUTURE_LOCAL_SEGMENT_SCAN.future_segment_count {
        if !valid_segment_base(system_ram, segment_offset) {
            return Ok(best);
        }
        if let Some(candidate) = nearest_sample_on_segment(system_ram, segment_offset, world_pos)? {
            best = Some(match best {
                Some(previous) if previous.distance <= candidate.distance => previous,
                _ => candidate,
            });
        }
        let Some(next_segment_offset) =
            read_segment_pointer(system_ram, segment_offset + COURSE_SEGMENT.next)?
        else {
            return Ok(best);
        };
        segment_offset = next_segment_offset;
    }
    Ok(best)
}

fn nearest_sample_on_segment(
    system_ram: &[u8],
    segment_offset: usize,
    world_pos: Vec3,
) -> Result<Option<NearestFutureSegment>, CoreError> {
    let Some(index) = read_segment_index(system_ram, Some(segment_offset))? else {
        return Ok(None);
    };

    let mut best_distance = f32::INFINITY;
    let denominator = (FUTURE_LOCAL_SEGMENT_SCAN.samples_per_segment - 1).max(1) as f32;
    for sample_index in 0..FUTURE_LOCAL_SEGMENT_SCAN.samples_per_segment {
        let t = sample_index as f32 / denominator;
        let Some(sample_pos) = sample_segment_spline_position(system_ram, segment_offset, t)?
        else {
            continue;
        };
        best_distance = best_distance.min(euclidean_distance(world_pos, sample_pos));
    }
    if best_distance.is_finite() {
        Ok(Some(NearestFutureSegment {
            index,
            distance: best_distance,
        }))
    } else {
        Ok(None)
    }
}

fn sample_segment_spline_position(
    system_ram: &[u8],
    segment_offset: usize,
    t: f32,
) -> Result<Option<Vec3>, CoreError> {
    let Some(prev_segment_offset) =
        read_segment_pointer(system_ram, segment_offset + COURSE_SEGMENT.prev)?
    else {
        return Ok(None);
    };
    let Some(next_segment_offset) =
        read_segment_pointer(system_ram, segment_offset + COURSE_SEGMENT.next)?
    else {
        return Ok(None);
    };
    let Some(next_next_segment_offset) =
        read_segment_pointer(system_ram, next_segment_offset + COURSE_SEGMENT.next)?
    else {
        return Ok(None);
    };
    if !valid_segment_base(system_ram, prev_segment_offset)
        || !valid_segment_base(system_ram, next_segment_offset)
        || !valid_segment_base(system_ram, next_next_segment_offset)
    {
        return Ok(None);
    }

    let prev_pos = read_segment_pos(system_ram, prev_segment_offset)?;
    let pos = read_segment_pos(system_ram, segment_offset)?;
    let next_pos = read_segment_pos(system_ram, next_segment_offset)?;
    let next_next_pos = read_segment_pos(system_ram, next_next_segment_offset)?;
    let tension = read_f32(system_ram, segment_offset + COURSE_SEGMENT.tension)?;
    let t_square = t * t;
    let t_cube = t_square * t;
    let prev_weight = (2.0 * t_square - t - t_cube) * tension;
    let current_weight = (2.0 - tension).mul_add(t_cube, (tension - 3.0) * t_square) + 1.0;
    let next_weight =
        (tension - 2.0).mul_add(t_cube, (3.0 - 2.0 * tension) * t_square) + (tension * t);
    let next_next_weight = (t_cube - t_square) * tension;
    Ok(Some(Vec3 {
        x: prev_weight.mul_add(
            prev_pos.x,
            current_weight.mul_add(
                pos.x,
                next_weight.mul_add(next_pos.x, next_next_weight * next_next_pos.x),
            ),
        ),
        y: prev_weight.mul_add(
            prev_pos.y,
            current_weight.mul_add(
                pos.y,
                next_weight.mul_add(next_pos.y, next_next_weight * next_next_pos.y),
            ),
        ),
        z: prev_weight.mul_add(
            prev_pos.z,
            current_weight.mul_add(
                pos.z,
                next_weight.mul_add(next_pos.z, next_next_weight * next_next_pos.z),
            ),
        ),
    }))
}

fn valid_segment_base(system_ram: &[u8], segment_offset: usize) -> bool {
    segment_offset + COURSE_SEGMENT.prev + size_of::<u32>() <= system_ram.len()
}

fn euclidean_distance(lhs: Vec3, rhs: Vec3) -> f32 {
    let x = lhs.x - rhs.x;
    let y = lhs.y - rhs.y;
    let z = lhs.z - rhs.z;
    x.mul_add(x, y.mul_add(y, z * z)).sqrt()
}

fn read_current_segment_offset(
    system_ram: &[u8],
    segment_info_base: usize,
) -> Result<Option<usize>, CoreError> {
    let pointer = read_u32(
        system_ram,
        segment_info_base + RACER_SEGMENT_POSITION_INFO.course_segment,
    )?;
    Ok(kseg0_pointer_to_offset(pointer, system_ram.len()))
}

fn read_segment_index(
    system_ram: &[u8],
    segment_offset: Option<usize>,
) -> Result<Option<i32>, CoreError> {
    let Some(segment_offset) = segment_offset else {
        return Ok(None);
    };
    let segment_index_offset = segment_offset + COURSE_SEGMENT.segment_index;
    if segment_index_offset + size_of::<i32>() > system_ram.len() {
        return Ok(None);
    }
    Ok(Some(read_i32(system_ram, segment_index_offset)?))
}

fn read_segment_pointer(system_ram: &[u8], offset: usize) -> Result<Option<usize>, CoreError> {
    if offset + size_of::<u32>() > system_ram.len() {
        return Ok(None);
    }
    Ok(kseg0_pointer_to_offset(
        read_u32(system_ram, offset)?,
        system_ram.len(),
    ))
}

fn read_segment_pos(system_ram: &[u8], segment_offset: usize) -> Result<Vec3, CoreError> {
    read_vec3(system_ram, segment_offset + COURSE_SEGMENT.pos)
}
