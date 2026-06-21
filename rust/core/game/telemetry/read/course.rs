// rust/core/game/telemetry/read/course.rs
//! Current-course metadata decoding from live RAM.

use std::mem::size_of;

use crate::core::error::CoreError;
use crate::core::game::memory::{read_f32, read_i32, read_u32};
use crate::core::telemetry::layout::{COURSE_INFO, GLOBALS};

use super::scalars::kseg0_pointer_to_offset;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CurrentCourseInfo {
    pub(super) segment_count: i32,
    pub(super) length: f32,
}

pub(super) fn read_current_course_info(system_ram: &[u8]) -> Result<CurrentCourseInfo, CoreError> {
    let pointer = read_u32(system_ram, GLOBALS.current_course_info)?;
    let Some(course_info_offset) = kseg0_pointer_to_offset(pointer, system_ram.len()) else {
        return Ok(CurrentCourseInfo::default());
    };
    let segment_count_offset = course_info_offset + COURSE_INFO.segment_count;
    let length_offset = course_info_offset + COURSE_INFO.length;
    if segment_count_offset + size_of::<i32>() > system_ram.len()
        || length_offset + size_of::<f32>() > system_ram.len()
    {
        return Ok(CurrentCourseInfo::default());
    }
    Ok(CurrentCourseInfo {
        segment_count: read_i32(system_ram, segment_count_offset)?,
        length: read_f32(system_ram, length_offset)?,
    })
}
