// rust/core/rom.rs
//! ROM identity checks for the fixed-offset F-Zero X integration.

use std::path::Path;

use crate::core::error::CoreError;

const N64_HEADER_MIN_LEN: usize = 0x40;
const N64_MAGIC_BIG_ENDIAN: [u8; 4] = [0x80, 0x37, 0x12, 0x40];
const N64_MAGIC_BYTE_SWAPPED_16: [u8; 4] = [0x37, 0x80, 0x40, 0x12];
const N64_MAGIC_LITTLE_ENDIAN_32: [u8; 4] = [0x40, 0x12, 0x37, 0x80];

const TITLE_RANGE: std::ops::Range<usize> = 0x20..0x34;
const GAME_CODE_RANGE: std::ops::Range<usize> = 0x3B..0x3F;
const REVISION_OFFSET: usize = 0x3F;

const SUPPORTED_TITLE: &str = "F-ZERO X";
const SUPPORTED_GAME_CODE: &str = "CFZE";
const SUPPORTED_REVISION: u8 = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum N64ByteOrder {
    BigEndian,
    ByteSwapped16,
    LittleEndian32,
}

#[derive(Debug, Eq, PartialEq)]
pub struct RomIdentity {
    pub title: String,
    pub game_code: String,
    pub revision: u8,
}

pub fn validate_supported_rom(path: &Path, rom_bytes: &[u8]) -> Result<(), CoreError> {
    let identity = read_identity(rom_bytes).ok_or_else(|| CoreError::InvalidRomHeader {
        path: path.to_path_buf(),
    })?;
    if identity.title == SUPPORTED_TITLE
        && identity.game_code == SUPPORTED_GAME_CODE
        && identity.revision == SUPPORTED_REVISION
    {
        return Ok(());
    }
    Err(CoreError::UnsupportedRom {
        path: path.to_path_buf(),
        title: identity.title,
        game_code: identity.game_code,
        revision: identity.revision,
        expected_title: SUPPORTED_TITLE,
        expected_game_code: SUPPORTED_GAME_CODE,
        expected_revision: SUPPORTED_REVISION,
    })
}

fn read_identity(rom_bytes: &[u8]) -> Option<RomIdentity> {
    if rom_bytes.len() < N64_HEADER_MIN_LEN {
        return None;
    }
    let byte_order = byte_order(rom_bytes)?;
    let title = ascii_header_string(rom_bytes, TITLE_RANGE, byte_order);
    let game_code = ascii_header_string(rom_bytes, GAME_CODE_RANGE, byte_order);
    let revision = header_byte(rom_bytes, REVISION_OFFSET, byte_order);
    Some(RomIdentity {
        title,
        game_code,
        revision,
    })
}

fn byte_order(rom_bytes: &[u8]) -> Option<N64ByteOrder> {
    let magic = rom_bytes.get(0..4)?;
    if magic == N64_MAGIC_BIG_ENDIAN {
        Some(N64ByteOrder::BigEndian)
    } else if magic == N64_MAGIC_BYTE_SWAPPED_16 {
        Some(N64ByteOrder::ByteSwapped16)
    } else if magic == N64_MAGIC_LITTLE_ENDIAN_32 {
        Some(N64ByteOrder::LittleEndian32)
    } else {
        None
    }
}

fn ascii_header_string(
    rom_bytes: &[u8],
    range: std::ops::Range<usize>,
    byte_order: N64ByteOrder,
) -> String {
    range
        .map(|offset| header_byte(rom_bytes, offset, byte_order))
        .take_while(|byte| *byte != 0)
        .map(char::from)
        .collect::<String>()
        .trim_end()
        .to_string()
}

fn header_byte(rom_bytes: &[u8], offset: usize, byte_order: N64ByteOrder) -> u8 {
    let source_offset = match byte_order {
        N64ByteOrder::BigEndian => offset,
        N64ByteOrder::ByteSwapped16 => offset ^ 1,
        N64ByteOrder::LittleEndian32 => offset ^ 3,
    };
    rom_bytes[source_offset]
}

#[cfg(test)]
#[path = "rom/tests.rs"]
mod tests;
