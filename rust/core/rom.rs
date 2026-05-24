// rust/core/rom.rs
//! ROM identity checks for the fixed-offset F-Zero X integration.

use std::path::Path;

use crate::core::error::CoreError;

#[derive(Clone, Copy)]
struct N64HeaderLayout {
    min_len: usize,
    title: HeaderField,
    game_code: HeaderField,
    revision_offset: usize,
}

#[derive(Clone, Copy)]
struct HeaderField {
    start: usize,
    end: usize,
}

impl HeaderField {
    fn range(self) -> std::ops::Range<usize> {
        self.start..self.end
    }
}

#[derive(Clone, Copy)]
struct N64MagicValues {
    big_endian: [u8; 4],
    byte_swapped_16: [u8; 4],
    little_endian_32: [u8; 4],
}

#[derive(Clone, Copy)]
struct SupportedRomIdentity {
    title: &'static str,
    game_code: &'static str,
    revision: u8,
}

const N64_HEADER: N64HeaderLayout = N64HeaderLayout {
    min_len: 0x40,
    title: HeaderField {
        start: 0x20,
        end: 0x34,
    },
    game_code: HeaderField {
        start: 0x3B,
        end: 0x3F,
    },
    revision_offset: 0x3F,
};

const N64_MAGIC: N64MagicValues = N64MagicValues {
    big_endian: [0x80, 0x37, 0x12, 0x40],
    byte_swapped_16: [0x37, 0x80, 0x40, 0x12],
    little_endian_32: [0x40, 0x12, 0x37, 0x80],
};

const SUPPORTED_ROM: SupportedRomIdentity = SupportedRomIdentity {
    title: "F-ZERO X",
    game_code: "CFZE",
    revision: 0,
};

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
    if identity.title == SUPPORTED_ROM.title
        && identity.game_code == SUPPORTED_ROM.game_code
        && identity.revision == SUPPORTED_ROM.revision
    {
        return Ok(());
    }
    Err(CoreError::UnsupportedRom {
        path: path.to_path_buf(),
        title: identity.title,
        game_code: identity.game_code,
        revision: identity.revision,
        expected_title: SUPPORTED_ROM.title,
        expected_game_code: SUPPORTED_ROM.game_code,
        expected_revision: SUPPORTED_ROM.revision,
    })
}

fn read_identity(rom_bytes: &[u8]) -> Option<RomIdentity> {
    if rom_bytes.len() < N64_HEADER.min_len {
        return None;
    }
    let byte_order = byte_order(rom_bytes)?;
    let title = ascii_header_string(rom_bytes, N64_HEADER.title.range(), byte_order);
    let game_code = ascii_header_string(rom_bytes, N64_HEADER.game_code.range(), byte_order);
    let revision = header_byte(rom_bytes, N64_HEADER.revision_offset, byte_order);
    Some(RomIdentity {
        title,
        game_code,
        revision,
    })
}

fn byte_order(rom_bytes: &[u8]) -> Option<N64ByteOrder> {
    let magic = rom_bytes.get(0..4)?;
    if magic == N64_MAGIC.big_endian {
        Some(N64ByteOrder::BigEndian)
    } else if magic == N64_MAGIC.byte_swapped_16 {
        Some(N64ByteOrder::ByteSwapped16)
    } else if magic == N64_MAGIC.little_endian_32 {
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
#[path = "tests/rom_tests.rs"]
mod tests;
