// rust/core/rom/tests.rs

use std::path::Path;

use super::validate_supported_rom;

fn big_endian_rom(title: &str, game_code: &str, revision: u8) -> Vec<u8> {
    let mut rom = vec![0_u8; 0x40];
    rom[0..4].copy_from_slice(&[0x80, 0x37, 0x12, 0x40]);
    write_ascii(&mut rom, 0x20, 0x34, title);
    write_ascii(&mut rom, 0x3B, 0x3F, game_code);
    rom[0x3F] = revision;
    rom
}

fn byte_swapped_16(mut rom: Vec<u8>) -> Vec<u8> {
    for chunk in rom.chunks_exact_mut(2) {
        chunk.swap(0, 1);
    }
    rom
}

fn little_endian_32(mut rom: Vec<u8>) -> Vec<u8> {
    for chunk in rom.chunks_exact_mut(4) {
        chunk.reverse();
    }
    rom
}

fn write_ascii(rom: &mut [u8], start: usize, end: usize, value: &str) {
    for (offset, byte) in value.bytes().take(end - start).enumerate() {
        rom[start + offset] = byte;
    }
}

#[test]
fn accepts_us_fzero_x_big_endian_header() {
    let rom = big_endian_rom("F-ZERO X", "CFZE", 0);

    assert!(validate_supported_rom(Path::new("fzerox.z64"), &rom).is_ok());
}

#[test]
fn accepts_us_fzero_x_byte_swapped_header() {
    let rom = byte_swapped_16(big_endian_rom("F-ZERO X", "CFZE", 0));

    assert!(validate_supported_rom(Path::new("fzerox.n64"), &rom).is_ok());
}

#[test]
fn accepts_us_fzero_x_little_endian_header() {
    let rom = little_endian_32(big_endian_rom("F-ZERO X", "CFZE", 0));

    assert!(validate_supported_rom(Path::new("fzerox.n64"), &rom).is_ok());
}

#[test]
fn rejects_pal_fzero_x_header() {
    let rom = big_endian_rom("F-ZERO X", "NFZP", 0);
    let error = validate_supported_rom(Path::new("fzerox-pal.z64"), &rom)
        .expect_err("PAL ROM should be rejected")
        .to_string();

    assert!(error.contains("game_code='NFZP'"));
    assert!(error.contains("game_code='CFZE'"));
}

#[test]
fn rejects_japanese_fzero_x_header() {
    let rom = big_endian_rom("F-ZERO X", "CFZJ", 0);
    let error = validate_supported_rom(Path::new("fzerox-jp.z64"), &rom)
        .expect_err("Japanese ROM should be rejected")
        .to_string();

    assert!(error.contains("game_code='CFZJ'"));
    assert!(error.contains("game_code='CFZE'"));
}

#[test]
fn rejects_unknown_rom_header() {
    let rom = big_endian_rom("OTHER GAME", "ABCE", 0);
    let error = validate_supported_rom(Path::new("other.z64"), &rom)
        .expect_err("non-F-Zero X ROM should be rejected")
        .to_string();

    assert!(error.contains("title='OTHER GAME'"));
}

#[test]
fn rejects_invalid_header_magic() {
    let rom = vec![0_u8; 0x40];
    let error = validate_supported_rom(Path::new("not-a-rom.bin"), &rom)
        .expect_err("invalid N64 magic should be rejected")
        .to_string();

    assert!(error.contains("not a recognized N64 ROM image"));
}
