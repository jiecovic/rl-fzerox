# tests/core/save_game/test_unlocks.py

import pytest

from rl_fzerox.core.save_game import FZeroXSaveDecodeError, decode_fzerox_unlock_state
from rl_fzerox.core.save_game.unlocks import FZEROX_SAVE_LAYOUT


def test_decode_raw_sra_gp_progress() -> None:
    save_data = _logical_sra({"jack": 1, "queen": 2, "king": 4})

    unlock_state = decode_fzerox_unlock_state(save_data)

    assert unlock_state.gp_cup_cleared(difficulty="novice", cup_id="jack")
    assert not unlock_state.gp_cup_cleared(difficulty="standard", cup_id="jack")
    assert unlock_state.gp_cup_cleared(difficulty="standard", cup_id="queen")
    assert unlock_state.gp_cup_cleared(difficulty="master", cup_id="king")
    assert not unlock_state.gp_cup_cleared(difficulty="novice", cup_id="joker")


def test_decode_libretro_srm_gp_progress_with_byteswapped_sra_payload() -> None:
    sra_payload = _logical_sra({"jack": 4, "queen": 3, "king": 2, "joker": 1})
    srm_payload = bytearray(FZEROX_SAVE_LAYOUT.libretro_srm_size)
    srm_payload[
        FZEROX_SAVE_LAYOUT.libretro_sra_offset : FZEROX_SAVE_LAYOUT.libretro_sra_offset
        + FZEROX_SAVE_LAYOUT.raw_sra_size
    ] = _byteswap_words(sra_payload)

    unlock_state = decode_fzerox_unlock_state(bytes(srm_payload))

    assert unlock_state.gp_cup_cleared(difficulty="master", cup_id="jack")
    assert unlock_state.gp_cup_cleared(difficulty="expert", cup_id="queen")
    assert unlock_state.gp_cup_cleared(difficulty="standard", cup_id="king")
    assert unlock_state.gp_cup_cleared(difficulty="novice", cup_id="joker")
    assert not unlock_state.gp_cup_cleared(difficulty="standard", cup_id="joker")


def test_decode_rejects_unknown_save_bytes() -> None:
    with pytest.raises(FZeroXSaveDecodeError):
        decode_fzerox_unlock_state(bytes(FZEROX_SAVE_LAYOUT.raw_sra_size))


def _logical_sra(cup_progress: dict[str, int]) -> bytes:
    payload = bytearray(FZEROX_SAVE_LAYOUT.raw_sra_size)
    payload[: len(FZEROX_SAVE_LAYOUT.title)] = FZEROX_SAVE_LAYOUT.title
    for progress_offset in FZEROX_SAVE_LAYOUT.gp_progress_offsets:
        payload[progress_offset.offset] = cup_progress.get(progress_offset.cup_id, 0)
    return bytes(payload)


def _byteswap_words(payload: bytes) -> bytes:
    word_size = FZEROX_SAVE_LAYOUT.byteswap_word_size
    chunks = (
        payload[index : index + word_size][::-1] for index in range(0, len(payload), word_size)
    )
    return b"".join(chunks)
