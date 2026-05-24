# src/fzerox_emulator/emulator/race_start.py
"""Race-start patching methods mixed into the concrete emulator wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fzerox_emulator._native import Emulator as NativeEmulator
    from fzerox_emulator.boundary import RaceStartRequestDict


class RaceStartMixin:
    """Python method surface for native-owned race-start RAM patching."""

    _native: NativeEmulator

    def patch_race_start_setup(
        self,
        *,
        mode: str,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
        gp_difficulty_raw_value: int = -1,
    ) -> None:
        """Patch one live race start using native-owned RAM layout rules."""

        self._native.patch_race_start_setup(
            _race_start_request(
                mode=mode,
                course_index=course_index,
                character_index=character_index,
                engine_setting_raw_value=engine_setting_raw_value,
                total_lap_count=total_lap_count,
                gp_difficulty_raw_value=gp_difficulty_raw_value,
            )
        )

    def patch_machine_settings(
        self,
        *,
        mode: str,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
        gp_difficulty_raw_value: int = -1,
    ) -> None:
        """Patch menu-level machine settings before race initialization."""

        self._native.patch_machine_settings(
            _race_start_request(
                mode=mode,
                course_index=course_index,
                character_index=character_index,
                engine_setting_raw_value=engine_setting_raw_value,
                total_lap_count=total_lap_count,
                gp_difficulty_raw_value=gp_difficulty_raw_value,
            )
        )

    def patch_engine_settings(
        self,
        *,
        mode: str,
        engine_setting_raw_value: int,
    ) -> None:
        """Patch only engine-related globals for the already selected machine."""

        self._native.patch_engine_settings(mode, engine_setting_raw_value)

    def patch_time_attack_menu_mode(self) -> None:
        """Select the Time Attack branch in the main-menu globals."""

        self._native.patch_time_attack_menu_mode()

    def force_race_reinit(self, *, mode: str) -> None:
        """Force the game to rebuild the current race from menu globals."""

        self._native.force_race_reinit(mode)

    def validate_race_start_setup(
        self,
        *,
        mode: str,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
        gp_difficulty_raw_value: int = -1,
    ) -> None:
        """Validate that the native race-start RAM view matches the requested setup."""

        self._native.validate_race_start_setup(
            _race_start_request(
                mode=mode,
                course_index=course_index,
                character_index=character_index,
                engine_setting_raw_value=engine_setting_raw_value,
                total_lap_count=total_lap_count,
                gp_difficulty_raw_value=gp_difficulty_raw_value,
            )
        )

    def vehicle_setup_info(self) -> dict[str, object]:
        """Return native-decoded setup info for HUD/debug checks."""

        return dict(self._native.vehicle_setup_info())


def _race_start_request(
    *,
    mode: str,
    course_index: int,
    character_index: int,
    engine_setting_raw_value: int,
    total_lap_count: int,
    gp_difficulty_raw_value: int,
) -> RaceStartRequestDict:
    return {
        "mode": mode,
        "course_index": course_index,
        "character_index": character_index,
        "machine_skin_index": -1,
        "engine_setting_raw_value": engine_setting_raw_value,
        "total_lap_count": total_lap_count,
        "gp_difficulty_raw_value": gp_difficulty_raw_value,
    }
