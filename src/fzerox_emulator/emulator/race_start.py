# src/fzerox_emulator/emulator/race_start.py
from __future__ import annotations

from fzerox_emulator._native import Emulator as NativeEmulator


class RaceStartMixin:
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

        if mode == "time_attack":
            self._native.patch_time_attack_race_start_setup(
                course_index=course_index,
                character_index=character_index,
                machine_skin_index=-1,
                engine_setting_raw_value=engine_setting_raw_value,
                total_lap_count=total_lap_count,
            )
            return
        if mode != "gp_race":
            raise ValueError(f"Unsupported race-start mode {mode!r}")
        self._native.patch_gp_race_start_setup(
            course_index=course_index,
            character_index=character_index,
            machine_skin_index=-1,
            engine_setting_raw_value=engine_setting_raw_value,
            total_lap_count=total_lap_count,
            gp_difficulty_raw_value=gp_difficulty_raw_value,
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

        if mode == "time_attack":
            self._native.patch_time_attack_machine_settings(
                course_index=course_index,
                character_index=character_index,
                machine_skin_index=-1,
                engine_setting_raw_value=engine_setting_raw_value,
                total_lap_count=total_lap_count,
            )
            return
        if mode != "gp_race":
            raise ValueError(f"Unsupported race-start mode {mode!r}")
        self._native.patch_gp_race_machine_settings(
            course_index=course_index,
            character_index=character_index,
            machine_skin_index=-1,
            engine_setting_raw_value=engine_setting_raw_value,
            total_lap_count=total_lap_count,
            gp_difficulty_raw_value=gp_difficulty_raw_value,
        )

    def patch_engine_settings(
        self,
        *,
        mode: str,
        engine_setting_raw_value: int,
    ) -> None:
        """Patch only engine-related globals for the already selected machine."""

        if mode == "time_attack":
            self._native.patch_time_attack_engine_settings(
                engine_setting_raw_value=engine_setting_raw_value,
            )
            return
        if mode != "gp_race":
            raise ValueError(f"Unsupported race-start mode {mode!r}")
        self._native.patch_gp_race_engine_settings(
            engine_setting_raw_value=engine_setting_raw_value,
        )

    def patch_time_attack_menu_mode(self) -> None:
        """Select the Time Attack branch in the main-menu globals."""

        self._native.patch_time_attack_menu_mode()

    def force_race_reinit(self, *, mode: str) -> None:
        """Force the game to rebuild the current race from menu globals."""

        if mode == "time_attack":
            self._native.force_time_attack_reinit()
            return
        if mode != "gp_race":
            raise ValueError(f"Unsupported race-start mode {mode!r}")
        self._native.force_gp_race_reinit()

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

        if mode == "time_attack":
            self._native.validate_time_attack_race_start_setup(
                course_index=course_index,
                character_index=character_index,
                machine_skin_index=-1,
                engine_setting_raw_value=engine_setting_raw_value,
                total_lap_count=total_lap_count,
            )
            return
        if mode != "gp_race":
            raise ValueError(f"Unsupported race-start mode {mode!r}")
        self._native.validate_gp_race_start_setup(
            course_index=course_index,
            character_index=character_index,
            machine_skin_index=-1,
            engine_setting_raw_value=engine_setting_raw_value,
            total_lap_count=total_lap_count,
            gp_difficulty_raw_value=gp_difficulty_raw_value,
        )

    def patch_time_attack_race_start_setup(
        self,
        *,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
    ) -> None:
        self.patch_race_start_setup(
            mode="time_attack",
            course_index=course_index,
            character_index=character_index,
            engine_setting_raw_value=engine_setting_raw_value,
            total_lap_count=total_lap_count,
        )

    def patch_time_attack_machine_settings(
        self,
        *,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
    ) -> None:
        self.patch_machine_settings(
            mode="time_attack",
            course_index=course_index,
            character_index=character_index,
            engine_setting_raw_value=engine_setting_raw_value,
            total_lap_count=total_lap_count,
        )

    def force_time_attack_reinit(self) -> None:
        self.force_race_reinit(mode="time_attack")

    def validate_time_attack_race_start_setup(
        self,
        *,
        course_index: int,
        character_index: int,
        engine_setting_raw_value: int,
        total_lap_count: int,
    ) -> None:
        self.validate_race_start_setup(
            mode="time_attack",
            course_index=course_index,
            character_index=character_index,
            engine_setting_raw_value=engine_setting_raw_value,
            total_lap_count=total_lap_count,
        )

    def vehicle_setup_info(self) -> dict[str, object]:
        """Return native-decoded setup info for HUD/debug checks."""

        return dict(self._native.vehicle_setup_info())
