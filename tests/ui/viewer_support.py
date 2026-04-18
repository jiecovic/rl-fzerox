# tests/ui/viewer_support.py
from fzerox_emulator import FZeroXTelemetry
from tests.support.native_objects import make_telemetry


def sample_telemetry(
    *,
    state_labels: tuple[str, ...] = ("active",),
    state_flags: int | None = None,
    reverse_timer: int = 0,
    boost_timer: int = 0,
    difficulty_raw: int = 0,
    difficulty_name: str = "novice",
    camera_setting_raw: int = 2,
    camera_setting_name: str = "regular",
    energy: float = 178.0,
    max_energy: float = 178.0,
    position: int = 30,
    total_racers: int = 30,
) -> FZeroXTelemetry:
    return make_telemetry(
        state_labels=state_labels,
        state_flags=state_flags,
        speed_kph=0.0,
        reverse_timer=reverse_timer,
        boost_timer=boost_timer,
        difficulty_raw=difficulty_raw,
        difficulty_name=difficulty_name,
        camera_setting_raw=camera_setting_raw,
        camera_setting_name=camera_setting_name,
        energy=energy,
        max_energy=max_energy,
        race_distance=-3040.8,
        lap_distance=75987.2,
        race_time_ms=116,
        position=position,
        total_racers=total_racers,
    )
