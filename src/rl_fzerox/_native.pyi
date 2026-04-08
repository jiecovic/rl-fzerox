# src/rl_fzerox/_native.pyi
from typing import Any, TypeAlias, TypedDict

class ObservationSpecDict(TypedDict):
    preset: str
    width: int
    height: int
    channels: int
    display_width: int
    display_height: int

PlayerTelemetryFlat: TypeAlias = tuple[
    int,
    tuple[str, ...],
    float,
    float,
    float,
    float,
    int,
    float,
    float,
    float,
    float,
    int,
    int,
    int,
    int,
    int,
    int,
]
TelemetryFlat: TypeAlias = tuple[
    int,
    int,
    int,
    str,
    int,
    bool,
    PlayerTelemetryFlat,
]
StepSummaryFlat: TypeAlias = tuple[
    int,
    float,
    float,
    int,
    float,
    int,
    int,
    int,
]
StepRepeatRawFlat: TypeAlias = tuple[Any, StepSummaryFlat, TelemetryFlat]


class Emulator:
    def __init__(
        self,
        core_path: str,
        rom_path: str,
        runtime_dir: str | None = None,
        baseline_state_path: str | None = None,
        renderer: str = "angrylion",
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def native_fps(self) -> float: ...
    @property
    def display_aspect_ratio(self) -> float: ...
    @property
    def frame_shape(self) -> tuple[int, int, int]: ...
    @property
    def frame_index(self) -> int: ...
    @property
    def system_ram_size(self) -> int: ...
    @property
    def baseline_kind(self) -> str: ...
    def reset(self) -> None: ...
    def step_frames(self, count: int = 1, capture_video: bool = True) -> None: ...
    def step_repeat_raw(
        self,
        action_repeat: int,
        preset: str,
        frame_stack: int,
        stuck_min_speed_kph: float,
        reverse_progress_epsilon: float,
        energy_loss_epsilon: float,
        wrong_way_progress_epsilon: float,
        joypad_mask: int = 0,
        left_stick_x: float = 0.0,
        left_stick_y: float = 0.0,
        right_stick_x: float = 0.0,
        right_stick_y: float = 0.0,
    ) -> StepRepeatRawFlat: ...
    def set_controller_state(
        self,
        joypad_mask: int = 0,
        left_stick_x: float = 0.0,
        left_stick_y: float = 0.0,
        right_stick_x: float = 0.0,
        right_stick_y: float = 0.0,
    ) -> None: ...
    def save_state(self, path: str) -> None: ...
    def capture_current_as_baseline(self, path: str | None = None) -> None: ...
    def frame_rgb(self) -> bytes: ...
    def observation_spec(self, preset: str) -> ObservationSpecDict: ...
    def frame_observation(self, preset: str, frame_stack: int) -> Any: ...
    def frame_display(self, preset: str) -> Any: ...
    def telemetry(self) -> dict[str, object]: ...
    def telemetry_flat(self) -> TelemetryFlat: ...
    def read_system_ram(self, offset: int, length: int) -> bytes: ...
    def close(self) -> None: ...

class CoreInfo:
    api_version: int
    library_name: str
    library_version: str
    valid_extensions: list[str]
    requires_full_path: bool
    blocks_extract: bool

def probe_core(core_path: str) -> CoreInfo: ...
def joypad_mask(*buttons: int) -> int: ...

JOYPAD_B: int
JOYPAD_Y: int
JOYPAD_SELECT: int
JOYPAD_START: int
JOYPAD_UP: int
JOYPAD_DOWN: int
JOYPAD_LEFT: int
JOYPAD_RIGHT: int
JOYPAD_A: int
JOYPAD_X: int
JOYPAD_L: int
JOYPAD_R: int
JOYPAD_L2: int
JOYPAD_R2: int
JOYPAD_L3: int
JOYPAD_R3: int
