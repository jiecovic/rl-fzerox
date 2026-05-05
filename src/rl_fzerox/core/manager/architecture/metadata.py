# src/rl_fzerox/core/manager/architecture/metadata.py
from __future__ import annotations

from collections.abc import Iterable

from rl_fzerox.core.config.vehicle_catalog import CATALOG, vehicle_menu_row_and_column
from rl_fzerox.core.domain.courses import BUILT_IN_COURSES, built_in_course_refs_by_cup
from rl_fzerox.core.envs.observations.state.components import state_component_definition
from rl_fzerox.core.envs.observations.state.types import StateFeature
from rl_fzerox.core.manager.architecture.models import (
    BuiltInCourseInfo,
    EngineSettingPresetInfo,
    ObservationPresetInfo,
    RunManagerConfigMetadata,
    SelectOption,
    StateComponentInfo,
    StateFeatureInfo,
    TrackCupInfo,
    VehicleInfo,
)
from rl_fzerox.core.manager.config import (
    ManagedStateComponentConfig,
    ObservationPreset,
    default_state_components,
)


def run_manager_config_metadata() -> RunManagerConfigMetadata:
    """Return stable manager options derived from backend-supported config values."""

    return RunManagerConfigMetadata(
        observation_presets=tuple(
            ObservationPresetInfo(
                value=value,
                label=value.replace("crop_", "").replace("x", " x "),
                height=height,
                width=width,
            )
            for value, height, width in OBSERVATION_PRESET_GEOMETRIES
        ),
        track_pool_modes=(
            SelectOption(value="built_in", label="Built-in cups"),
            SelectOption(value="x_cup", label="X Cup"),
        ),
        race_modes=_options(("time_attack", "gp_race")),
        track_sampling_modes=_options(("step_balanced", "equal")),
        track_cups=tuple(cup_infos()),
        built_in_courses=tuple(course_infos()),
        vehicles=tuple(vehicle_infos()),
        engine_setting_presets=tuple(engine_setting_preset_infos()),
        steering_modes=(
            SelectOption(value="continuous", label="Continuous"),
            SelectOption(value="discrete", label="Discrete"),
        ),
        drive_modes=(
            SelectOption(value="pwm", label="Continuous PWM"),
            SelectOption(value="on_off", label="Discrete button"),
        ),
        lean_output_modes=(
            SelectOption(value="three_way", label="3-way axis"),
            SelectOption(value="independent_buttons", label="Independent buttons"),
        ),
        lean_modes=(
            SelectOption(value="release_cooldown", label="Release cooldown"),
            SelectOption(value="minimum_hold", label="Minimum hold"),
            SelectOption(value="timer_assist", label="Timer assist"),
            SelectOption(value="raw", label="Raw"),
        ),
        stack_modes=_options(("rgb", "gray", "luma_chroma")),
        resize_filters=_options(("nearest", "bilinear")),
        progress_sources=_options(("lap_progress", "segment_progress", "none")),
        action_history_controls=_options(
            ("steer", "thrust", "air_brake", "boost", "lean", "pitch")
        ),
        state_components=tuple(
            StateComponentInfo(
                name=component.name,
                label=component.name.replace("_", " "),
                features=tuple(
                    StateFeatureInfo(name=feature.name, low=feature.low, high=feature.high)
                    for feature in component_features(component)
                ),
            )
            for component in default_state_components()
        ),
        conv_profiles=_options(
            (
                "auto",
                "nature",
                "nature_32_64_128",
                "nature_wide",
                "nature_extra_k3",
                "compact_deep",
                "compact_bottleneck",
                "tiny_256",
                "custom",
            )
        ),
        activation_functions=_options(("relu", "gelu", "tanh")),
        net_arch_presets=(
            SelectOption(value="256,128", label="[256, 128]"),
            SelectOption(value="512,256", label="[512, 256]"),
            SelectOption(value="256", label="[256]"),
            SelectOption(value="128", label="[128]"),
        ),
    )


def component_features(
    component: ManagedStateComponentConfig,
    *,
    independent_lean_buttons: bool = False,
) -> tuple[StateFeature, ...]:
    settings = component.data()
    if component.name != "control_history":
        return state_component_definition(settings).features(settings)
    return state_component_definition(settings).features(
        settings,
        independent_lean_buttons=independent_lean_buttons,
    )


def preset_geometry(preset: ObservationPreset) -> tuple[int, int]:
    for value, height, width in OBSERVATION_PRESET_GEOMETRIES:
        if value == preset:
            return height, width
    raise ValueError(f"Unsupported observation preset: {preset!r}")


def _options(values: Iterable[str]) -> tuple[SelectOption, ...]:
    return tuple(SelectOption(value=value, label=value.replace("_", " ")) for value in values)


def cup_infos() -> tuple[TrackCupInfo, ...]:
    return tuple(
        TrackCupInfo(
            id=cup_id,
            label=f"{cup_id.title()} Cup",
            order=order,
            course_ids=tuple(
                course_ref.split("/", maxsplit=1)[1]
                for course_ref in built_in_course_refs_by_cup(cup_id)
            ),
        )
        for order, cup_id in enumerate(("jack", "queen", "king", "joker"))
    )


def course_infos() -> tuple[BuiltInCourseInfo, ...]:
    cup_labels = {cup.id: cup.label for cup in cup_infos()}
    return tuple(
        BuiltInCourseInfo(
            id=course.id,
            ref=course.ref,
            display_name=course.display_name,
            cup=course.cup,
            cup_label=cup_labels.get(course.cup, course.cup.title()),
            course_index=course.course_index,
        )
        for course in BUILT_IN_COURSES
    )


def vehicle_infos() -> tuple[VehicleInfo, ...]:
    return tuple(
        VehicleInfo(
            id=vehicle.id,
            display_name=vehicle.display_name,
            character_index=vehicle.character_index,
            machine_select_slot=vehicle.machine_select_slot,
            menu_row=vehicle_menu_row_and_column(vehicle.machine_select_slot)[0],
            menu_column=vehicle_menu_row_and_column(vehicle.machine_select_slot)[1],
        )
        for vehicle in CATALOG.vehicles
    )


def engine_setting_preset_infos() -> tuple[EngineSettingPresetInfo, ...]:
    return tuple(
        EngineSettingPresetInfo(
            id=preset.id,
            display_name=preset.display_name,
            raw_value=preset.raw_value,
        )
        for preset in CATALOG.engine_presets
    )


OBSERVATION_PRESET_GEOMETRIES: tuple[tuple[ObservationPreset, int, int], ...] = (
    ("crop_84x116", 84, 116),
    ("crop_92x124", 92, 124),
    ("crop_116x164", 116, 164),
    ("crop_98x130", 98, 130),
    ("crop_66x82", 66, 82),
    ("crop_60x76", 60, 76),
    ("crop_68x68", 68, 68),
    ("crop_84x84", 84, 84),
    ("crop_76x100", 76, 100),
    ("crop_64x64", 64, 64),
)
