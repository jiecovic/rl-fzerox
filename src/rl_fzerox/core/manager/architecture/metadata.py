# src/rl_fzerox/core/manager/architecture/metadata.py
from __future__ import annotations

from collections.abc import Iterable

from rl_fzerox.core.domain.camera import CAMERA_SETTINGS
from rl_fzerox.core.domain.courses import BUILT_IN_COURSES, built_in_course_refs_by_cup
from rl_fzerox.core.domain.observation_components import (
    ObservationStateComponentName,
    state_feature_default_enabled,
)
from rl_fzerox.core.domain.observation_image import (
    OBSERVATION_IMAGE_GEOMETRY,
)
from rl_fzerox.core.domain.race_difficulty import race_difficulty_names
from rl_fzerox.core.envs.observations.state.components import state_component_definition
from rl_fzerox.core.envs.observations.state.types import StateFeature
from rl_fzerox.core.manager.architecture.models import (
    BuiltInCourseInfo,
    ObservationPresetInfo,
    ObservationResolutionBounds,
    ObservationSourceGeometryInfo,
    RunManagerConfigMetadata,
    SelectOption,
    StateComponentInfo,
    StateFeatureInfo,
    TrackCupInfo,
    VehicleInfo,
)
from rl_fzerox.core.manager.run_spec import (
    ManagedStateComponentConfig,
    default_state_components,
)
from rl_fzerox.core.policy.auxiliary_state.targets import (
    auxiliary_state_target_name_for_feature,
    auxiliary_state_target_supports_grounded_only,
)
from rl_fzerox.core.runtime_spec.vehicle_catalog import CATALOG, vehicle_menu_row_and_column


def run_manager_config_metadata() -> RunManagerConfigMetadata:
    """Return stable manager options derived from backend-supported config values."""

    return RunManagerConfigMetadata(
        observation_presets=tuple(
            ObservationPresetInfo(
                value=geometry.name,
                label=_observation_preset_label(geometry.name),
                height=geometry.height,
                width=geometry.width,
            )
            for geometry in OBSERVATION_IMAGE_GEOMETRY.presets
        ),
        observation_resolution_bounds=ObservationResolutionBounds(
            min_dimension=OBSERVATION_IMAGE_GEOMETRY.custom_bounds.min_dimension,
            max_height=OBSERVATION_IMAGE_GEOMETRY.custom_bounds.max_height,
            max_width=OBSERVATION_IMAGE_GEOMETRY.custom_bounds.max_width,
        ),
        observation_source_geometries=tuple(
            ObservationSourceGeometryInfo(
                renderer=geometry.renderer,
                height=geometry.height,
                width=geometry.width,
            )
            for geometry in OBSERVATION_IMAGE_GEOMETRY.source_geometries
        ),
        camera_settings=tuple(
            SelectOption(value=setting.name, label=setting.name.replace("_", " "))
            for setting in CAMERA_SETTINGS
        ),
        race_modes=_options(("time_attack", "gp_race")),
        gp_difficulties=tuple(
            SelectOption(value=name, label=name.title()) for name in race_difficulty_names()
        ),
        track_sampling_modes=_options(
            (
                "step_balanced",
                "deficit_budget",
                "fixed_env",
                "equal",
            )
        ),
        track_cups=tuple(cup_infos()),
        built_in_courses=tuple(course_infos()),
        vehicles=tuple(vehicle_infos()),
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
            SelectOption(value="four_way_categorical", label="4-way categorical"),
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
                    _state_feature_info(feature, component_name=component.name)
                    for feature in component_features(component)
                ),
            )
            for component in default_state_components()
        ),
        conv_profiles=_conv_profile_options(),
        activation_functions=_options(("relu", "gelu", "tanh")),
        net_arch_presets=(
            SelectOption(value="256,128", label="[256, 128]"),
            SelectOption(value="512,256", label="[512, 256]"),
            SelectOption(value="256", label="[256]"),
            SelectOption(value="128", label="[128]"),
        ),
    )


def _observation_preset_label(preset_name: str) -> str:
    if preset_name == "crop_84x84":
        return "84 x 84 DQN/Atari"
    if preset_name == "crop_72x96":
        return "72 x 96 IMPALA"
    return preset_name.replace("crop_", "").replace("x", " x ")


def _conv_profile_options() -> tuple[SelectOption, ...]:
    return (
        SelectOption(value="nature", label="Nature CNN"),
        SelectOption(value="impala_small", label="IMPALA small"),
        SelectOption(value="impala_large", label="IMPALA large"),
        SelectOption(value="custom", label="Custom"),
    )


def component_features(
    component: ManagedStateComponentConfig,
    *,
    split_lean_history: bool = False,
) -> tuple[StateFeature, ...]:
    settings = component.data()
    if component.name != "control_history":
        return state_component_definition(settings).features(settings)
    return state_component_definition(settings).features(
        settings,
        split_lean_history=split_lean_history,
    )


def _state_feature_info(
    feature: StateFeature,
    *,
    component_name: ObservationStateComponentName,
) -> StateFeatureInfo:
    auxiliary_target_name = auxiliary_state_target_name_for_feature(feature.name)
    return StateFeatureInfo(
        # Metadata lists every currently supported row, including opt-in entries.
        name=feature.name,
        low=feature.low,
        high=feature.high,
        default_enabled=state_feature_default_enabled(component_name, feature.name),
        auxiliary_target_name=auxiliary_target_name,
        auxiliary_supports_grounded_only=(
            False
            if auxiliary_target_name is None
            else auxiliary_state_target_supports_grounded_only(auxiliary_target_name)
        ),
    )


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
        for vehicle in sorted(CATALOG.vehicles, key=lambda item: item.machine_select_slot)
    )
