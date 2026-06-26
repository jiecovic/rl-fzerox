# src/rl_fzerox/core/career_mode/policy/resolver.py
"""Resolve configured policy artifacts for the active Career Mode course.

Policy runners are loaded lazily on the first handoff to a matching course, then
cached by policy-source/artifact key. This keeps Career Mode startup independent
of the size of the configured cup/course setup pool.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rl_fzerox.core.career_mode.course_setup import (
    CourseSetupTarget,
    resolve_course_setup,
)
from rl_fzerox.core.career_mode.navigation import course_id_from_info
from rl_fzerox.core.career_mode.policy.runtime import CareerModePolicyControl
from rl_fzerox.core.domain.race import CameraSettingName
from rl_fzerox.core.manager.models import (
    ManagedPolicySource,
    ManagedSaveCourseSetup,
    PolicySourceArtifact,
    PolicySourceKind,
)
from rl_fzerox.core.manager.projection.assembly import effective_train_algorithm
from rl_fzerox.core.runtime_spec.schema import CareerModeRaceSetupConfig
from rl_fzerox.core.training.inference import (
    PolicyRunner,
    load_policy_runner,
    load_policy_runner_from_paths,
)


class CareerPolicySourceStore(Protocol):
    """Store surface needed to resolve trained policy artifacts for a course."""

    def resolve_policy_source(
        self,
        *,
        policy_source_kind: PolicySourceKind,
        policy_source_id: str,
        policy_artifact: PolicySourceArtifact,
        require_policy_artifact: bool = False,
    ) -> ManagedPolicySource: ...


@dataclass(frozen=True, slots=True)
class CareerPolicyResolution:
    control: CareerModePolicyControl
    camera_setting: CameraSettingName | None
    activated_new_policy: bool


@dataclass(frozen=True, slots=True)
class _LoadedPolicy:
    source: ManagedPolicySource
    runner: PolicyRunner


class CareerPolicyResolver:
    """Resolve and cache the trained policy selected for the current course."""

    def __init__(
        self,
        *,
        store: CareerPolicySourceStore,
        setup: CareerModeRaceSetupConfig,
        course_setups: Sequence[ManagedSaveCourseSetup],
        device: str,
    ) -> None:
        self._store = store
        self._setup = setup
        self._course_setups = tuple(course_setups)
        self._device = device
        self._policy_cache: dict[tuple[str, str, str], _LoadedPolicy] = {}
        self._active_policy_key: tuple[str, str, str] | None = None

    def update_context(
        self,
        *,
        setup: CareerModeRaceSetupConfig,
        course_setups: Sequence[ManagedSaveCourseSetup],
    ) -> None:
        self._setup = setup
        self._course_setups = tuple(course_setups)
        self._active_policy_key = None

    def resolve(
        self,
        info: dict[str, object],
        *,
        refresh_artifact: bool = False,
    ) -> CareerPolicyResolution | None:
        course_setup = self.resolve_course_setup(info)
        if course_setup is None:
            return None

        key = (
            course_setup.policy_source_kind,
            course_setup.policy_source_id,
            course_setup.policy_artifact,
        )
        loaded_policy = self._load_policy(
            course_setup,
            refresh_artifact=refresh_artifact,
        )

        activated_new_policy = key != self._active_policy_key
        if activated_new_policy:
            self._active_policy_key = key
        return CareerPolicyResolution(
            control=CareerModePolicyControl(
                course_setup=course_setup,
                policy_source=loaded_policy.source,
                runner=loaded_policy.runner,
            ),
            camera_setting=validated_camera_setting(
                loaded_policy.source.config.environment.camera_setting
            ),
            activated_new_policy=activated_new_policy,
        )

    def resolve_course_setup(
        self,
        info: dict[str, object],
    ) -> ManagedSaveCourseSetup | None:
        course_id = course_id_from_info(info)
        if course_id is None:
            return None
        course_setup = resolve_course_setup(
            self._course_setups,
            CourseSetupTarget(
                difficulty=self._setup.difficulty,
                cup_id=self._setup.cup_id,
                course_id=course_id,
            ),
        )
        if course_setup is None:
            raise RuntimeError(f"no Career Mode course setup matches {course_id!r}")
        return course_setup

    def _load_policy(
        self,
        course_setup: ManagedSaveCourseSetup,
        *,
        refresh_artifact: bool = False,
    ) -> _LoadedPolicy:
        key = (
            course_setup.policy_source_kind,
            course_setup.policy_source_id,
            course_setup.policy_artifact,
        )
        loaded_policy = self._policy_cache.get(key)
        if loaded_policy is not None:
            if refresh_artifact:
                loaded_policy.runner.refresh()
            return loaded_policy

        policy_source = self._store.resolve_policy_source(
            policy_source_kind=course_setup.policy_source_kind,
            policy_source_id=course_setup.policy_source_id,
            policy_artifact=course_setup.policy_artifact,
            require_policy_artifact=True,
        )
        algorithm = effective_train_algorithm(policy_source.config)
        if policy_source.policy_path is None:
            runner = load_policy_runner(
                policy_source.source_dir,
                artifact=course_setup.policy_artifact,
                device=self._device,
                algorithm=algorithm,
            )
        else:
            runner = load_policy_runner_from_paths(
                source_dir=policy_source.source_dir,
                policy_path=policy_source.policy_path,
                model_path=policy_source.model_path,
                artifact=course_setup.policy_artifact,
                device=self._device,
                algorithm=algorithm,
            )
        loaded_policy = _LoadedPolicy(source=policy_source, runner=runner)
        self._policy_cache[key] = loaded_policy
        return loaded_policy


def validated_camera_setting(value: str | None) -> CameraSettingName | None:
    match value:
        case None:
            return None
        case "overhead" | "close_behind" | "regular" | "wide":
            return value
        case _:
            raise ValueError(f"Unsupported camera setting {value!r}")
