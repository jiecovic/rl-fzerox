# src/rl_fzerox/ui/watch/runtime/career_mode/recording/summary/models.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from rl_fzerox.ui.watch.runtime.career_mode.recording.summary.extract import (
    _course_engine_observation,
    _course_result,
    _course_result_identity,
    _course_result_signature,
    _existing_course_result_index,
    _merge_missing_course_result_fields,
    _merge_selected_summary_info,
    _policy_checkpoint_signature,
    _policy_checkpoint_summary,
    _selected_summary_info,
    last_finished_attempt_status,
)
from rl_fzerox.ui.watch.runtime.career_mode.recording.summary.values import utc_timestamp


@dataclass(frozen=True, slots=True)
class _SegmentSummarySnapshot:
    segment_index: int
    label: str
    attempt_id: str | None
    status: str | None
    source_path: Path
    started_at_utc: str
    closed_at_utc: str
    frame_count: int
    course_results: tuple[dict[str, object], ...]
    final_info: dict[str, object]
    policy_checkpoints: tuple[dict[str, object], ...] = ()


@dataclass(slots=True)
class _SegmentSummaryBuilder:
    segment_index: int
    label: str
    attempt_id: str | None
    started_at_utc: str
    frame_count: int = 0
    status: str | None = None
    final_info: dict[str, object] | None = None
    course_results: list[dict[str, object]] = field(default_factory=list)
    policy_checkpoints: list[dict[str, object]] = field(default_factory=list)
    observed_course_engine_raw_values: dict[tuple[str, object], int] = field(default_factory=dict)
    last_course_result_signature: tuple[object, ...] | None = None
    last_policy_checkpoint_signature: tuple[object, ...] | None = None

    def observe(self, info: Mapping[str, object]) -> None:
        self.frame_count += 1
        self.observe_event(info)

    def observe_event(self, info: Mapping[str, object]) -> None:
        if status := last_finished_attempt_status(info):
            self.status = status
        self.final_info = _merge_selected_summary_info(
            self.final_info,
            _selected_summary_info(info),
        )
        if engine_observation := _course_engine_observation(info):
            identity, engine_raw = engine_observation
            self.observed_course_engine_raw_values[identity] = engine_raw
        course_result = _course_result(info)
        if course_result is not None:
            self._apply_observed_course_engine(course_result)
            existing_index = _existing_course_result_index(self.course_results, course_result)
            if existing_index is not None:
                self.course_results[existing_index] = _merge_missing_course_result_fields(
                    self.course_results[existing_index],
                    course_result,
                )
                self.last_course_result_signature = _course_result_signature(course_result)
            else:
                signature = _course_result_signature(course_result)
                if signature != self.last_course_result_signature:
                    self.course_results.append(course_result)
                    self.last_course_result_signature = signature
        policy_checkpoint = _policy_checkpoint_summary(info)
        if policy_checkpoint is None:
            return
        signature = _policy_checkpoint_signature(policy_checkpoint)
        if signature == self.last_policy_checkpoint_signature:
            return
        self.policy_checkpoints.append(policy_checkpoint)
        self.last_policy_checkpoint_signature = signature

    def _apply_observed_course_engine(self, course_result: dict[str, object]) -> None:
        identity = _course_result_identity(course_result)
        if identity is None:
            return
        engine_raw = self.observed_course_engine_raw_values.get(identity)
        if engine_raw is None:
            return
        course_result["engine_setting_raw_value"] = engine_raw

    def snapshot(self, *, source_path: Path, status: str | None) -> _SegmentSummarySnapshot:
        return _SegmentSummarySnapshot(
            segment_index=self.segment_index,
            label=self.label,
            attempt_id=self.attempt_id,
            status=status or self.status,
            source_path=source_path,
            started_at_utc=self.started_at_utc,
            closed_at_utc=utc_timestamp(),
            frame_count=self.frame_count,
            course_results=tuple(dict(result) for result in self.course_results),
            policy_checkpoints=tuple(dict(checkpoint) for checkpoint in self.policy_checkpoints),
            final_info=dict(self.final_info or {}),
        )
