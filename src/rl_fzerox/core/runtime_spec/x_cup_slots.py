# src/rl_fzerox/core/runtime_spec/x_cup_slots.py
"""Extract active generated X-Cup slot identities from track sampling config.

Training and UI code use these values to correlate generated-course slots,
rotations, and runtime statistics without depending on the full sampling entry
schema.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_fzerox.core.domain.courses import X_CUP_COURSE
from rl_fzerox.core.runtime_spec.schema.tracks import (
    TrackSamplingConfig,
    TrackSamplingEntryConfig,
)


@dataclass(frozen=True, slots=True)
class GeneratedXCupSlot:
    """Active generated-course identity for one X Cup sampling slot."""

    course_key: str
    slot: int
    generation: int
    course_id: str
    course_name: str
    course_hash: str
    course_seed: int
    segment_count: int | None
    course_length: float | None


def generated_x_cup_slots_from_track_sampling(
    config: TrackSamplingConfig,
) -> tuple[GeneratedXCupSlot, ...]:
    slots_by_index: dict[int, GeneratedXCupSlot] = {}
    for entry in config.entries:
        slot = generated_x_cup_slot_from_entry(entry)
        if slot is None:
            continue
        previous = slots_by_index.get(slot.slot)
        if previous is None:
            slots_by_index[slot.slot] = slot
            continue
        if _slot_identity(previous) != _slot_identity(slot):
            raise ValueError(
                "generated X Cup slot has conflicting active identities: "
                f"slot={slot.slot} {previous.course_id!r} vs {slot.course_id!r}"
            )
        slots_by_index[slot.slot] = _merge_slot_metadata(previous, slot)
    return tuple(slots_by_index[index] for index in sorted(slots_by_index))


def generated_x_cup_slot_from_entry(
    entry: TrackSamplingEntryConfig,
) -> GeneratedXCupSlot | None:
    if entry.generated_course_kind != X_CUP_COURSE.generated_kind:
        return None
    course_key = entry.runtime_course_key or entry.course_id or entry.id
    course_id = entry.course_id
    course_name = entry.course_name or entry.display_name
    slot = entry.generated_course_slot
    generation = entry.generated_course_generation
    course_hash = entry.generated_course_hash
    course_seed = entry.generated_course_seed
    if (
        course_id is None
        or course_name is None
        or slot is None
        or generation is None
        or course_hash is None
        or course_seed is None
    ):
        return None
    return GeneratedXCupSlot(
        course_key=course_key,
        slot=int(slot),
        generation=int(generation),
        course_id=course_id,
        course_name=course_name,
        course_hash=course_hash,
        course_seed=int(course_seed),
        segment_count=(
            None
            if entry.generated_course_segment_count is None
            else int(entry.generated_course_segment_count)
        ),
        course_length=(
            None if entry.generated_course_length is None else float(entry.generated_course_length)
        ),
    )


def _slot_identity(slot: GeneratedXCupSlot) -> tuple[object, ...]:
    return (
        slot.course_key,
        slot.slot,
        slot.generation,
        slot.course_id,
        slot.course_name,
        slot.course_hash,
        slot.course_seed,
    )


def _merge_slot_metadata(
    left: GeneratedXCupSlot,
    right: GeneratedXCupSlot,
) -> GeneratedXCupSlot:
    return GeneratedXCupSlot(
        course_key=left.course_key,
        slot=left.slot,
        generation=left.generation,
        course_id=left.course_id,
        course_name=left.course_name,
        course_hash=left.course_hash,
        course_seed=left.course_seed,
        segment_count=(
            left.segment_count if left.segment_count is not None else right.segment_count
        ),
        course_length=left.course_length if left.course_length is not None else right.course_length,
    )
