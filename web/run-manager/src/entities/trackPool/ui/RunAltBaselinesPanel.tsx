// web/run-manager/src/entities/trackPool/ui/RunAltBaselinesPanel.tsx
import { useCallback, useEffect, useMemo, useState } from "react";

import { fetchRunAltBaselines } from "@/shared/api/client";
import type { ConfigMetadata, ManagedRunDetail, RunAltBaseline } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { ConfigDisclosure } from "@/shared/ui/config/ConfigDisclosure";
import { formatDate } from "@/shared/ui/format";
import { TrashIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

interface RunAltBaselinesPanelProps {
  clearingCourseKey?: string | null;
  isClearingAll: boolean;
  metadata: ConfigMetadata;
  onClearAll: () => void | Promise<void>;
  onClearCourse?: (courseKey: string) => void | Promise<void>;
  run: ManagedRunDetail;
}

interface PendingClear {
  count: number;
  courseKey: string | null;
  label: string;
}

interface CourseBaselineGroup {
  baselines: RunAltBaseline[];
  courseKey: string;
  courseLabel: string;
  cupId: string;
  cupLabel: string;
  difficultyLabel: string;
}

interface CupBaselineGroup {
  count: number;
  courses: CourseBaselineGroup[];
  cupId: string;
  cupLabel: string;
}

export function RunAltBaselinesPanel({
  clearingCourseKey = null,
  isClearingAll,
  metadata,
  onClearAll,
  onClearCourse,
  run,
}: RunAltBaselinesPanelProps) {
  const [open, setOpen] = useState(false);
  const [baselines, setBaselines] = useState<RunAltBaseline[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [pendingClear, setPendingClear] = useState<PendingClear | null>(null);
  const cupGroups = useMemo(
    () => groupAltBaselines(baselines ?? [], metadata),
    [baselines, metadata],
  );
  const activeCount = run.active_alt_baseline_count;
  const title =
    activeCount === 1 ? "Alt baselines (1)" : `Alt baselines (${activeCount.toLocaleString()})`;

  const loadBaselines = useCallback(
    async (signal?: AbortSignal) => {
      setLoadError(null);
      try {
        setBaselines(await fetchRunAltBaselines(run.id, signal === undefined ? {} : { signal }));
      } catch (caught: unknown) {
        if (isAbortError(caught)) {
          return;
        }
        setLoadError(caught instanceof Error ? caught.message : "failed to load alt baselines");
      }
    },
    [run.id],
  );

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    if (activeCount === 0) {
      setBaselines([]);
      return undefined;
    }
    const abortController = new AbortController();
    void loadBaselines(abortController.signal);
    return () => {
      abortController.abort();
    };
  }, [activeCount, loadBaselines, open]);

  async function clearPendingBaselines() {
    const pending = pendingClear;
    if (pending === null) {
      return;
    }
    setPendingClear(null);
    if (pending.courseKey === null) {
      await onClearAll();
    } else {
      await onClearCourse?.(pending.courseKey);
    }
    await loadBaselines();
  }

  return (
    <>
      <ConfigDisclosure defaultOpen={false} open={open} title={title} onToggle={setOpen}>
        <div className="grid gap-3">
          <div className="flex flex-wrap justify-end gap-2">
            <Button
              className="h-8 gap-1.5 px-3 text-xs"
              disabled={activeCount === 0 || isClearingAll}
              tone="danger"
              type="button"
              onClick={() =>
                setPendingClear({
                  count: activeCount,
                  courseKey: null,
                  label: "all courses",
                })
              }
            >
              <TrashIcon />
              <span>{isClearingAll ? "Clearing" : "Clear all"}</span>
            </Button>
          </div>
          {loadError !== null ? (
            <p className="m-0 text-sm text-app-danger">{loadError}</p>
          ) : baselines === null ? (
            <p className="m-0 text-sm text-app-muted">Loading alt baselines...</p>
          ) : cupGroups.length === 0 ? (
            <p className="m-0 text-sm text-app-muted">No active alt baselines.</p>
          ) : (
            <div className="grid gap-2">
              {cupGroups.map((group) => (
                <CupAltBaselineGroup
                  clearingCourseKey={clearingCourseKey}
                  group={group}
                  key={group.cupId}
                  onClearCourse={onClearCourse}
                  onRequestClearCourse={(course) =>
                    setPendingClear({
                      count: course.baselines.length,
                      courseKey: course.courseKey,
                      label: course.courseLabel,
                    })
                  }
                />
              ))}
            </div>
          )}
        </div>
      </ConfigDisclosure>
      <ConfirmDialog
        busy={
          pendingClear?.courseKey === null
            ? isClearingAll
            : pendingClear?.courseKey === clearingCourseKey
        }
        confirmLabel={
          pendingClear?.courseKey === null ? "Clear alt baselines" : "Clear course baselines"
        }
        description={
          pendingClear === null
            ? ""
            : pendingClear.courseKey === null
              ? `Delete ${formatBaselineCount(pendingClear.count)} for "${run.name}"? This removes the recorded state files from disk.`
              : `Delete ${formatBaselineCount(pendingClear.count)} for ${pendingClear.label}? This removes the recorded state files from disk.`
        }
        open={pendingClear !== null}
        title={pendingClear?.courseKey === null ? "Clear alt baselines" : "Clear course baselines"}
        onClose={() => setPendingClear(null)}
        onConfirm={() => void clearPendingBaselines()}
      />
    </>
  );
}

function CupAltBaselineGroup({
  clearingCourseKey,
  group,
  onClearCourse,
  onRequestClearCourse,
}: {
  clearingCourseKey: string | null;
  group: CupBaselineGroup;
  onClearCourse?: (courseKey: string) => void | Promise<void>;
  onRequestClearCourse: (course: CourseBaselineGroup) => void;
}) {
  return (
    <section className="grid gap-1.5 border-t border-app-border pt-2 first:border-t-0 first:pt-0">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <strong className="text-app-muted">{group.cupLabel}</strong>
        <span className="tabular-nums text-app-muted">{formatBaselineCount(group.count)}</span>
      </div>
      <div className="divide-y divide-app-border/70">
        {group.courses.map((course) => (
          <CourseAltBaselineRow
            clearing={clearingCourseKey === course.courseKey}
            course={course}
            key={course.courseKey}
            onClearCourse={
              onClearCourse === undefined ? undefined : () => onRequestClearCourse(course)
            }
          />
        ))}
      </div>
    </section>
  );
}

function CourseAltBaselineRow({
  clearing,
  course,
  onClearCourse,
}: {
  clearing: boolean;
  course: CourseBaselineGroup;
  onClearCourse?: () => void;
}) {
  const latest = course.baselines.at(-1) ?? null;

  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-3 py-2 text-sm">
      <div className="min-w-0">
        <strong className="block truncate text-app-text">{course.courseLabel}</strong>
        <div className="truncate text-xs text-app-muted">
          {latest === null
            ? course.difficultyLabel
            : `${course.difficultyLabel} · ${latest.label} · ${formatDate(latest.created_at)}`}
        </div>
      </div>
      <span className="text-xs tabular-nums text-app-muted">
        {course.baselines.length.toLocaleString()}
      </span>
      <TooltipIconButton
        aria-label={`Clear alt baselines for ${course.courseLabel}`}
        disabled={onClearCourse === undefined || clearing}
        size="small"
        tone="danger"
        tooltip={clearing ? "Clearing" : "Clear course baselines"}
        onClick={onClearCourse}
      >
        <TrashIcon />
      </TooltipIconButton>
    </div>
  );
}

function groupAltBaselines(
  baselines: readonly RunAltBaseline[],
  metadata: ConfigMetadata,
): CupBaselineGroup[] {
  const courseGroups = courseAltBaselineGroups(baselines, metadata);
  const cups = new Map<string, CupBaselineGroup>();
  for (const course of courseGroups) {
    const existing = cups.get(course.cupId);
    if (existing === undefined) {
      cups.set(course.cupId, {
        count: course.baselines.length,
        courses: [course],
        cupId: course.cupId,
        cupLabel: course.cupLabel,
      });
    } else {
      existing.count += course.baselines.length;
      existing.courses.push(course);
    }
  }
  return [...cups.values()];
}

function courseAltBaselineGroups(
  baselines: readonly RunAltBaseline[],
  metadata: ConfigMetadata,
): CourseBaselineGroup[] {
  const coursesById = new Map(metadata.built_in_courses.map((course) => [course.id, course]));
  const cupsById = new Map(metadata.track_cups.map((cup) => [cup.id, cup]));
  const difficultyLabels = new Map(
    metadata.gp_difficulties.map((difficulty) => [difficulty.value, difficulty.label]),
  );
  const grouped = new Map<string, RunAltBaseline[]>();
  for (const baseline of baselines) {
    const existing = grouped.get(baseline.course_key);
    if (existing === undefined) {
      grouped.set(baseline.course_key, [baseline]);
    } else {
      existing.push(baseline);
    }
  }
  return [...grouped.entries()]
    .map(([courseKey, items]) => {
      const course = coursesById.get(courseKey) ?? null;
      const cup = course === null ? null : (cupsById.get(course.cup) ?? null);
      return {
        baselines: [...items].sort(
          (left, right) =>
            left.created_at.localeCompare(right.created_at) || left.id.localeCompare(right.id),
        ),
        courseKey,
        courseLabel: course?.display_name ?? humanizeKey(courseKey),
        cupId: cup?.id ?? "other",
        cupLabel: course?.cup_label ?? cup?.label ?? "Other courses",
        difficultyLabel: baselineDifficultyLabel(items, difficultyLabels),
      };
    })
    .sort((left, right) => {
      const leftCupOrder = cupsById.get(left.cupId)?.order ?? Number.MAX_SAFE_INTEGER;
      const rightCupOrder = cupsById.get(right.cupId)?.order ?? Number.MAX_SAFE_INTEGER;
      if (leftCupOrder !== rightCupOrder) {
        return leftCupOrder - rightCupOrder;
      }
      const leftCourseOrder =
        coursesById.get(left.courseKey)?.course_index ?? Number.MAX_SAFE_INTEGER;
      const rightCourseOrder =
        coursesById.get(right.courseKey)?.course_index ?? Number.MAX_SAFE_INTEGER;
      if (leftCourseOrder !== rightCourseOrder) {
        return leftCourseOrder - rightCourseOrder;
      }
      return left.courseLabel.localeCompare(right.courseLabel);
    });
}

function formatBaselineCount(count: number) {
  return count === 1 ? "1 alt baseline" : `${count.toLocaleString()} alt baselines`;
}

function baselineDifficultyLabel(
  baselines: readonly RunAltBaseline[],
  difficultyLabels: ReadonlyMap<string, string>,
) {
  const labels: string[] = [];
  for (const baseline of baselines) {
    const difficulty = resetVariantDifficulty(baseline.reset_variant_key);
    if (difficulty === null) {
      continue;
    }
    const label = difficultyLabels.get(difficulty) ?? humanizeKey(difficulty);
    if (!labels.includes(label)) {
      labels.push(label);
    }
  }
  if (labels.length === 0) {
    return "difficulty unknown";
  }
  if (labels.length <= 2) {
    return labels.join(", ");
  }
  return `${labels.slice(0, 2).join(", ")} +${labels.length - 2}`;
}

function resetVariantDifficulty(resetVariantKey: string) {
  for (const part of resetVariantKey.split("|")) {
    const [key, value] = part.split("=", 2);
    if (key === "gp_difficulty" && value !== undefined && value.length > 0) {
      return value;
    }
  }
  const [, legacyDifficulty] = resetVariantKey.split("|");
  return legacyDifficulty === undefined || legacyDifficulty.length === 0 ? null : legacyDifficulty;
}

function humanizeKey(value: string) {
  return value
    .split("_")
    .filter((part) => part.length > 0)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function isAbortError(error: unknown) {
  return error instanceof DOMException && error.name === "AbortError";
}
