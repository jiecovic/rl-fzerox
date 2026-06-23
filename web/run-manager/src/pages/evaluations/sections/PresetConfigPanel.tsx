// web/run-manager/src/pages/evaluations/sections/PresetConfigPanel.tsx
import { type SetStateAction, useEffect, useMemo, useState } from "react";

import {
  allBuiltInCourseIds,
  buildTrackCupViews,
  defaultSelectedCourseIds,
} from "@/entities/runConfig/ui/sections/tracks/coursePoolModel";
import {
  defaultPresetForm,
  presetFormFromPreset,
  presetRequestFromForm,
} from "@/pages/evaluations/sections/presetConfig/formModel";
import { PresetEditor } from "@/pages/evaluations/sections/presetConfig/PresetEditor";
import {
  type PresetEditorState,
  type PresetFormState,
  TARGET_MODE_LABELS,
} from "@/pages/evaluations/sections/presetConfig/types";
import type {
  ConfigMetadata,
  CreateEvaluationPresetRequest,
  EvaluationBaselineSuite,
  ManagedEvaluation,
  ManagedEvaluationPreset,
} from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { cn } from "@/shared/ui/cn";
import { formatDate } from "@/shared/ui/format";
import { CopyIcon, PlusIcon, TrashIcon } from "@/shared/ui/icons";
import { Notice } from "@/shared/ui/Panel";

interface PresetConfigPanelProps {
  baselineSuites: EvaluationBaselineSuite[];
  evaluations: ManagedEvaluation[];
  metadata: ConfigMetadata | null;
  presets: ManagedEvaluationPreset[];
  onCreatePreset: (request: CreateEvaluationPresetRequest) => Promise<ManagedEvaluationPreset>;
  onDeletePreset: (preset: ManagedEvaluationPreset) => Promise<void>;
  onGlobalError: (message: string | null) => void;
}

export function PresetConfigPanel({
  baselineSuites,
  evaluations,
  metadata,
  presets,
  onCreatePreset,
  onDeletePreset,
  onGlobalError,
}: PresetConfigPanelProps) {
  const [collapsedCupIds, setCollapsedCupIds] = useState<readonly string[]>([]);
  const [deleteRequested, setDeleteRequested] = useState(false);
  const [editor, setEditor] = useState<PresetEditorState | null>(null);
  const [editorError, setEditorError] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(presets[0]?.id ?? null);

  const allCourseIds = useMemo(
    () => (metadata === null ? [] : allBuiltInCourseIds(metadata)),
    [metadata],
  );
  const cups = useMemo(() => (metadata === null ? [] : buildTrackCupViews(metadata)), [metadata]);
  const collapsibleCupIds = useMemo(() => cups.map((cup) => cup.id), [cups]);
  const collapsedCupIdSet = useMemo(() => new Set(collapsedCupIds), [collapsedCupIds]);
  const defaultCourseIds = useMemo(
    () => (metadata === null ? [] : defaultSelectedCourseIds(metadata, allCourseIds)),
    [allCourseIds, metadata],
  );
  const suitesByPresetVersion = useMemo(
    () =>
      new Map(
        baselineSuites.map((suite) => [
          presetVersionKey(suite.preset_id, suite.preset_version),
          suite,
        ]),
      ),
    [baselineSuites],
  );
  const usedPresetKeys = useMemo(
    () =>
      new Set(
        evaluations.map((evaluation) =>
          presetVersionKey(evaluation.preset_id, evaluation.preset_version),
        ),
      ),
    [evaluations],
  );
  const selectedPreset =
    selectedPresetId === null
      ? null
      : (presets.find((preset) => preset.id === selectedPresetId) ?? null);
  const selectedSuite =
    selectedPreset === null
      ? null
      : (suitesByPresetVersion.get(presetVersionKey(selectedPreset.id, selectedPreset.version)) ??
        null);
  const selectedPresetUsed =
    selectedPreset !== null &&
    usedPresetKeys.has(presetVersionKey(selectedPreset.id, selectedPreset.version));
  const canDeleteSelected =
    selectedPreset !== null && !selectedPreset.builtin && !selectedPresetUsed;
  const canEditPresets = metadata !== null && !isSaving;

  useEffect(() => {
    if (selectedPresetId !== null && presets.some((preset) => preset.id === selectedPresetId)) {
      return;
    }
    setSelectedPresetId(presets[0]?.id ?? null);
  }, [presets, selectedPresetId]);

  useEffect(() => {
    setCollapsedCupIds([]);
    if (metadata === null) {
      setEditor(null);
    }
  }, [metadata]);

  function setCupCollapsed(cupId: string, collapsed: boolean) {
    setCollapsedCupIds((current) => {
      const next = new Set(current);
      if (collapsed) {
        next.add(cupId);
      } else {
        next.delete(cupId);
      }
      return [...next].filter((id) => collapsibleCupIds.includes(id));
    });
  }

  function setEditorForm(update: SetStateAction<PresetFormState>) {
    setEditor((current) => {
      if (current === null) {
        return null;
      }
      const nextForm = typeof update === "function" ? update(current.form) : update;
      return { ...current, form: nextForm };
    });
  }

  function startCreatePreset() {
    if (metadata === null) {
      onGlobalError("Run metadata is unavailable.");
      return;
    }
    setEditor({
      form: defaultPresetForm(metadata),
      mode: "create",
    });
    setEditorError(null);
    onGlobalError(null);
  }

  function startDuplicatePreset() {
    if (metadata === null || selectedPreset === null) {
      onGlobalError("Run metadata is unavailable.");
      return;
    }
    setEditor({
      form: presetFormFromPreset(selectedPreset, metadata, { copyName: true }),
      mode: "duplicate",
    });
    setEditorError(null);
    onGlobalError(null);
  }

  async function createPreset(request: CreateEvaluationPresetRequest) {
    onGlobalError(null);
    try {
      return await onCreatePreset(request);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "failed to create preset";
      onGlobalError(message);
      throw caught;
    }
  }

  async function savePreset() {
    if (editor === null) {
      return;
    }
    const request = presetRequestFromForm(editor.form);
    if (typeof request === "string") {
      setEditorError(request);
      return;
    }
    setIsSaving(true);
    setEditorError(null);
    try {
      const createdPreset = await createPreset(request);
      setSelectedPresetId(createdPreset.id);
      setEditor(null);
    } catch {
      setEditorError("Preset could not be saved.");
    } finally {
      setIsSaving(false);
    }
  }

  async function confirmDeletePreset() {
    if (selectedPreset === null) {
      return;
    }
    setIsDeleting(true);
    onGlobalError(null);
    try {
      await onDeletePreset(selectedPreset);
      setDeleteRequested(false);
    } catch (caught) {
      onGlobalError(caught instanceof Error ? caught.message : "failed to delete preset");
    } finally {
      setIsDeleting(false);
    }
  }

  if (presets.length === 0) {
    return <Notice>No evaluation presets are available.</Notice>;
  }

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm text-app-muted">
          {presets.length} preset{presets.length === 1 ? "" : "s"}
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            className="gap-2"
            disabled={!canEditPresets}
            variant="primary"
            onClick={startCreatePreset}
          >
            <PlusIcon />
            <span>New preset</span>
          </Button>
          <Button
            className="gap-2"
            disabled={!canEditPresets || selectedPreset === null}
            onClick={startDuplicatePreset}
          >
            <CopyIcon />
            <span>Duplicate</span>
          </Button>
          <Button
            className="gap-2"
            disabled={!canDeleteSelected || isDeleting || isSaving}
            tone="danger"
            onClick={() => setDeleteRequested(true)}
          >
            <TrashIcon />
            <span>Delete</span>
          </Button>
        </div>
      </div>

      <div className="overflow-x-auto border border-app-border bg-app-surface">
        <table className="w-full min-w-[760px] border-collapse text-left text-sm">
          <thead className="border-b border-app-border text-xs font-bold tracking-[0.04em] text-app-muted uppercase">
            <tr>
              <th className="px-4 py-3">Preset</th>
              <th className="px-4 py-3">Target</th>
              <th className="px-4 py-3">Runtime</th>
              <th className="px-4 py-3">Baseline suite</th>
            </tr>
          </thead>
          <tbody>
            {presets.map((preset) => {
              const suite = suitesByPresetVersion.get(presetVersionKey(preset.id, preset.version));
              const selected = preset.id === selectedPresetId;
              return (
                <tr
                  className={cn(
                    "cursor-pointer border-b border-app-border last:border-b-0 hover:bg-app-surface-muted",
                    selected ? "bg-app-surface-muted" : undefined,
                  )}
                  key={preset.id}
                  tabIndex={0}
                  onClick={() => setSelectedPresetId(preset.id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setSelectedPresetId(preset.id);
                    }
                  }}
                >
                  <td className="px-4 py-3 align-top">
                    <div className="grid gap-1">
                      <strong className="text-app-text">{preset.name}</strong>
                      <span className="text-xs text-app-muted">
                        {preset.id} · v{preset.version}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3 align-top text-app-muted">
                    <div className="grid gap-1">
                      <span>
                        {TARGET_MODE_LABELS[preset.target.mode]} ·{" "}
                        {preset.target.repeats_per_target}x
                      </span>
                      <span className="text-xs">{targetSelectionLabel(preset.target)}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 align-top text-app-muted">
                    <div className="grid gap-1">
                      <span>{preset.renderer}</span>
                      <span className="text-xs">seed {preset.seed}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 align-top text-app-muted">
                    {suite === undefined ? (
                      "not registered"
                    ) : (
                      <div className="grid gap-1">
                        <span className={suiteStatusClass(suite.status)}>
                          {suiteStatusLabel(suite.status)}
                        </span>
                        <span className="text-xs">{suiteTimestampLabel(suite)}</span>
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {metadata === null ? <Notice tone="error">Run metadata is unavailable.</Notice> : null}

      {editor !== null && metadata !== null ? (
        <PresetEditor
          allCourseIds={allCourseIds}
          collapsedCupIdSet={collapsedCupIdSet}
          collapsibleCupIds={collapsibleCupIds}
          cups={cups}
          defaultCourseIds={defaultCourseIds}
          editor={editor}
          error={editorError}
          isSaving={isSaving}
          metadata={metadata}
          setCollapsedCupIds={setCollapsedCupIds}
          setForm={setEditorForm}
          onCancel={() => {
            if (!isSaving) {
              setEditor(null);
            }
          }}
          onCupCollapsed={setCupCollapsed}
          onSave={() => void savePreset()}
        />
      ) : null}

      {editor === null && selectedPreset !== null && metadata !== null ? (
        <PresetEditor
          allCourseIds={allCourseIds}
          collapsedCupIdSet={collapsedCupIdSet}
          collapsibleCupIds={collapsibleCupIds}
          cups={cups}
          defaultCourseIds={defaultCourseIds}
          editor={{
            form: presetFormFromPreset(selectedPreset, metadata),
            mode: "view",
          }}
          error={null}
          isSaving={false}
          metadata={metadata}
          readOnly
          subtitle={`${selectedPreset.builtin ? "built-in" : "custom"} · ${
            selectedPresetUsed ? "referenced" : "unused"
          } · ${selectedSuite === null ? "not registered" : suiteStatusLabel(selectedSuite.status)}`}
          setCollapsedCupIds={setCollapsedCupIds}
          setForm={() => undefined}
          onCancel={() => undefined}
          onCupCollapsed={setCupCollapsed}
          onSave={() => undefined}
        />
      ) : null}

      <ConfirmDialog
        busy={isDeleting}
        confirmLabel="Delete preset"
        description={
          selectedPreset === null
            ? ""
            : `Delete preset "${selectedPreset.name}" and its baseline suite?`
        }
        open={deleteRequested}
        title="Delete preset"
        onClose={() => setDeleteRequested(false)}
        onConfirm={() => void confirmDeletePreset()}
      />
    </div>
  );
}

function presetVersionKey(presetId: string, presetVersion: number) {
  return `${presetId}\n${presetVersion}`;
}

function suiteStatusClass(status: EvaluationBaselineSuite["status"]) {
  if (status === "ready") {
    return "font-semibold text-app-accent";
  }
  if (status === "failed") {
    return "font-semibold text-app-danger";
  }
  return "text-app-muted";
}

function suiteStatusLabel(status: EvaluationBaselineSuite["status"]) {
  if (status === "not_created") {
    return "not created";
  }
  return status;
}

function suiteTimestampLabel(suite: EvaluationBaselineSuite) {
  if (suite.error_message !== null) {
    return suite.error_message;
  }
  if (suite.materialized_at !== null) {
    return `materialized ${formatDate(suite.materialized_at)}`;
  }
  if (suite.updated_at !== null) {
    return `updated ${formatDate(suite.updated_at)}`;
  }
  return "first evaluation will materialize it";
}

function targetSelectionLabel(target: ManagedEvaluationPreset["target"]) {
  const parts = [
    selectionCountLabel(target.cup_ids, "cup"),
    selectionCountLabel(target.course_ids, "course"),
    difficultySelectionLabel(target.difficulties),
  ].filter((part) => part !== null);
  return parts.length === 0 ? "all targets" : parts.join(" · ");
}

function difficultySelectionLabel(difficulties: readonly string[]) {
  if (difficulties.length === 0) {
    return null;
  }
  return difficulties.map(titleLabel).join(", ");
}

function selectionCountLabel(values: readonly string[], singular: string) {
  if (values.length === 0) {
    return null;
  }
  return `${values.length} ${pluralize(values.length, singular)}`;
}

function pluralize(count: number, singular: string) {
  return count === 1 ? singular : `${singular}s`;
}

function titleLabel(value: string) {
  return value
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(" ");
}
