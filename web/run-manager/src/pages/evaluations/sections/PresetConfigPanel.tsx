// web/run-manager/src/pages/evaluations/sections/PresetConfigPanel.tsx
import type { EvaluationBaselineSuite, ManagedEvaluationPreset } from "@/shared/api/contract";
import { formatDate } from "@/shared/ui/format";
import { Notice } from "@/shared/ui/Panel";

interface PresetConfigPanelProps {
  baselineSuites: EvaluationBaselineSuite[];
  presets: ManagedEvaluationPreset[];
}

const TARGET_MODE_LABELS: Record<ManagedEvaluationPreset["target"]["mode"], string> = {
  gp_course: "GP course",
  time_attack_course: "Time Attack course",
};

export function PresetConfigPanel({ baselineSuites, presets }: PresetConfigPanelProps) {
  if (presets.length === 0) {
    return <Notice>No evaluation presets are available.</Notice>;
  }
  const suitesByPresetVersion = new Map(
    baselineSuites.map((suite) => [presetVersionKey(suite.preset_id, suite.preset_version), suite]),
  );
  return (
    <div className="overflow-x-auto border border-app-border bg-app-surface">
      <table className="w-full min-w-[860px] border-collapse text-left text-sm">
        <thead className="border-b border-app-border text-xs font-bold tracking-[0.04em] text-app-muted uppercase">
          <tr>
            <th className="px-4 py-3">Preset</th>
            <th className="px-4 py-3">Target</th>
            <th className="px-4 py-3">Source</th>
            <th className="px-4 py-3">Runtime</th>
            <th className="px-4 py-3">Baseline suite</th>
          </tr>
        </thead>
        <tbody>
          {presets.map((preset) => {
            const suite = suitesByPresetVersion.get(presetVersionKey(preset.id, preset.version));
            return (
              <tr className="border-b border-app-border last:border-b-0" key={preset.id}>
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
                      {TARGET_MODE_LABELS[preset.target.mode]} · {preset.target.repeats_per_target}x
                    </span>
                    <span className="text-xs">{targetSelectionLabel(preset.target)}</span>
                  </div>
                </td>
                <td className="px-4 py-3 align-top text-app-muted">{preset.source_artifact}</td>
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
                      <span className={suiteStatusClass(suite.status)}>{suite.status}</span>
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
    selectionCountLabel(target.difficulties, "difficulty"),
    selectionCountLabel(target.vehicle_ids, "vehicle"),
  ].filter((part) => part !== null);
  return parts.length === 0 ? "all targets" : parts.join(" · ");
}

function selectionCountLabel(values: readonly string[], singular: string) {
  if (values.length === 0) {
    return null;
  }
  return `${values.length} ${pluralize(values.length, singular)}`;
}

function pluralize(count: number, singular: string) {
  if (count === 1) {
    return singular;
  }
  return singular === "difficulty" ? "difficulties" : `${singular}s`;
}
