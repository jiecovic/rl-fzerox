// web/run-manager/src/pages/evaluations/sections/presetConfig/types.ts
import type { EvaluationMode, ManagedEvaluationPreset } from "@/shared/api/contract";

export interface PresetFormState {
  courseIds: readonly string[];
  difficulties: readonly string[];
  name: string;
  renderer: ManagedEvaluationPreset["renderer"];
  repeatsPerTarget: string;
  seed: string;
  targetMode: EvaluationMode;
}

export interface PresetEditorState {
  mode: "create" | "duplicate" | "view";
  form: PresetFormState;
}

export const TARGET_MODE_LABELS: Record<EvaluationMode, string> = {
  gp_course: "GP course",
  time_attack_course: "Time Attack course",
};
