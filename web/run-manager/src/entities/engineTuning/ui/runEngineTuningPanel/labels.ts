// web/run-manager/src/entities/engineTuning/ui/runEngineTuningPanel/labels.ts
import type {
  ConfigMetadata,
  EngineTuningRuntimeCandidate,
  EngineTuningRuntimeContext,
  EngineTuningRuntimeState,
} from "@/shared/api/contract";

import { humanizeKey } from "./format";

export interface EngineTuningLabels {
  courses: ReadonlyMap<string, string>;
  vehicles: ReadonlyMap<string, string>;
}

export type EngineTuningViewMode = "bandit" | "aggregate" | "model";

export function backendLabel(backend: EngineTuningRuntimeState["model_backend"]) {
  if (backend === "mlp_ensemble") {
    return "MLP ensemble (experimental)";
  }
  if (backend === "gaussian_process") {
    return "GP (experimental)";
  }
  if (backend === "bandit") {
    return "Bandit";
  }
  return "no model";
}

export function contextLabel(context: EngineTuningRuntimeContext, labels: EngineTuningLabels) {
  const courseLabel = labels.courses.get(context.course_key) ?? humanizeKey(context.course_key);
  const vehicleLabel = labels.vehicles.get(context.vehicle_id) ?? humanizeKey(context.vehicle_id);
  return `${courseLabel} · ${vehicleLabel}`;
}

export function engineTuningLabels(metadata: ConfigMetadata): EngineTuningLabels {
  return {
    courses: new Map([
      ...metadata.built_in_courses.map((course) => [course.id, course.display_name] as const),
      ["x_cup", "X Cup"],
    ]),
    vehicles: new Map(
      metadata.vehicles.map((vehicle) => [vehicle.id, vehicle.display_name] as const),
    ),
  };
}

export function engineTuningViewMode(
  backend: EngineTuningRuntimeState["model_backend"],
): EngineTuningViewMode {
  if (backend === "bandit") {
    return "bandit";
  }
  if (backend === "mlp_ensemble") {
    return "model";
  }
  return "aggregate";
}

export function measuredCandidatesForContext(
  candidates: readonly EngineTuningRuntimeCandidate[],
  contextKey: string,
) {
  return candidates.filter(
    (candidate) => candidate.context_key === contextKey && candidate.finish_count > 0,
  );
}

export function objectiveCountLabel(
  context: EngineTuningRuntimeContext,
  objective: EngineTuningRuntimeState["objective"],
) {
  if (objective === "safe_finish_time" || objective === "finish_rate") {
    return `${context.episode_count.toLocaleString()} episodes`;
  }
  return `${context.finish_count.toLocaleString()} successful finishes`;
}

export function sortedContexts(
  contexts: readonly EngineTuningRuntimeContext[],
  labels: EngineTuningLabels,
) {
  return [...contexts].sort((left, right) => {
    const labelOrder = contextLabel(left, labels).localeCompare(contextLabel(right, labels));
    return labelOrder === 0 ? left.context_key.localeCompare(right.context_key) : labelOrder;
  });
}
