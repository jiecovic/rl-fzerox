// src/rl_fzerox/apps/run_manager/web/src/app/workspace/runDetails.ts
import type { ManagedRun, ManagedRunDetail } from "@/shared/api/contract";

const MAX_CACHED_RUN_DETAILS = 16;

export function hasRunDetail(run: ManagedRun): run is ManagedRunDetail {
  return "config" in run;
}

export function rememberRunDetailAccess(order: string[], runId: string) {
  const existingIndex = order.indexOf(runId);
  if (existingIndex !== -1) {
    order.splice(existingIndex, 1);
  }
  order.push(runId);
}

export function trimRunDetailCache(
  current: Record<string, ManagedRunDetail>,
  visibleRunIds: Set<string> | null,
  accessOrder: string[],
) {
  let entries = Object.entries(current);
  if (visibleRunIds !== null) {
    entries = entries.filter(([runId]) => visibleRunIds.has(runId));
  }
  const entryIds = new Set(entries.map(([runId]) => runId));
  for (const [runId] of entries) {
    if (!accessOrder.includes(runId)) {
      accessOrder.push(runId);
    }
  }
  const prunedOrder = accessOrder.filter((runId) => entryIds.has(runId));
  accessOrder.splice(0, accessOrder.length, ...prunedOrder);
  if (entries.length <= MAX_CACHED_RUN_DETAILS) {
    return entries.length === Object.keys(current).length ? current : Object.fromEntries(entries);
  }
  const keepIds = new Set(prunedOrder.slice(-MAX_CACHED_RUN_DETAILS));
  accessOrder.splice(0, accessOrder.length, ...prunedOrder.filter((runId) => keepIds.has(runId)));
  return Object.fromEntries(entries.filter(([runId]) => keepIds.has(runId)));
}

export function sameRunDetailsById(
  left: Record<string, ManagedRunDetail>,
  right: Record<string, ManagedRunDetail>,
) {
  const leftKeys = Object.keys(left);
  const rightKeys = Object.keys(right);
  if (leftKeys.length !== rightKeys.length) {
    return false;
  }
  return leftKeys.every((key) => left[key] === right[key]);
}
