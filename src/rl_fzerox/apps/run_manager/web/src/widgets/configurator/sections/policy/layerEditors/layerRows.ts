// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/policy/layerEditors/layerRows.ts
export function layerListResetHandler<T>(
  value: T[],
  resetValue: T[],
  onChange: (value: T[]) => void,
) {
  if (sameLayerList(value, resetValue)) {
    return undefined;
  }
  return () => onChange(resetValue);
}

export function syncLayerRowIds(rowIds: string[], length: number, label: string) {
  while (rowIds.length < length) {
    rowIds.push(`${label}-${crypto.randomUUID()}`);
  }
  rowIds.length = length;
}

function sameLayerList<T>(left: T[], right: T[]) {
  return (
    left.length === right.length && left.every((value, index) => sameValue(value, right[index]))
  );
}

function sameValue(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) {
    return true;
  }
  if (Array.isArray(left) && Array.isArray(right)) {
    return sameLayerList(left, right);
  }
  if (!isRecord(left) || !isRecord(right)) {
    return false;
  }
  const leftEntries = Object.entries(left);
  const rightEntries = Object.entries(right);
  if (leftEntries.length !== rightEntries.length) {
    return false;
  }
  return leftEntries.every(([key, value]) => sameValue(value, right[key]));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object";
}
