// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/policy/layerEditors/layerRows.ts
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
    left.length === right.length && left.every((value, index) => deepEqual(value, right[index]))
  );
}

function deepEqual(left: unknown, right: unknown) {
  return JSON.stringify(left) === JSON.stringify(right);
}
