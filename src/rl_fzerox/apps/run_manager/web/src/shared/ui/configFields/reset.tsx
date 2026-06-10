// src/rl_fzerox/apps/run_manager/web/src/shared/ui/configFields/reset.tsx
export function resetHandler<T>(value: T, resetValue: T | undefined, onChange: (value: T) => void) {
  if (resetValue === undefined || Object.is(value, resetValue)) {
    return undefined;
  }
  return () => onChange(resetValue);
}
