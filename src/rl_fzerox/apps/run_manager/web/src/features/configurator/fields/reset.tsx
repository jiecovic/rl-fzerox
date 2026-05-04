export { ResetIcon } from "@/shared/ui/icons";

export function resetHandler<T>(value: T, resetValue: T | undefined, onChange: (value: T) => void) {
  if (resetValue === undefined || Object.is(value, resetValue)) {
    return undefined;
  }
  return () => onChange(resetValue);
}
