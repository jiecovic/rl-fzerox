export function ResetIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="12" viewBox="0 0 24 24" width="12">
      <path
        d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
      <path
        d="M3 3v5h5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
    </svg>
  );
}

export function resetHandler<T>(value: T, resetValue: T | undefined, onChange: (value: T) => void) {
  if (resetValue === undefined || Object.is(value, resetValue)) {
    return undefined;
  }
  return () => onChange(resetValue);
}
