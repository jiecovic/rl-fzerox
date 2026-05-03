export function RandomizeIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="18" viewBox="0 0 20 20" width="18">
      <path
        d="M15.2 7.2a5.6 5.6 0 0 0-9.7-1.4M4.8 12.8a5.6 5.6 0 0 0 9.7 1.4"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.7"
      />
      <path
        d="M15.6 3.8v3.6h-3.6M4.4 16.2v-3.6h3.6"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.7"
      />
    </svg>
  );
}

export function SaveDraftIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="14" viewBox="0 0 20 20" width="14">
      <path
        d="M4.5 3.5h8l3 3v10h-11zM7 3.5v5h6v-5M7 13h6"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
    </svg>
  );
}

export function UnsavedDot({ active }: { active: boolean }) {
  return (
    <span aria-hidden="true" className={active ? "dirty-action-dot active" : "dirty-action-dot"} />
  );
}
