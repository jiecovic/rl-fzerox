export { RandomizeIcon, SaveDraftIcon } from "@/shared/ui/icons";

export function UnsavedDot({ active }: { active: boolean }) {
  return (
    <span aria-hidden="true" className={active ? "dirty-action-dot active" : "dirty-action-dot"} />
  );
}
