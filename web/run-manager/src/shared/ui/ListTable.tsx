// web/run-manager/src/shared/ui/ListTable.tsx
import type { ReactNode } from "react";

import { cn } from "@/shared/ui/cn";

interface ListTableProps {
  children: ReactNode;
  minWidthClass?: string;
}

interface ListRowProps {
  children: ReactNode;
  selected?: boolean;
  onOpen?: () => void;
}

interface ListCheckboxCellProps {
  "aria-label": string;
  checked: boolean;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}

interface ListActionsCellProps {
  children: ReactNode;
}

export function ListTable({ children, minWidthClass = "min-w-[760px]" }: ListTableProps) {
  return (
    <div className="overflow-x-auto border border-app-border bg-app-surface">
      <table className={cn("w-full border-collapse text-left text-sm", minWidthClass)}>
        {children}
      </table>
    </div>
  );
}

export function ListTableHead({ children }: { children: ReactNode }) {
  return (
    <thead className="border-b border-app-border text-xs font-bold tracking-[0.04em] text-app-muted uppercase">
      {children}
    </thead>
  );
}

export function ListRow({ children, selected = false, onOpen }: ListRowProps) {
  return (
    <tr
      className={listRowClass(selected, onOpen !== undefined)}
      tabIndex={onOpen === undefined ? undefined : 0}
      onClick={(event) => {
        if (onOpen === undefined || isListRowInteractionTarget(event.target)) {
          return;
        }
        onOpen();
      }}
      onKeyDown={(event) => {
        if (event.target !== event.currentTarget || onOpen === undefined) {
          return;
        }
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onOpen();
        }
      }}
    >
      {children}
    </tr>
  );
}

export function ListSelectAllHeaderCell(props: ListCheckboxCellProps) {
  return (
    <th className="w-10 px-4 py-3">
      <ListCheckbox {...props} />
    </th>
  );
}

export function ListSelectionCell(props: ListCheckboxCellProps) {
  return (
    <td className="px-4 py-3 align-top" data-list-row-interaction>
      <ListCheckbox {...props} />
    </td>
  );
}

export function ListActionsHeaderCell() {
  return (
    <th className="w-12 px-4 py-3">
      <span className="sr-only">Actions</span>
    </th>
  );
}

export function ListActionsCell({ children }: ListActionsCellProps) {
  return (
    <td className="px-4 py-3 align-top" data-list-row-interaction>
      <div className="flex items-center justify-end gap-2">{children}</div>
    </td>
  );
}

function ListCheckbox({
  "aria-label": ariaLabel,
  checked,
  disabled = false,
  onChange,
}: ListCheckboxCellProps) {
  return (
    <label className="grid place-items-center" data-list-row-interaction>
      <input
        aria-label={ariaLabel}
        checked={checked}
        className="h-4 w-4 accent-app-accent"
        disabled={disabled}
        type="checkbox"
        onChange={(event) => onChange(event.currentTarget.checked)}
      />
    </label>
  );
}

function listRowClass(selected: boolean, clickable: boolean) {
  return cn(
    "border-b border-app-border transition-colors last:border-b-0 hover:bg-app-surface-muted",
    clickable
      ? "cursor-pointer focus-visible:outline focus-visible:outline-2 focus-visible:outline-app-accent"
      : undefined,
    selected ? "bg-app-surface-muted" : undefined,
  );
}

function isListRowInteractionTarget(target: EventTarget | null): boolean {
  return (
    target instanceof Element &&
    target.closest("[data-list-row-interaction],a,button,input,label,select,textarea") !== null
  );
}
