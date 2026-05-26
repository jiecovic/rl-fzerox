// src/rl_fzerox/apps/run_manager/web/src/shared/ui/Tabs.tsx

import { cn } from "@/shared/ui/cn";
import { CloseIcon } from "@/shared/ui/icons";

export interface TabItem<T extends string> {
  id: T;
  label: string;
  closable?: boolean;
  tone?: "draft" | "run";
}

interface TabsProps<T extends string> {
  label: string;
  activeId: T;
  items: TabItem<T>[];
  variant?: "workspace" | "section";
  onSelect: (id: T) => void;
  onClose?: (id: T) => void;
}

export function Tabs<T extends string>({
  label,
  activeId,
  items,
  variant = "workspace",
  onSelect,
  onClose,
}: TabsProps<T>) {
  return (
    <nav
      aria-label={label}
      className={cn("flex items-end gap-0", variant === "workspace" ? "mt-2.5" : "mb-[-1px]")}
    >
      {items.map((item) => (
        <span className={tabShellClass(item, item.id === activeId)} key={item.id}>
          <button
            className="min-h-[42px] border-0 bg-transparent px-4 text-[13px] text-app-muted group-hover:text-app-text group-[.is-active]:text-app-text"
            type="button"
            onClick={() => onSelect(item.id)}
          >
            {item.label}
          </button>
          {item.closable === true && onClose !== undefined ? (
            <button
              aria-label={`Close ${item.label}`}
              className="min-h-[42px] border-0 border-l border-app-border bg-transparent px-3 text-app-muted group-hover:text-app-text group-[.is-active]:text-app-text"
              type="button"
              onClick={() => onClose(item.id)}
            >
              <CloseIcon />
            </button>
          ) : null}
        </span>
      ))}
    </nav>
  );
}

function tabShellClass<T extends string>(item: TabItem<T>, active: boolean) {
  return cn(
    "group mb-[-1px] inline-flex overflow-hidden border border-b-0 border-app-border bg-app-surface-muted hover:border-app-border-strong hover:bg-[color-mix(in_srgb,var(--accent)_9%,var(--surface-muted))]",
    active
      ? "is-active border-app-border-strong border-t-app-accent bg-app-surface shadow-[inset_0_2px_0_var(--accent)]"
      : undefined,
    item.tone === "draft"
      ? "bg-[color-mix(in_srgb,var(--accent)_8%,var(--surface-muted))]"
      : undefined,
    item.tone === "run"
      ? "bg-[color-mix(in_srgb,var(--run-accent)_10%,var(--surface-muted))]"
      : undefined,
    active && item.tone === "draft"
      ? "border-t-[color-mix(in_srgb,var(--accent)_76%,white)] shadow-[inset_0_2px_0_color-mix(in_srgb,var(--accent)_76%,white)]"
      : undefined,
    active && item.tone === "run"
      ? "border-t-[color-mix(in_srgb,var(--run-accent)_76%,white)] shadow-[inset_0_2px_0_color-mix(in_srgb,var(--run-accent)_76%,white)]"
      : undefined,
  );
}
