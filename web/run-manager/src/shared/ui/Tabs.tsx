// web/run-manager/src/shared/ui/Tabs.tsx

import { cn } from "@/shared/ui/cn";
import {
  CareerTabIcon,
  ChartIcon,
  CloseIcon,
  DraftTabIcon,
  EvaluationTabIcon,
  ImportIcon,
  RunTabIcon,
} from "@/shared/ui/icons";

export interface TabItem<T extends string> {
  activity?: "running";
  id: T;
  label: string;
  closable?: boolean;
  icon?: "career" | "charts" | "checkpoint" | "draft" | "evaluation" | "run";
  shortLabel?: string;
  tone?: "draft" | "run";
}

interface TabsProps<T extends string> {
  label: string;
  activeId: T;
  items: readonly TabItem<T>[];
  variant?: "sidebar" | "workspace" | "section";
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
    <nav aria-label={label} className={tabsNavClass(variant)}>
      {items.map((item) => (
        <span className={tabShellClass(item, item.id === activeId, variant)} key={item.id}>
          <button
            aria-label={variant === "sidebar" ? item.label : undefined}
            className={tabButtonClass(variant)}
            type="button"
            onClick={() => onSelect(item.id)}
          >
            <span className={tabContentClass(variant)}>
              <TabIcon icon={item.icon} />
              <TabActivity activity={item.activity} />
              <span className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">
                {variant === "sidebar" ? (item.shortLabel ?? item.label) : item.label}
              </span>
            </span>
          </button>
          {item.closable === true && onClose !== undefined ? (
            <button
              aria-label={`Close ${item.label}`}
              className={tabCloseButtonClass(variant)}
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

function tabsNavClass(variant: TabsProps<string>["variant"]) {
  if (variant === "sidebar") {
    return "flex flex-col gap-1";
  }
  return cn(
    "flex min-w-0 items-end gap-0 overflow-hidden",
    variant === "workspace" ? "mt-2.5" : "mb-[-1px]",
  );
}

function TabIcon({ icon }: { icon: TabItem<string>["icon"] }) {
  switch (icon) {
    case "career":
      return <CareerTabIcon />;
    case "charts":
      return <ChartIcon />;
    case "checkpoint":
      return <ImportIcon />;
    case "draft":
      return <DraftTabIcon />;
    case "evaluation":
      return <EvaluationTabIcon />;
    case "run":
      return <RunTabIcon />;
    default:
      return null;
  }
}

function TabActivity({ activity }: { activity: TabItem<string>["activity"] }) {
  if (activity !== "running") {
    return null;
  }
  return (
    <>
      <span
        aria-hidden="true"
        className="h-2 w-2 shrink-0 rounded-full bg-app-accent shadow-[0_0_8px_color-mix(in_srgb,var(--accent)_72%,transparent)]"
      />
      <span className="sr-only">running</span>
    </>
  );
}

function tabContentClass(variant: TabsProps<string>["variant"]) {
  if (variant === "sidebar") {
    return "inline-flex min-w-0 items-center gap-2";
  }
  return "inline-flex min-w-0 items-center gap-2";
}

function tabButtonClass(variant: TabsProps<string>["variant"]) {
  if (variant === "sidebar") {
    return "min-h-[44px] w-full border-0 bg-transparent px-3 text-left text-[13px] text-app-muted group-hover:text-app-text group-[.is-active]:text-app-text";
  }
  return "min-h-[42px] min-w-0 border-0 bg-transparent px-4 text-[13px] text-app-muted group-hover:text-app-text group-[.is-active]:text-app-text";
}

function tabCloseButtonClass(variant: TabsProps<string>["variant"]) {
  if (variant === "sidebar") {
    return "min-h-[46px] border-0 border-l border-app-border bg-transparent px-3 text-app-muted group-hover:text-app-text group-[.is-active]:text-app-text";
  }
  return "min-h-[42px] border-0 border-l border-app-border bg-transparent px-3 text-app-muted group-hover:text-app-text group-[.is-active]:text-app-text";
}

function tabShellClass<T extends string>(
  item: TabItem<T>,
  active: boolean,
  variant: TabsProps<string>["variant"],
) {
  if (variant === "sidebar") {
    return cn(
      "group flex w-full overflow-hidden border border-app-border bg-app-surface-muted hover:border-app-border-strong hover:bg-[color-mix(in_srgb,var(--accent)_9%,var(--surface-muted))]",
      active
        ? "is-active border-app-border-strong bg-app-surface shadow-[inset_3px_0_0_var(--accent)]"
        : undefined,
    );
  }
  return cn(
    "group mb-[-1px] inline-flex min-w-0 max-w-full overflow-hidden border border-b-0 border-app-border bg-app-surface-muted hover:border-app-border-strong hover:bg-[color-mix(in_srgb,var(--accent)_9%,var(--surface-muted))]",
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
