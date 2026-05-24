// src/rl_fzerox/apps/run_manager/web/src/shared/ui/Tabs.tsx
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
      className={
        variant === "workspace" ? "tab-strip tab-strip-workspace" : "tab-strip tab-strip-section"
      }
    >
      {items.map((item) => (
        <span className={tabShellClass(item, item.id === activeId)} key={item.id}>
          <button className="tab-item" type="button" onClick={() => onSelect(item.id)}>
            {item.label}
          </button>
          {item.closable === true && onClose !== undefined ? (
            <button
              aria-label={`Close ${item.label}`}
              className="tab-close"
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
  const classes = ["tab-shell"];
  if (active) {
    classes.push("active");
  }
  if (item.tone === "draft") {
    classes.push("tab-shell-draft");
  }
  if (item.tone === "run") {
    classes.push("tab-shell-run");
  }
  return classes.join(" ");
}
