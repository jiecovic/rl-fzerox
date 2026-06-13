// web/run-manager/src/shared/ui/Panel.tsx
import type { ReactNode } from "react";

import { cn } from "@/shared/ui/cn";

interface PanelProps {
  children: ReactNode;
}

interface PanelHeaderProps {
  title: ReactNode;
  subtitle: ReactNode;
}

export function Panel({ children }: PanelProps) {
  return <section className="border-0 bg-transparent p-0">{children}</section>;
}

export function PanelHeader({ title, subtitle }: PanelHeaderProps) {
  return (
    <div className="mb-5">
      <h2 className="m-0 text-2xl font-semibold tracking-normal text-app-text">{title}</h2>
      <p className="mt-2 mb-0 text-app-muted">{subtitle}</p>
    </div>
  );
}

export function Notice({
  children,
  tone = "default",
}: PanelProps & { tone?: "default" | "error" }) {
  return (
    <div
      className={cn(
        "rounded-lg border border-app-border bg-app-surface p-4 text-app-muted",
        tone === "error" ? "text-app-danger" : undefined,
      )}
      role={tone === "error" ? "alert" : undefined}
    >
      {children}
    </div>
  );
}
