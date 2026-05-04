import type { ReactNode } from "react";

interface PanelProps {
  children: ReactNode;
}

interface PanelHeaderProps {
  title: ReactNode;
  subtitle: ReactNode;
}

export function Panel({ children }: PanelProps) {
  return <section className="panel">{children}</section>;
}

export function PanelHeader({ title, subtitle }: PanelHeaderProps) {
  return (
    <div className="panel-header">
      <h2>{title}</h2>
      <p>{subtitle}</p>
    </div>
  );
}

export function Notice({
  children,
  tone = "default",
}: PanelProps & { tone?: "default" | "error" }) {
  return <div className={tone === "error" ? "notice error" : "notice"}>{children}</div>;
}
