// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/action/ActionLayout.tsx
import type { FieldsetHTMLAttributes, HTMLAttributes } from "react";

import { cn } from "@/shared/ui/cn";

interface ActionCardProps extends HTMLAttributes<HTMLElement> {
  description: string;
  title: string;
}

interface ActionFieldGridProps extends FieldsetHTMLAttributes<HTMLFieldSetElement> {
  columns?: "single" | "two" | "three";
}

export function ActionStack({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("grid gap-3", className)} {...props} />;
}

export function ActionAuxStack({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("grid gap-6", className)} {...props} />;
}

export function ActionRuntimeStack({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "mt-1.5 grid grid-cols-1 gap-[18px] border-app-border border-t pt-[18px]",
        className,
      )}
      {...props}
    />
  );
}

export function ActionCard({ children, className, description, title, ...props }: ActionCardProps) {
  return (
    <section
      className={cn(
        "grid min-w-0 content-start gap-4 border border-app-border bg-app-surface p-3.5",
        className,
      )}
      {...props}
    >
      <ActionCardHeader description={description} title={title} />
      {children}
    </section>
  );
}

export function ActionCardHeader({ description, title }: { description: string; title: string }) {
  return (
    <div className="grid min-w-0 gap-2 pb-0.5">
      <strong className="block text-[13px] text-app-text">{title}</strong>
      <span className="block text-xs leading-[1.4] text-app-muted">{description}</span>
    </div>
  );
}

export function ActionNote({ className, ...props }: HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p className={cn("m-0 block text-xs leading-[1.4] text-app-muted", className)} {...props} />
  );
}

export function ActionFieldset({
  className,
  ...props
}: FieldsetHTMLAttributes<HTMLFieldSetElement>) {
  return <fieldset className={cn("dependent-fieldset grid min-w-0 gap-4", className)} {...props} />;
}

export function ActionFields({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("grid min-w-0 gap-4", className)} {...props} />;
}

export function ActionTripleFields({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "grid items-start gap-3 grid-cols-[repeat(auto-fit,minmax(180px,1fr))] max-[1240px]:grid-cols-1",
        className,
      )}
      {...props}
    />
  );
}

export function ActionTwoColumn({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("grid grid-cols-2 gap-3 max-[1120px]:grid-cols-1", className)} {...props} />
  );
}

export function ActionPanelGrid({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "grid items-stretch gap-3 grid-cols-[minmax(280px,0.78fr)_minmax(560px,1.22fr)] max-[1120px]:grid-cols-1",
        className,
      )}
      {...props}
    />
  );
}

export function ActionControlPanel({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "grid min-w-0 content-start gap-3 border border-app-border bg-[color-mix(in_srgb,var(--surface-muted)_40%,transparent)] p-3",
        className,
      )}
      {...props}
    />
  );
}

export function ActionControlHeader({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3", className)}
      {...props}
    />
  );
}

export function ActionFieldGrid({
  children,
  className,
  columns = "three",
  ...props
}: ActionFieldGridProps) {
  return (
    <fieldset
      className={cn(
        "dependent-fieldset grid items-start gap-3",
        actionFieldGridColumns(columns),
        className,
      )}
      {...props}
    >
      {children}
    </fieldset>
  );
}

export function ActionToggleHeader({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "grid grid-cols-[minmax(0,1fr)_84px_84px] gap-3 px-2.5 pb-0.5 text-xs font-semibold text-app-muted",
        "[&>span:nth-child(2)]:justify-self-center [&>span:nth-child(3)]:justify-self-center",
        className,
      )}
      {...props}
    />
  );
}

export function ActionToggleHeading({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return <span className={cn("inline-flex items-center gap-1.5", className)} {...props} />;
}

export function ActionToggleGrid({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("grid gap-3", className)} {...props} />;
}

export function ActionToggleRowLayout({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "grid min-h-12 grid-cols-[minmax(0,1fr)_84px_84px] items-center gap-3 border border-app-border bg-app-surface p-2.5",
        className,
      )}
      {...props}
    />
  );
}

export function ActionToggleCopy({
  children,
  className,
  description,
  title,
  ...props
}: HTMLAttributes<HTMLDivElement> & {
  description?: string;
  title?: string;
}) {
  return (
    <div className={cn("grid gap-2", className)} {...props}>
      {title === undefined ? null : <strong className="text-app-text">{title}</strong>}
      {description === undefined ? null : (
        <small className="block text-xs leading-[1.4] text-app-muted">{description}</small>
      )}
      {children}
    </div>
  );
}

export function ActionInlineToggle({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border border-app-border bg-app-surface px-3 py-2.5",
        className,
      )}
      {...props}
    />
  );
}

export function ActionSummaryGrid({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("grid grid-cols-3 gap-3 max-[1120px]:grid-cols-1", className)} {...props} />
  );
}

export function ActionSummaryItem({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "grid min-h-[72px] gap-1 border border-app-border bg-app-surface p-2.5 [&>span]:text-xs [&>span]:leading-[1.4] [&>span]:text-app-muted [&>strong]:text-[15px] [&>strong]:text-app-text",
        className,
      )}
      {...props}
    />
  );
}

function actionFieldGridColumns(columns: "single" | "two" | "three"): string {
  if (columns === "single") {
    return "grid-cols-[minmax(128px,180px)]";
  }
  if (columns === "two") {
    return "grid-cols-2 max-[1120px]:grid-cols-1";
  }
  return "grid-cols-3 max-[1120px]:grid-cols-[repeat(auto-fit,minmax(128px,1fr))]";
}
