// src/rl_fzerox/apps/run_manager/web/src/features/configurator/ConfigLayout.tsx
import type { FieldsetHTMLAttributes, HTMLAttributes } from "react";

import { cn } from "@/shared/ui/cn";

type ConfigGridColumns = "two" | "three";

interface ConfigGridProps extends HTMLAttributes<HTMLDivElement> {
  columns?: ConfigGridColumns;
}

export function ConfigStack({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("grid gap-3.5", className)} {...props} />;
}

export function ConfigGrid({ className, columns, ...props }: ConfigGridProps) {
  return (
    <div
      className={cn(
        "grid gap-4",
        columns === "two" ? "grid-cols-2" : undefined,
        columns === "three" ? "grid-cols-3" : undefined,
        className,
      )}
      {...props}
    />
  );
}

export function ConfigPanelGrid({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <ConfigGrid className={cn("grid-cols-3 items-stretch", className)} {...props} />;
}

export function ConfigFieldGroup({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("grid grid-cols-1 gap-3", className)} {...props} />;
}

export function ConfigFieldset({
  className,
  ...props
}: FieldsetHTMLAttributes<HTMLFieldSetElement>) {
  return (
    <fieldset
      className={cn(
        "m-0 grid min-w-0 gap-3 border-0 p-0 disabled:opacity-[0.52]",
        "disabled:[&_.range-pair-slider]:pointer-events-none",
        className,
      )}
      {...props}
    />
  );
}
