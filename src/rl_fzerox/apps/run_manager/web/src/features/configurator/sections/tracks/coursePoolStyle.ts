// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/tracks/coursePoolStyle.ts
import { cn } from "@/shared/ui/cn";

export const cupSwitchGroupClass =
  "inline-flex items-center gap-2 [&_.switch-button.active>span]:bg-app-surface [&_.switch-button.active]:border-[color:var(--cup-accent)] [&_.switch-button.active]:bg-[color-mix(in_srgb,var(--cup-accent)_88%,var(--surface))] [&_.switch-button]:w-[46px]";

export const courseSelectedBadgeClass =
  "pointer-events-none absolute top-2 right-2 z-10 border border-[color:var(--cup-accent)] bg-[color:var(--cup-accent)] px-2 py-0.5 text-[10px] font-bold tracking-wide text-app-accent-text uppercase";

export function courseCardClass(selected: boolean, className?: string) {
  return cn(
    "course-card relative grid min-h-[138px] gap-2 border border-app-border bg-app-surface-muted p-2.5 text-left hover:border-app-border-strong",
    selected
      ? "border-[color:var(--cup-accent)] bg-[color-mix(in_srgb,var(--cup-accent)_18%,var(--surface-muted))] shadow-[inset_0_0_0_2px_color-mix(in_srgb,var(--cup-accent)_48%,transparent),0_0_0_1px_color-mix(in_srgb,var(--cup-accent)_34%,transparent)]"
      : undefined,
    className,
  );
}
