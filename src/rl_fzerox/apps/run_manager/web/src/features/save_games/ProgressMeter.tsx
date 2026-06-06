// src/rl_fzerox/apps/run_manager/web/src/features/save_games/ProgressMeter.tsx
import { cn } from "@/shared/ui/cn";

interface ProgressMeterProps {
  className?: string;
  label: string;
  value: number;
}

export function ProgressMeter({ className, label, value }: ProgressMeterProps) {
  const percent = Math.round(clampUnit(value) * 100);
  return (
    <div
      aria-label={label}
      aria-valuemax={100}
      aria-valuemin={0}
      aria-valuenow={percent}
      className={cn(
        "h-2 overflow-hidden rounded-full border border-app-border bg-app-surface-muted",
        className,
      )}
      role="progressbar"
    >
      <div
        className="h-full rounded-full bg-app-accent transition-[width] duration-200"
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}

function clampUnit(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}
