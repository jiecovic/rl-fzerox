// src/rl_fzerox/apps/run_manager/web/src/shared/ui/Field.tsx
import * as ToggleGroup from "@radix-ui/react-toggle-group";
import type {
  ButtonHTMLAttributes,
  HTMLAttributes,
  InputHTMLAttributes,
  ReactNode,
  SelectHTMLAttributes,
} from "react";

import { cn } from "@/shared/ui/cn";

interface SwitchButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  active: boolean;
  activeLabel?: string;
  inactiveLabel?: string;
  label: string;
}

interface SegmentedChoiceButtonProps
  extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "type" | "value"> {
  active: boolean;
  disabledChoice?: boolean;
  value: string;
}

const fieldInputClass =
  "box-border h-10 w-full rounded-lg border border-app-border bg-app-surface px-2.5 text-app-text selection:bg-app-selection-background selection:text-app-selection-text disabled:opacity-[0.42] focus:border-app-border-strong focus:bg-app-field-active focus:caret-app-text focus:outline-none focus:shadow-[inset_0_0_0_1px_var(--focus-ring),0_1px_0_rgba(255,255,255,0.04)]";

export function FieldShell({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("grid min-w-0 gap-1.5 text-sm text-app-muted", className)} {...props} />
  );
}

export function FieldLabelRow({ className, ...props }: HTMLAttributes<HTMLSpanElement>) {
  return <span className={cn("flex items-center gap-1.5", className)} {...props} />;
}

export function FieldNote({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("text-xs tabular-nums text-app-muted", className)} {...props} />;
}

export function FieldInput({ className, type, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        fieldInputClass,
        type === "number"
          ? "text-center indent-[-8px] [&::-webkit-inner-spin-button]:ml-2"
          : undefined,
        className,
      )}
      type={type}
      {...props}
    />
  );
}

export function FieldSelect({ className, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={cn(fieldInputClass, className)} {...props} />;
}

export function SwitchButton({
  active,
  activeLabel = "On",
  className,
  inactiveLabel = "Off",
  label,
  type = "button",
  ...props
}: SwitchButtonProps) {
  return (
    <button
      aria-label={label}
      aria-pressed={active}
      className={cn(
        "relative inline-flex h-6 w-[46px] items-center rounded-full border p-0 text-transparent focus-visible:outline-none focus-visible:shadow-none",
        active ? "border-app-accent bg-app-accent" : "border-app-border bg-app-surface",
        className,
      )}
      type={type}
      {...props}
    >
      <span
        aria-hidden="true"
        className={cn(
          "absolute left-[3px] h-[18px] w-[18px] rounded-full border border-app-border-strong bg-app-surface-muted transition-[left] duration-150 ease-in-out",
          active ? "left-[23px] bg-app-surface" : undefined,
        )}
      />
      <strong className="hidden">{active ? activeLabel : inactiveLabel}</strong>
    </button>
  );
}

export function RangeRow({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("grid grid-cols-[minmax(0,1fr)_104px] items-start gap-3", className)}
      {...props}
    />
  );
}

export function RangeReadonly({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "grid min-h-[34px] place-items-center border border-app-border bg-app-surface text-center text-[13px] tabular-nums text-app-text",
        className,
      )}
      {...props}
    />
  );
}

export function optionalNumberRowClass(disabled: boolean) {
  return cn(
    "grid grid-cols-[52px_minmax(0,1fr)_96px] items-center gap-3",
    disabled ? "[&_.slider-control]:pointer-events-none [&_input]:pointer-events-none" : undefined,
  );
}

interface SegmentedChoiceGroupProps {
  children: ReactNode;
  className?: string;
  label: string;
  onValueChange: (value: string) => void;
  value: string;
}

export function SegmentedChoiceGroup({
  children,
  className,
  label,
  onValueChange,
  value,
}: SegmentedChoiceGroupProps) {
  return (
    <ToggleGroup.Root
      aria-label={label}
      className={cn(
        "inline-flex w-fit max-w-full min-w-0 flex-wrap justify-self-start overflow-hidden rounded-lg border border-app-border bg-app-surface-muted p-0.5",
        className,
      )}
      type="single"
      value={value}
      onValueChange={(nextValue) => {
        if (nextValue !== "") {
          onValueChange(nextValue);
        }
      }}
    >
      {children}
    </ToggleGroup.Root>
  );
}

export function SegmentedChoiceButton({
  active,
  className,
  disabledChoice = false,
  ...props
}: SegmentedChoiceButtonProps) {
  return (
    <ToggleGroup.Item
      className={cn(
        "min-h-8 min-w-0 rounded-md border border-transparent px-3.5 text-sm font-medium whitespace-nowrap text-app-muted transition-colors",
        active ? "border-app-border-strong bg-app-surface text-app-text" : undefined,
        disabledChoice ? "cursor-not-allowed opacity-55" : undefined,
        !active && !disabledChoice
          ? "hover:bg-[color-mix(in_srgb,var(--accent)_8%,var(--surface-muted))] hover:text-app-text"
          : undefined,
        className,
      )}
      data-disabled={disabledChoice ? "" : undefined}
      disabled={disabledChoice}
      {...props}
    />
  );
}
