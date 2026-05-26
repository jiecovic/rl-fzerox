// src/rl_fzerox/apps/run_manager/web/src/shared/ui/Button.tsx
import type { ButtonHTMLAttributes } from "react";

import { cn } from "@/shared/ui/cn";

type ButtonVariant = "accentSoft" | "primary" | "secondary";
type ButtonTone = "danger" | "default";
type IconButtonSize = "compact" | "default" | "small" | "theme";
type IconButtonTone = "danger" | "default" | "muted";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  tone?: ButtonTone;
  variant?: ButtonVariant;
}

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  "aria-label": string;
  size?: IconButtonSize;
  tone?: IconButtonTone;
}

const baseButtonClass =
  "inline-flex h-10 items-center justify-center rounded-lg border px-4 text-sm font-semibold leading-none whitespace-nowrap transition-colors disabled:cursor-not-allowed disabled:opacity-[0.56]";

const buttonVariantClasses: Record<ButtonVariant, string> = {
  accentSoft:
    "border-app-accent bg-[color-mix(in_srgb,var(--accent)_14%,var(--surface))] text-app-text hover:border-app-accent hover:bg-[color-mix(in_srgb,var(--accent)_22%,var(--surface))]",
  primary: "border-app-accent bg-app-accent text-app-accent-text",
  secondary:
    "border-app-border bg-app-surface text-app-text hover:border-app-border-strong hover:bg-app-surface-muted",
};

const iconButtonBaseClass =
  "grid place-items-center border border-app-border bg-app-surface p-0 transition-colors hover:border-app-border-strong hover:bg-app-surface-muted disabled:cursor-not-allowed disabled:opacity-[0.56]";

const iconButtonSizeClasses: Record<IconButtonSize, string> = {
  compact: "h-[30px] w-[30px] rounded-md",
  default: "h-10 w-10 rounded-lg",
  small: "h-7 w-7 rounded-md",
  theme: "h-10 w-11 rounded-lg",
};

const iconButtonToneClasses: Record<IconButtonTone, string> = {
  danger: "text-app-danger",
  default: "text-app-text",
  muted: "text-app-muted hover:text-app-text",
};

export function Button({
  className,
  tone = "default",
  type = "button",
  variant = "secondary",
  ...props
}: ButtonProps) {
  return (
    <button className={buttonClassName({ className, tone, variant })} type={type} {...props} />
  );
}

export function IconButton({
  className,
  size = "default",
  tone = "default",
  type = "button",
  ...props
}: IconButtonProps) {
  return (
    <button
      className={cn(
        iconButtonBaseClass,
        iconButtonSizeClasses[size],
        iconButtonToneClasses[tone],
        className,
      )}
      type={type}
      {...props}
    />
  );
}

export function buttonClassName({
  className,
  tone = "default",
  variant = "secondary",
}: {
  className?: string;
  tone?: ButtonTone;
  variant?: ButtonVariant;
}) {
  return cn(
    baseButtonClass,
    buttonVariantClasses[variant],
    tone === "danger" ? "text-app-danger" : undefined,
    className,
  );
}
