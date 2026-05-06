// src/rl_fzerox/apps/run_manager/web/src/shared/ui/ThemeToggle.tsx
import { MoonIcon, SunIcon } from "@/shared/ui/icons";

export type Theme = "light" | "dark";

interface ThemeToggleProps {
  theme: Theme;
  onToggle: () => void;
}

export function ThemeToggle({ theme, onToggle }: ThemeToggleProps) {
  return (
    <button
      aria-label={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
      className="theme-button"
      type="button"
      onClick={onToggle}
    >
      <ThemeIcon theme={theme} />
    </button>
  );
}

function ThemeIcon({ theme }: { theme: Theme }) {
  return theme === "light" ? <MoonIcon /> : <SunIcon />;
}
