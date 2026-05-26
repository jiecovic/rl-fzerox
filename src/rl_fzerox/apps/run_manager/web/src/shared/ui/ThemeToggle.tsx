// src/rl_fzerox/apps/run_manager/web/src/shared/ui/ThemeToggle.tsx

import { IconButton } from "@/shared/ui/Button";
import { MoonIcon, SunIcon } from "@/shared/ui/icons";

export type Theme = "light" | "dark";

interface ThemeToggleProps {
  theme: Theme;
  onToggle: () => void;
}

export function ThemeToggle({ theme, onToggle }: ThemeToggleProps) {
  return (
    <IconButton
      aria-label={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
      size="theme"
      onClick={onToggle}
    >
      <ThemeIcon theme={theme} />
    </IconButton>
  );
}

function ThemeIcon({ theme }: { theme: Theme }) {
  return theme === "light" ? <MoonIcon /> : <SunIcon />;
}
