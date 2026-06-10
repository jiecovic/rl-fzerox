// web/run-manager/src/shared/ui/ThemeToggle.tsx

import { MoonIcon, SunIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

export type Theme = "light" | "dark";

interface ThemeToggleProps {
  theme: Theme;
  onToggle: () => void;
}

export function ThemeToggle({ theme, onToggle }: ThemeToggleProps) {
  const label = theme === "light" ? "Switch to dark mode" : "Switch to light mode";
  return (
    <TooltipIconButton aria-label={label} size="theme" tooltip={label} onClick={onToggle}>
      <ThemeIcon theme={theme} />
    </TooltipIconButton>
  );
}

function ThemeIcon({ theme }: { theme: Theme }) {
  return theme === "light" ? <MoonIcon /> : <SunIcon />;
}
