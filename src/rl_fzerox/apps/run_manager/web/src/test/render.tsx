// src/rl_fzerox/apps/run_manager/web/src/test/render.tsx
import { type RenderOptions, render as renderWithTestingLibrary } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";

import { AppTooltipProvider } from "@/shared/ui/Tooltip";

export * from "@testing-library/react";

export function render(ui: ReactElement, options?: Omit<RenderOptions, "wrapper">) {
  return renderWithTestingLibrary(ui, {
    wrapper: TestProviders,
    ...options,
  });
}

function TestProviders({ children }: { children: ReactNode }) {
  return <AppTooltipProvider>{children}</AppTooltipProvider>;
}
