// src/rl_fzerox/apps/run_manager/web/src/shared/ui/ScrollButtons.tsx
import { useEffect, useState } from "react";
import { ArrowDownIcon, ArrowUpIcon } from "@/shared/ui/icons";
import { TooltipIconButton } from "@/shared/ui/TooltipIconButton";

export function ScrollButtons() {
  const [position, setPosition] = useState(scrollPosition());

  useEffect(() => {
    const updatePosition = () => setPosition(scrollPosition());
    updatePosition();
    window.addEventListener("scroll", updatePosition, { passive: true });
    window.addEventListener("resize", updatePosition);
    const resizeObserver =
      typeof ResizeObserver !== "undefined" ? new ResizeObserver(updatePosition) : null;
    resizeObserver?.observe(document.documentElement);
    resizeObserver?.observe(document.body);
    return () => {
      window.removeEventListener("scroll", updatePosition);
      window.removeEventListener("resize", updatePosition);
      resizeObserver?.disconnect();
    };
  }, []);

  if (position.atTop && position.atBottom) {
    return null;
  }

  return (
    <div className="fixed right-[max(18px,calc((100vw-var(--app-shell-max-width))/2-54px))] bottom-7 z-10 grid gap-2">
      {!position.atTop ? (
        <TooltipIconButton
          aria-label="Scroll to top"
          className="shadow-[0_6px_18px_rgba(0,0,0,0.2)]"
          side="left"
          size="compact"
          tooltip="Scroll to top"
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
        >
          <ArrowUpIcon />
        </TooltipIconButton>
      ) : null}
      {!position.atBottom ? (
        <TooltipIconButton
          aria-label="Scroll to bottom"
          className="shadow-[0_6px_18px_rgba(0,0,0,0.2)]"
          side="left"
          size="compact"
          tooltip="Scroll to bottom"
          onClick={() =>
            window.scrollTo({
              top: document.documentElement.scrollHeight,
              behavior: "smooth",
            })
          }
        >
          <ArrowDownIcon />
        </TooltipIconButton>
      ) : null}
    </div>
  );
}

function scrollPosition() {
  if (typeof window === "undefined") {
    return { atBottom: true, atTop: true };
  }
  const scrollTop = window.scrollY;
  const maxScrollTop = document.documentElement.scrollHeight - window.innerHeight;
  return {
    atBottom: scrollTop >= maxScrollTop - 4,
    atTop: scrollTop <= 4,
  };
}
