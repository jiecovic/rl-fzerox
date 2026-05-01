import { useEffect, useState } from "react";

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
    <div className="scroll-buttons">
      {!position.atTop ? (
        <button
          aria-label="Scroll to top"
          className="icon-button compact-icon-button tooltip-anchor"
          data-tooltip="Scroll to top"
          data-tooltip-position="left"
          type="button"
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
        >
          <ArrowUpIcon />
        </button>
      ) : null}
      {!position.atBottom ? (
        <button
          aria-label="Scroll to bottom"
          className="icon-button compact-icon-button tooltip-anchor"
          data-tooltip="Scroll to bottom"
          data-tooltip-position="left"
          type="button"
          onClick={() =>
            window.scrollTo({
              top: document.documentElement.scrollHeight,
              behavior: "smooth",
            })
          }
        >
          <ArrowDownIcon />
        </button>
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

function ArrowUpIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="16" viewBox="0 0 20 20" width="16">
      <path
        d="M10 16V4M5 9l5-5 5 5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function ArrowDownIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="16" viewBox="0 0 20 20" width="16">
      <path
        d="M10 4v12M5 11l5 5 5-5"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}
