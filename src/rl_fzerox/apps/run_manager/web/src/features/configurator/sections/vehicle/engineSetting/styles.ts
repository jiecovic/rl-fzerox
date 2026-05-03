import type { CSSProperties } from "react";

export function thumbStyle(ratio: number): CSSProperties {
  return { "--vehicle-slider-ratio": `${ratio}` } as CSSProperties;
}

export function tickStyle(ratio: number): CSSProperties {
  return { "--vehicle-slider-ratio": `${ratio}` } as CSSProperties;
}

export function singleFillStyle(ratio: number): CSSProperties {
  return {
    "--vehicle-slider-fill-start": "0",
    "--vehicle-slider-fill-end": `${ratio}`,
  } as CSSProperties;
}

export function rangeFillStyle(minRatio: number, maxRatio: number): CSSProperties {
  return {
    "--vehicle-slider-fill-start": `${minRatio}`,
    "--vehicle-slider-fill-end": `${maxRatio}`,
  } as CSSProperties;
}
