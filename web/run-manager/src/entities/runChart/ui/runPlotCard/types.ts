// web/run-manager/src/entities/runChart/ui/runPlotCard/types.ts
import type UPlot from "uplot";

export type RunPlotPoint = { step: number; value: number };

export type RunPlotSeries = {
  color: string;
  latest: number | null;
  name: string;
  points: readonly RunPlotPoint[];
  runId: string;
};

export interface RunPlotCardProps {
  emptyText: string;
  formatValue: (value: number | null) => string;
  series: readonly RunPlotSeries[];
  title: string;
}

export type PlotData = [number[], ...(number | null)[][]];
export type TooltipEntry = { color: string; name: string; value: number };
export type TooltipState = {
  left: number;
  step: number;
  top: number;
  values: readonly TooltipEntry[];
};
export type XRange = { max: number; min: number };
export type UPlotConstructor = new (
  opts: UPlot.Options,
  data?: UPlot.AlignedData,
  targ?: HTMLElement | ((self: UPlot, init: () => void) => void),
) => UPlot;
