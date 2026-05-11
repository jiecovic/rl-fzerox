// src/rl_fzerox/apps/run_manager/web/src/features/runs/charts_panel/model.ts

export {
  buildChartGroups,
  CHART_RANGE_OPTIONS,
  DEFAULT_CHART_RANGE_MODE,
  INITIAL_GROUP_OPEN,
  RUN_CHART_GROUPS,
  type RunChartGroupId,
} from "@/features/runs/charts_panel/model/catalog";
export {
  buildChartColorByRunId,
  chartSeriesColor,
} from "@/features/runs/charts_panel/model/colors";
export {
  formatChartValue,
  latestPointValue,
} from "@/features/runs/charts_panel/model/format";
export {
  buildLineageInfoById,
  buildLineageRunGroups,
  defaultSelectedRunIds,
  type LineageInfo,
  type LineageRunGroup,
  type LineageSelectionState,
  lineageSelectionState,
} from "@/features/runs/charts_panel/model/lineages";
export type {
  RunChartDescriptor,
  RunChartGroup,
} from "@/features/runs/charts_panel/model/types";
