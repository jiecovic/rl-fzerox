// src/rl_fzerox/apps/run_manager/web/src/features/runs/chartsPanel/model/index.ts

export {
  buildChartGroups,
  CHART_RANGE_OPTIONS,
  DEFAULT_CHART_RANGE_MODE,
  INITIAL_GROUP_OPEN,
  RUN_CHART_GROUPS,
  type RunChartGroupId,
} from "@/features/runs/chartsPanel/model/catalog";
export {
  buildChartColorByRunId,
  chartSeriesColor,
} from "@/features/runs/chartsPanel/model/colors";
export {
  formatChartValue,
  latestPointValue,
} from "@/features/runs/chartsPanel/model/format";
export {
  buildLineageInfoById,
  buildLineageRunGroups,
  defaultSelectedRunIds,
  type LineageInfo,
  type LineageRunGroup,
  type LineageSelectionState,
  lineageSelectionState,
} from "@/features/runs/chartsPanel/model/lineages";
export type {
  RunChartDescriptor,
  RunChartGroup,
} from "@/features/runs/chartsPanel/model/types";
