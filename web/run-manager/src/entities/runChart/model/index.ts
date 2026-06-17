// web/run-manager/src/entities/runChart/model/index.ts

export {
  buildChartGroups,
  CHART_RANGE_OPTIONS,
  DEFAULT_CHART_RANGE_MODE,
  INITIAL_GROUP_OPEN,
  RUN_CHART_GROUPS,
  type RunChartGroupId,
} from "@/entities/runChart/model/catalog";
export {
  buildChartColorByRunId,
  type ChartColorMode,
  chartSeriesColor,
} from "@/entities/runChart/model/colors";
export {
  formatChartValue,
  latestPointValue,
} from "@/entities/runChart/model/format";
export {
  type BranchRunGroup,
  buildBranchRunGroups,
  buildLineageInfoById,
  buildLineageRunGroups,
  defaultSelectedRunIds,
  type LineageInfo,
  type LineageRunGroup,
  type LineageSelectionState,
  lineageSelectionState,
} from "@/entities/runChart/model/lineages";
export type {
  RunChartDescriptor,
  RunChartGroup,
} from "@/entities/runChart/model/types";
