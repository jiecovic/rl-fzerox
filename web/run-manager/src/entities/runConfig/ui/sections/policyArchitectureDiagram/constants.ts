// web/run-manager/src/entities/runConfig/ui/sections/policyArchitectureDiagram/constants.ts
export const diagramMetrics = {
  canvasMinHeight: 230,
  canvasMinWidth: 1456,
  contentRightPadding: 84,
  graphMargin: 18,
  summaryBox: {
    height: 44,
    paddingX: 12,
    reservedHeight: 92,
    width: 156,
    xInset: 12,
    yInset: 10,
  },
  headBranchYOffset: 14,
  inputImageBranchYOffset: -24,
  inputStateBranchYOffset: 10,
  junctionSize: 20,
  node: {
    characterWidth: 6.9,
    detailLineHeight: 15,
    maxWidth: 196,
    minHeight: 84,
    minWidth: 124,
    paramLineHeight: 14,
    paddingX: 24,
    paramTopGap: 8,
    titleAndPaddingHeight: 44,
  },
} as const;

export const elkLayoutOptions = {
  "elk.algorithm": "layered",
  "elk.direction": "RIGHT",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
  "elk.layered.spacing.nodeNodeBetweenLayers": "58",
  "elk.spacing.nodeNode": "28",
} as const;
