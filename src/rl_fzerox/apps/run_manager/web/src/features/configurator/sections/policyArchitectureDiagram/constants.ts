export const diagramMetrics = {
  canvasMinHeight: 230,
  canvasMinWidth: 1456,
  graphMargin: 18,
  headBranchYOffset: 14,
  inputBranchYOffset: 10,
  junctionSize: 20,
  node: {
    characterWidth: 6.9,
    detailLineHeight: 15,
    maxWidth: 204,
    minHeight: 72,
    minWidth: 124,
    paddingX: 24,
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
