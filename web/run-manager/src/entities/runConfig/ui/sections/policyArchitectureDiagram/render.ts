// web/run-manager/src/entities/runConfig/ui/sections/policyArchitectureDiagram/render.ts
import type { ElkEdgeSection, ElkExtendedEdge, ElkNode } from "elkjs/lib/elk.bundled.js";

import { diagramMetrics } from "@/entities/runConfig/ui/sections/policyArchitectureDiagram/constants";
import type {
  NodeVisual,
  RenderedEdge,
  RenderedGraph,
  RenderedNode,
} from "@/entities/runConfig/ui/sections/policyArchitectureDiagram/types";

export function toRenderedGraph(
  elkGraph: ElkNode,
  visuals: Map<string, NodeVisual>,
): RenderedGraph {
  const nodes = (elkGraph.children ?? []).flatMap((node) => {
    const visual = visuals.get(node.id);
    if (visual === undefined) {
      return [];
    }
    return [
      {
        height: numberOrZero(node.height),
        id: node.id,
        visual,
        width: numberOrZero(node.width),
        x: numberOrZero(node.x),
        y: numberOrZero(node.y) + nodeYOffset(node.id),
      },
    ];
  });
  shiftNodesY(nodes, diagramMetrics.summaryBox.reservedHeight);
  alignInputFusionJunction(nodes);
  alignPostConcatPipeline(nodes);
  alignOutputHeadCluster(nodes);
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const rawPathYOffset = diagramMetrics.summaryBox.reservedHeight;
  const edges = (elkGraph.edges ?? []).flatMap((edge) => edgePaths(edge, nodeById, rawPathYOffset));
  const width = Math.max(
    diagramMetrics.canvasMinWidth,
    numberOrZero(elkGraph.width),
    maxNodeExtent(nodes, "x") + diagramMetrics.contentRightPadding,
  );
  const height = Math.max(
    diagramMetrics.canvasMinHeight,
    numberOrZero(elkGraph.height),
    maxNodeExtent(nodes, "y"),
  );
  const margin = diagramMetrics.graphMargin;

  return {
    edges,
    height,
    nodes,
    viewBox: `${-margin} ${-margin} ${width + margin * 2} ${height + margin * 2}`,
    width,
  };
}

function shiftNodesY(nodes: readonly RenderedNode[], offset: number) {
  for (const node of nodes) {
    node.y += offset;
  }
}

function alignInputFusionJunction(nodes: RenderedNode[]) {
  const junction = nodes.find((node) => node.id === "concat");
  const imageSource = firstExistingNode(nodes, ["image_projection", "cnn", "image"]);
  const stateSource = firstExistingNode(nodes, ["state_mlp", "state"]);
  if (junction === undefined || imageSource === undefined || stateSource === undefined) {
    return;
  }

  const centerY = (nodeCenterY(imageSource) + nodeCenterY(stateSource)) / 2;
  junction.y = centerY - junction.height / 2;
}

function alignPostConcatPipeline(nodes: readonly RenderedNode[]) {
  const junction = nodes.find((node) => node.id === "concat");
  if (junction === undefined) {
    return;
  }

  const centerY = nodeCenterY(junction);
  for (const id of ["fusion", "layer_norm", "lstm"]) {
    alignNodeCenterY(nodes, id, centerY);
  }
}

function alignOutputHeadCluster(nodes: RenderedNode[]) {
  const junction = nodes.find((node) => node.id === "heads");
  const auxiliaryHead = nodes.find((node) => node.id === "aux_head");
  const source = outputHeadSource(nodes);
  if (junction === undefined || auxiliaryHead === undefined || source === undefined) {
    return;
  }

  const centerY = source.y + source.height / 2;
  auxiliaryHead.y = centerY - auxiliaryHead.height / 2;
  junction.y = centerY - junction.height / 2;

  const verticalGap = 28;
  const policyHead = nodes.find((node) => node.id === "policy_head");
  if (policyHead !== undefined) {
    policyHead.y = auxiliaryHead.y - verticalGap - policyHead.height;
    alignNodeCenterY(nodes, "action_net", policyHead.y + policyHead.height / 2);
  }

  const valueHead = nodes.find((node) => node.id === "value_head");
  if (valueHead !== undefined) {
    valueHead.y = auxiliaryHead.y + auxiliaryHead.height + verticalGap;
    alignNodeCenterY(nodes, "value_net", valueHead.y + valueHead.height / 2);
  }
}

function outputHeadSource(nodes: readonly RenderedNode[]) {
  return firstExistingNode(nodes, ["lstm", "layer_norm", "fusion", "concat"]);
}

function alignNodeCenterY(nodes: readonly RenderedNode[], nodeId: string, centerY: number) {
  const node = nodes.find((candidate) => candidate.id === nodeId);
  if (node !== undefined) {
    node.y = centerY - node.height / 2;
  }
}

function firstExistingNode(nodes: readonly RenderedNode[], ids: readonly string[]) {
  for (const id of ids) {
    const node = nodes.find((candidate) => candidate.id === id);
    if (node !== undefined) {
      return node;
    }
  }
  return undefined;
}

function nodeCenterY(node: RenderedNode) {
  return node.y + node.height / 2;
}

export function fallbackGraph(visuals: Map<string, NodeVisual>): RenderedGraph {
  const nodes = Array.from(visuals.entries()).map(([id, visual], index) => ({
    height: visual.height,
    id,
    visual,
    width: visual.width,
    x: index * (visual.width + 40),
    y: 20,
  }));
  return {
    edges: [],
    height: maxNodeExtent(nodes, "y"),
    nodes,
    viewBox: `0 0 ${maxNodeExtent(nodes, "x")} ${maxNodeExtent(nodes, "y")}`,
    width: maxNodeExtent(nodes, "x"),
  };
}

function edgePaths(
  edge: ElkExtendedEdge,
  nodeById: ReadonlyMap<string, RenderedNode>,
  rawPathYOffset: number,
): RenderedEdge[] {
  return (edge.sections ?? []).map((section) => ({
    hasArrow: hasEdgeArrow(edge.id),
    id: `${edge.id}-${section.id}`,
    path: pathForSection(edge.id, section, nodeById, rawPathYOffset),
  }));
}

function hasEdgeArrow(edgeId: string) {
  return !(
    edgeId.endsWith("-concat:image") ||
    edgeId.endsWith("-concat:state") ||
    edgeId.endsWith("-heads")
  );
}

function pathForSection(
  edgeId: string,
  section: ElkEdgeSection,
  nodeById: ReadonlyMap<string, RenderedNode>,
  rawPathYOffset: number,
) {
  if (edgeId.endsWith("-concat:image") || edgeId.endsWith("-concat:state")) {
    return concatBranchPath(edgeId, section, nodeById);
  }
  if (edgeId.startsWith("concat:out-")) {
    return concatOutPath(edgeId, section, nodeById);
  }
  if (edgeId.endsWith("-heads")) {
    return trunkToHeadJunctionPath(edgeId, section, nodeById);
  }
  if (
    edgeId.endsWith("-policy_head") ||
    edgeId.endsWith("-aux_head") ||
    edgeId.endsWith("-value_head")
  ) {
    return headBranchPath(edgeId, section, nodeById);
  }
  if (edgeId.endsWith("-action_net") || edgeId.endsWith("-value_net")) {
    return terminalHeadPath(edgeId, section, nodeById);
  }

  return pathForRawSection(section, rawPathYOffset);
}

function concatOutPath(
  edgeId: string,
  section: ElkEdgeSection,
  nodeById: ReadonlyMap<string, RenderedNode>,
) {
  const [, targetId] = splitEdgeId(edgeId);
  const source = nodeById.get("concat");
  const target = targetId === undefined ? undefined : nodeById.get(targetId);
  if (source === undefined || target === undefined) {
    return pathForRawSection(section, diagramMetrics.summaryBox.reservedHeight);
  }

  const start = {
    x: source.x + source.width,
    y: source.y + source.height / 2,
  };
  const end = {
    x: target.x,
    y: target.y + target.height / 2,
  };
  return orthogonalPath(start, end);
}

function trunkToHeadJunctionPath(
  edgeId: string,
  section: ElkEdgeSection,
  nodeById: ReadonlyMap<string, RenderedNode>,
) {
  const [sourceId, targetId] = splitEdgeId(edgeId);
  const source =
    sourceId === "concat:out" ? nodeById.get("concat") : endpointNode(sourceId, nodeById);
  const junction = targetId === undefined ? undefined : nodeById.get(targetId);
  if (source === undefined || junction === undefined) {
    return pathForRawSection(section, diagramMetrics.summaryBox.reservedHeight);
  }

  const start = {
    x: source.x + source.width,
    y: source.y + source.height / 2,
  };
  const end = {
    x: junction.x + junction.width / 2,
    y: junction.y + junction.height / 2,
  };
  const segments = [`M ${round(start.x)} ${round(start.y)}`, `H ${round(end.x)}`];
  if (Math.abs(start.y - end.y) >= 0.5) {
    segments.push(`V ${round(end.y)}`);
  }
  return segments.join(" ");
}

function terminalHeadPath(
  edgeId: string,
  section: ElkEdgeSection,
  nodeById: ReadonlyMap<string, RenderedNode>,
) {
  const [sourceId, targetId] = splitEdgeId(edgeId);
  const source = sourceId === undefined ? undefined : nodeById.get(sourceId);
  const target = targetId === undefined ? undefined : nodeById.get(targetId);
  if (source === undefined || target === undefined) {
    return pathForRawSection(section, diagramMetrics.summaryBox.reservedHeight);
  }

  const start = {
    x: source.x + source.width,
    y: source.y + source.height / 2,
  };
  const end = {
    x: target.x,
    y: target.y + target.height / 2,
  };
  return orthogonalPath(start, end);
}

function concatBranchPath(
  edgeId: string,
  section: ElkEdgeSection,
  nodeById: ReadonlyMap<string, RenderedNode>,
) {
  const [sourceId] = splitEdgeId(edgeId);
  const source = sourceId === undefined ? undefined : nodeById.get(sourceId);
  const concat = nodeById.get("concat");
  if (source === undefined || concat === undefined) {
    return pathForRawSection(section, diagramMetrics.summaryBox.reservedHeight);
  }

  const start = {
    x: source.x + source.width,
    y: source.y + source.height / 2,
  };
  const end = edgeId.endsWith("-concat:image")
    ? {
        x: concat.x + concat.width / 2,
        y: concat.y,
      }
    : {
        x: concat.x + concat.width / 2,
        y: concat.y + concat.height,
      };
  const mergeX = Math.max(start.x + 22, end.x);
  return [
    `M ${round(start.x)} ${round(start.y)}`,
    `H ${round(mergeX)}`,
    `V ${round(end.y)}`,
    `H ${round(end.x)}`,
  ].join(" ");
}

function headBranchPath(
  edgeId: string,
  section: ElkEdgeSection,
  nodeById: ReadonlyMap<string, RenderedNode>,
) {
  const [sourceId, targetId] = splitEdgeId(edgeId);
  const source = sourceId === undefined ? undefined : nodeById.get(sourceId);
  const target = targetId === undefined ? undefined : nodeById.get(targetId);
  if (source === undefined || target === undefined) {
    return pathForRawSection(section, diagramMetrics.summaryBox.reservedHeight);
  }

  if (source.visual.kind === "junction") {
    const start = {
      x: source.x + source.width / 2,
      y: source.y + source.height / 2,
    };
    const end = {
      x: target.x,
      y: target.y + target.height / 2,
    };
    if (targetId === "aux_head" && Math.abs(start.y - end.y) < 0.5) {
      return [`M ${round(start.x)} ${round(start.y)}`, `H ${round(end.x)}`].join(" ");
    }
    return [`M ${round(start.x)} ${round(start.y)}`, `V ${round(end.y)}`, `H ${round(end.x)}`].join(
      " ",
    );
  }

  const start = {
    x: source.x + source.width,
    y: source.y + source.height / 2,
  };
  const end = {
    x: target.x,
    y: target.y + target.height / 2,
  };
  const midX = Math.max(start.x + 16, (start.x + end.x) / 2 - 4);
  return orthogonalPath(start, end, midX);
}

function orthogonalPath(
  start: { x: number; y: number },
  end: { x: number; y: number },
  midX = (start.x + end.x) / 2,
) {
  const segments = [`M ${round(start.x)} ${round(start.y)}`];
  if (Math.abs(start.y - end.y) < 0.5) {
    segments.push(`H ${round(end.x)}`);
    return segments.join(" ");
  }
  segments.push(`H ${round(midX)}`, `V ${round(end.y)}`, `H ${round(end.x)}`);
  return segments.join(" ");
}

function endpointNode(nodeId: string | undefined, nodeById: ReadonlyMap<string, RenderedNode>) {
  return nodeId === undefined ? undefined : nodeById.get(nodeId);
}

function splitEdgeId(edgeId: string): [string | undefined, string | undefined] {
  const separatorIndex = edgeId.lastIndexOf("-");
  if (separatorIndex < 0) {
    return [undefined, undefined];
  }
  return [edgeId.slice(0, separatorIndex), edgeId.slice(separatorIndex + 1)];
}

function pathForRawSection(section: ElkEdgeSection, yOffset = 0) {
  const points = [section.startPoint, ...(section.bendPoints ?? []), section.endPoint];
  return points
    .map(
      (point, index) => `${index === 0 ? "M" : "L"} ${round(point.x)} ${round(point.y + yOffset)}`,
    )
    .join(" ");
}

function maxNodeExtent(nodes: readonly RenderedNode[], axis: "x" | "y") {
  return Math.max(
    0,
    ...nodes.map((node) => (axis === "x" ? node.x + node.width : node.y + node.height)),
  );
}

function nodeYOffset(nodeId: string) {
  if (["image", "cnn", "image_projection"].includes(nodeId)) {
    return diagramMetrics.inputImageBranchYOffset;
  }
  if (["state", "state_mlp"].includes(nodeId)) {
    return diagramMetrics.inputStateBranchYOffset;
  }
  return ["heads", "policy_head", "aux_head", "value_head", "action_net", "value_net"].includes(
    nodeId,
  )
    ? diagramMetrics.headBranchYOffset
    : 0;
}

function numberOrZero(value: number | undefined) {
  return value ?? 0;
}

function round(value: number) {
  return Math.round(value * 10) / 10;
}
