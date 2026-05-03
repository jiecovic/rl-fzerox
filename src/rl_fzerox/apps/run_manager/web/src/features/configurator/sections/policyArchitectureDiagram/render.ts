import type { ElkEdgeSection, ElkExtendedEdge, ElkNode } from "elkjs/lib/elk.bundled.js";

import { diagramMetrics } from "./constants";
import type { NodeVisual, RenderedEdge, RenderedGraph, RenderedNode } from "./types";

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
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const edges = (elkGraph.edges ?? []).flatMap((edge) => edgePaths(edge, nodeById));
  const width = Math.max(
    diagramMetrics.canvasMinWidth,
    numberOrZero(elkGraph.width),
    maxNodeExtent(nodes, "x"),
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
): RenderedEdge[] {
  return (edge.sections ?? []).map((section) => ({
    hasArrow: hasEdgeArrow(edge.id),
    id: `${edge.id}-${section.id}`,
    path: pathForSection(edge.id, section, nodeById),
  }));
}

function hasEdgeArrow(edgeId: string) {
  return !(edgeId.endsWith("-concat:image") || edgeId.endsWith("-concat:state"));
}

function pathForSection(
  edgeId: string,
  section: ElkEdgeSection,
  nodeById: ReadonlyMap<string, RenderedNode>,
) {
  if (edgeId.endsWith("-concat:image") || edgeId.endsWith("-concat:state")) {
    return concatBranchPath(edgeId, section, nodeById);
  }
  if (edgeId.endsWith("-policy_head") || edgeId.endsWith("-value_head")) {
    return headBranchPath(edgeId, section, nodeById);
  }
  if (edgeId.endsWith("-action_net") || edgeId.endsWith("-value_net")) {
    return terminalHeadPath(edgeId, section, nodeById);
  }

  return pathForRawSection(section);
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
    return pathForRawSection(section);
  }

  const start = {
    x: source.x + source.width,
    y: source.y + source.height / 2,
  };
  const end = {
    x: target.x,
    y: target.y + target.height / 2,
  };
  return [
    `M ${round(start.x)} ${round(start.y)}`,
    `H ${round((start.x + end.x) / 2)}`,
    `V ${round(end.y)}`,
    `H ${round(end.x)}`,
  ].join(" ");
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
    return pathForRawSection(section);
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
        x: concat.x,
        y: concat.y + concat.height / 2,
      };
  const mergeX = edgeId.endsWith("-concat:image")
    ? Math.max(start.x + 22, end.x)
    : Math.max(start.x + 24, end.x - 24);
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
    return pathForRawSection(section);
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
  return [
    `M ${round(start.x)} ${round(start.y)}`,
    `H ${round(midX)}`,
    `V ${round(end.y)}`,
    `H ${round(end.x)}`,
  ].join(" ");
}

function splitEdgeId(edgeId: string): [string | undefined, string | undefined] {
  const separatorIndex = edgeId.lastIndexOf("-");
  if (separatorIndex < 0) {
    return [undefined, undefined];
  }
  return [edgeId.slice(0, separatorIndex), edgeId.slice(separatorIndex + 1)];
}

function pathForRawSection(section: ElkEdgeSection) {
  const points = [section.startPoint, ...(section.bendPoints ?? []), section.endPoint];
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${round(point.x)} ${round(point.y)}`)
    .join(" ");
}

function maxNodeExtent(nodes: readonly RenderedNode[], axis: "x" | "y") {
  return Math.max(
    0,
    ...nodes.map((node) => (axis === "x" ? node.x + node.width : node.y + node.height)),
  );
}

function nodeYOffset(nodeId: string) {
  if (["image", "cnn", "image_projection", "state", "state_mlp"].includes(nodeId)) {
    return diagramMetrics.inputBranchYOffset;
  }
  return ["policy_head", "value_head", "action_net", "value_net"].includes(nodeId)
    ? diagramMetrics.headBranchYOffset
    : 0;
}

function numberOrZero(value: number | undefined) {
  return value ?? 0;
}

function round(value: number) {
  return Math.round(value * 10) / 10;
}
