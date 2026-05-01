import type { ElkEdgeSection, ElkExtendedEdge, ElkNode } from "elkjs/lib/elk.bundled.js";
import { useEffect, useState } from "react";
import type { PolicyArchitecturePreview } from "@/shared/api/contract";

const diagramMetrics = {
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

const elkLayoutOptions = {
  "elk.algorithm": "layered",
  "elk.direction": "RIGHT",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
  "elk.layered.spacing.nodeNodeBetweenLayers": "58",
  "elk.spacing.nodeNode": "28",
} as const;

export function PolicyArchitectureDiagram({ preview }: { preview: PolicyArchitecturePreview }) {
  const [layout, setLayout] = useState<RenderedGraph | null>(null);

  useEffect(() => {
    let cancelled = false;
    const graph = buildArchitectureGraph(preview);
    setLayout(null);

    layoutGraph(graph.elkGraph)
      .then((elkGraph) => {
        if (!cancelled) {
          setLayout(toRenderedGraph(elkGraph, graph.visuals));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setLayout(fallbackGraph(graph.visuals));
        }
      });

    return () => {
      cancelled = true;
    };
  }, [preview]);

  if (layout === null) {
    return <div className="architecture-layout-pending">Laying out architecture...</div>;
  }

  return (
    <svg
      aria-label="Policy architecture diagram"
      className="architecture-graph-svg"
      preserveAspectRatio="xMinYMid meet"
      role="img"
      viewBox={layout.viewBox}
    >
      <defs>
        <marker
          id="policy-architecture-arrow"
          markerHeight="8"
          markerWidth="8"
          orient="auto"
          refX="8"
          refY="4"
          viewBox="0 0 8 8"
        >
          <path className="architecture-arrow-head" d="M0 0 8 4 0 8z" />
        </marker>
      </defs>

      <g className="architecture-edges">
        {layout.edges.map((edge) => (
          <path
            className="architecture-edge"
            d={edge.path}
            fill="none"
            key={edge.id}
            markerEnd={edge.hasArrow ? "url(#policy-architecture-arrow)" : undefined}
          />
        ))}
      </g>

      <g className="architecture-nodes">
        {layout.nodes.map((node) =>
          node.visual.kind === "junction" ? (
            <circle
              className="architecture-junction"
              cx={node.x + node.width / 2}
              cy={node.y + node.height / 2}
              key={node.id}
              r={node.width / 2}
            />
          ) : (
            <ArchitectureSvgNode key={node.id} node={node} />
          ),
        )}
      </g>
    </svg>
  );
}

async function layoutGraph(elkGraph: ElkNode) {
  const { default: ELK } = await import("elkjs/lib/elk.bundled.js");
  const elk = new ELK({ defaultLayoutOptions: elkLayoutOptions });
  return elk.layout(elkGraph);
}

function ArchitectureSvgNode({ node }: { node: RenderedNode }) {
  return (
    <g className={node.visual.tone === "muted" ? "architecture-node muted" : "architecture-node"}>
      <rect height={node.height} rx={0} width={node.width} x={node.x} y={node.y} />
      <text className="architecture-node-title" x={node.x + 14} y={node.y + 24}>
        {node.visual.label}
      </text>
      <text className="architecture-node-detail" x={node.x + 14} y={node.y + 45}>
        {node.visual.detailLines.map((line, lineIndex) => (
          <tspan
            dy={lineIndex === 0 ? 0 : diagramMetrics.node.detailLineHeight}
            key={line}
            x={node.x + 14}
          >
            {line}
          </tspan>
        ))}
      </text>
    </g>
  );
}

function buildArchitectureGraph(preview: PolicyArchitecturePreview): ArchitectureGraph {
  const visuals = new Map<string, NodeVisual>();
  const children: ElkNode[] = [];
  const edges: ElkExtendedEdge[] = [];
  const visibleLanes = preview.architecture_lanes.map((lane) =>
    lane.nodes.filter(isVisibleArchitectureNode),
  );
  const sourceNodes = visibleLanes.flat();
  const sourceById = new Map(sourceNodes.map((node) => [node.id, node]));

  for (const node of sourceNodes) {
    addVisualNode(node, visuals, children);
  }

  if (!visuals.has("concat")) {
    addJunctionNode("concat", visuals, children);
  } else {
    const concatVisual = visuals.get("concat");
    if (concatVisual !== undefined) {
      visuals.set("concat", { ...concatVisual, kind: "junction" });
      updateElkNodeSize(
        children,
        "concat",
        diagramMetrics.junctionSize,
        diagramMetrics.junctionSize,
      );
    }
  }

  connectLane(visibleLanes[0] ?? [], edges);
  connectLane(visibleLanes[1] ?? [], edges);
  addEdgeFromLastNode(visibleLanes[0] ?? [], "concat:image", edges);
  addEdgeFromLastNode(visibleLanes[1] ?? [], "concat:state", edges);

  if (visuals.has("fusion")) {
    addEdge("concat:out", "fusion", edges);
  }

  const fusionPath = ["fusion", "layer_norm", "lstm"].filter((id) => visuals.has(id));
  connectIds(fusionPath, edges);

  const headSource = sourceById.has("lstm") ? "lstm" : lastId(fusionPath);
  for (const headId of ["policy_head", "value_head", "heads"]) {
    if (headSource !== undefined && visuals.has(headId)) {
      addEdge(headSource, headId, edges);
    }
  }
  if (visuals.has("policy_head") && visuals.has("action_net")) {
    addEdge("policy_head", "action_net", edges);
  }
  if (visuals.has("value_head") && visuals.has("value_net")) {
    addEdge("value_head", "value_net", edges);
  }

  return {
    elkGraph: {
      id: "policy-architecture",
      children,
      edges,
    },
    visuals,
  };
}

function isVisibleArchitectureNode(node: ArchitectureNode) {
  return node.tone !== "muted";
}

function addVisualNode(
  node: ArchitectureNode,
  visuals: Map<string, NodeVisual>,
  children: ElkNode[],
) {
  const visual = nodeVisual(node);
  visuals.set(node.id, visual);
  children.push({
    id: node.id,
    height: visual.height,
    width: visual.width,
  });
}

function addJunctionNode(id: string, visuals: Map<string, NodeVisual>, children: ElkNode[]) {
  visuals.set(id, {
    detailLines: [],
    height: diagramMetrics.junctionSize,
    kind: "junction",
    label: "",
    tone: "normal",
    width: diagramMetrics.junctionSize,
  });
  children.push({
    id,
    height: diagramMetrics.junctionSize,
    layoutOptions: {
      "elk.portConstraints": "FIXED_POS",
    },
    ports: concatPorts(),
    width: diagramMetrics.junctionSize,
  });
}

function updateElkNodeSize(children: ElkNode[], id: string, width: number, height: number) {
  const node = children.find((child) => child.id === id);
  if (node === undefined) {
    return;
  }
  node.width = width;
  node.height = height;
  node.layoutOptions = {
    ...node.layoutOptions,
    "elk.portConstraints": "FIXED_POS",
  };
  node.ports = concatPorts();
}

function concatPorts() {
  const center = diagramMetrics.junctionSize / 2 - 1;
  return [
    concatPort("concat:image", center, 0, "NORTH"),
    concatPort("concat:state", center, diagramMetrics.junctionSize - 2, "SOUTH"),
    concatPort("concat:out", diagramMetrics.junctionSize - 2, center, "EAST"),
  ];
}

function concatPort(id: string, x: number, y: number, side: "EAST" | "NORTH" | "SOUTH") {
  return {
    id,
    height: 2,
    layoutOptions: {
      "elk.port.side": side,
    },
    width: 2,
    x,
    y,
  };
}

function nodeVisual(node: ArchitectureNode): NodeVisual {
  const detailLines = detailLinesForNode(node);
  const longestLineLength = Math.max(node.label.length, ...detailLines.map((line) => line.length));
  const width = clamp(
    Math.ceil(
      longestLineLength * diagramMetrics.node.characterWidth + diagramMetrics.node.paddingX,
    ),
    diagramMetrics.node.minWidth,
    diagramMetrics.node.maxWidth,
  );
  const height = Math.max(
    diagramMetrics.node.minHeight,
    diagramMetrics.node.titleAndPaddingHeight +
      detailLines.length * diagramMetrics.node.detailLineHeight,
  );

  return {
    detailLines,
    height,
    kind: "node",
    label: node.label,
    tone: node.tone,
    width,
  };
}

function detailLinesForNode(node: ArchitectureNode) {
  const detail = normalizeDetail(node.detail);
  if (node.id === "lstm") {
    return detail.split(", ").map((part) => part.trim());
  }
  if (node.id === "policy_head" || node.id === "value_head") {
    const [shape, activation] = splitAtLastComma(detail);
    return activation === undefined ? [shape] : [shape, activation];
  }
  return wrapDetailLine(detail);
}

function normalizeDetail(value: string) {
  return value.replaceAll(" -> ", " → ").replaceAll(", ", ",").replaceAll(",", ", ");
}

function splitAtLastComma(value: string): [string, string | undefined] {
  const commaIndex = value.lastIndexOf(",");
  if (commaIndex < 0) {
    return [value, undefined];
  }
  return [value.slice(0, commaIndex).trim(), value.slice(commaIndex + 1).trim()];
}

function wrapDetailLine(value: string) {
  const maxChars = 26;
  if (value.length <= maxChars) {
    return [value];
  }

  const words = value.split(" ");
  const lines: string[] = [];
  let currentLine = "";
  for (const word of words) {
    const nextLine = currentLine.length === 0 ? word : `${currentLine} ${word}`;
    if (nextLine.length <= maxChars) {
      currentLine = nextLine;
      continue;
    }
    if (currentLine.length > 0) {
      lines.push(currentLine);
    }
    currentLine = word;
  }
  if (currentLine.length > 0) {
    lines.push(currentLine);
  }
  return lines;
}

function connectLane(nodes: readonly ArchitectureNode[], edges: ElkExtendedEdge[]) {
  connectIds(
    nodes.filter((node) => node.id !== "concat").map((node) => node.id),
    edges,
  );
}

function addEdgeFromLastNode(
  nodes: readonly ArchitectureNode[],
  target: string,
  edges: ElkExtendedEdge[],
) {
  const source = nodes.filter((node) => node.id !== "concat").at(-1);
  if (source !== undefined) {
    addEdge(source.id, target, edges);
  }
}

function connectIds(ids: readonly string[], edges: ElkExtendedEdge[]) {
  for (const [source, target] of pairwise(ids)) {
    addEdge(source, target, edges);
  }
}

function pairwise(values: readonly string[]) {
  const pairs: Array<[string, string]> = [];
  for (let index = 0; index < values.length - 1; index += 1) {
    const source = values[index];
    const target = values[index + 1];
    if (source !== undefined && target !== undefined) {
      pairs.push([source, target]);
    }
  }
  return pairs;
}

function addEdge(source: string, target: string, edges: ElkExtendedEdge[]) {
  edges.push({
    id: `${source}-${target}`,
    sources: [source],
    targets: [target],
  });
}

function lastId(ids: readonly string[]) {
  return ids.at(-1);
}

function toRenderedGraph(elkGraph: ElkNode, visuals: Map<string, NodeVisual>): RenderedGraph {
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
    nodes,
    viewBox: `${-margin} ${-margin} ${width + margin * 2} ${height + margin * 2}`,
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
  const end = {
    x: concat.x,
    y: concat.y + concat.height / 2,
  };
  const mergeX = Math.max(start.x + 24, end.x - 24);
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

function fallbackGraph(visuals: Map<string, NodeVisual>): RenderedGraph {
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
    nodes,
    viewBox: `0 0 ${maxNodeExtent(nodes, "x")} ${maxNodeExtent(nodes, "y")}`,
  };
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

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

type ArchitectureNode = PolicyArchitecturePreview["architecture_lanes"][number]["nodes"][number];

interface ArchitectureGraph {
  elkGraph: ElkNode;
  visuals: Map<string, NodeVisual>;
}

interface NodeVisual {
  detailLines: string[];
  height: number;
  kind: "junction" | "node";
  label: string;
  tone: string;
  width: number;
}

interface RenderedGraph {
  edges: RenderedEdge[];
  nodes: RenderedNode[];
  viewBox: string;
}

interface RenderedEdge {
  hasArrow: boolean;
  id: string;
  path: string;
}

interface RenderedNode {
  height: number;
  id: string;
  visual: NodeVisual;
  width: number;
  x: number;
  y: number;
}
