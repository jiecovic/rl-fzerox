// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/policyArchitectureDiagram/model.ts
import type { ElkExtendedEdge, ElkNode } from "elkjs/lib/elk.bundled.js";
import { formatParamCount } from "@/features/configurator/sections/policy/convPreviewFormatting";
import { diagramMetrics } from "@/features/configurator/sections/policyArchitectureDiagram/constants";
import type {
  ArchitectureGraph,
  ArchitectureNode,
  NodeVisual,
} from "@/features/configurator/sections/policyArchitectureDiagram/types";
import type { PolicyArchitecturePreview } from "@/shared/api/contract";

export function buildArchitectureGraph(preview: PolicyArchitecturePreview): ArchitectureGraph {
  const visuals = new Map<string, NodeVisual>();
  const children: ElkNode[] = [];
  const edges: ElkExtendedEdge[] = [];
  const visibleLanes = preview.architecture_lanes.map((lane) =>
    lane.nodes.filter(isVisibleArchitectureNode),
  );
  const sourceNodes = visibleLanes.flat();

  for (const node of sourceNodes) {
    addVisualNode(node, visuals, children);
  }

  if (!visuals.has("concat")) {
    addJunctionNode("concat", visuals, children, concatPorts());
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

  const postConcatPath = ["fusion", "layer_norm", "lstm"].filter((id) => visuals.has(id));
  const firstPostConcatNode = postConcatPath[0];
  if (firstPostConcatNode !== undefined) {
    addEdge("concat:out", firstPostConcatNode, edges);
  }
  connectIds(postConcatPath, edges);

  const headSource = lastId(postConcatPath) ?? "concat:out";
  const headIds = ["policy_head", "aux_head", "value_head"].filter((id) => visuals.has(id));
  if (headSource !== undefined && headIds.length > 1) {
    addJunctionNode("heads", visuals, children);
    addEdge(headSource, "heads", edges);
    for (const headId of headIds) {
      addEdge("heads", headId, edges);
    }
  } else {
    for (const headId of headIds) {
      if (headSource !== undefined) {
        addEdge(headSource, headId, edges);
      }
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

function addJunctionNode(
  id: string,
  visuals: Map<string, NodeVisual>,
  children: ElkNode[],
  ports?: ElkNode["ports"],
) {
  visuals.set(id, {
    detailLines: [],
    height: diagramMetrics.junctionSize,
    kind: "junction",
    label: "",
    tone: "normal",
    width: diagramMetrics.junctionSize,
  });
  const node: ElkNode = {
    id,
    height: diagramMetrics.junctionSize,
    width: diagramMetrics.junctionSize,
  };
  if (ports !== undefined) {
    node.layoutOptions = {
      "elk.portConstraints": "FIXED_POS",
    };
    node.ports = ports;
  }
  children.push(node);
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

function concatPort(id: string, x: number, y: number, side: "EAST" | "NORTH" | "SOUTH" | "WEST") {
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
  const paramLine =
    node.params === null || node.params === undefined ? undefined : formatParamLine(node.params);
  const longestLineLength = Math.max(
    node.label.length,
    ...(paramLine === undefined ? [] : [paramLine.length]),
    ...detailLines.map((line) => line.length),
  );
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
      detailLines.length * diagramMetrics.node.detailLineHeight +
      (paramLine === undefined
        ? 0
        : diagramMetrics.node.paramTopGap + diagramMetrics.node.paramLineHeight),
  );

  return {
    detailLines,
    height,
    kind: "node",
    label: node.label,
    paramLine,
    tone: node.tone,
    width,
  };
}

function detailLinesForNode(node: ArchitectureNode) {
  const detail = normalizeDetail(node.detail);
  if (node.id === "lstm") {
    return detail.split(", ").map((part) => part.trim());
  }
  if (activationDetailNodeIds.has(node.id)) {
    const [shape, activation] = splitAtLastComma(detail);
    return activation === undefined ? [shape] : [shape, activation];
  }
  return wrapDetailLine(detail);
}

const activationDetailNodeIds = new Set([
  "cnn",
  "image_projection",
  "state_mlp",
  "fusion",
  "policy_head",
  "aux_head",
  "value_head",
]);

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

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function formatParamLine(value: number) {
  return `${formatParamCount(value)} params`;
}
