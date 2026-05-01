import type { ElkNode } from "elkjs/lib/elk.bundled.js";
import type { PolicyArchitecturePreview } from "@/shared/api/contract";

export type ArchitectureNode =
  PolicyArchitecturePreview["architecture_lanes"][number]["nodes"][number];

export interface ArchitectureGraph {
  elkGraph: ElkNode;
  visuals: Map<string, NodeVisual>;
}

export interface NodeVisual {
  detailLines: string[];
  height: number;
  kind: "junction" | "node";
  label: string;
  tone: string;
  width: number;
}

export interface RenderedGraph {
  edges: RenderedEdge[];
  nodes: RenderedNode[];
  viewBox: string;
}

export interface RenderedEdge {
  hasArrow: boolean;
  id: string;
  path: string;
}

export interface RenderedNode {
  height: number;
  id: string;
  visual: NodeVisual;
  width: number;
  x: number;
  y: number;
}
