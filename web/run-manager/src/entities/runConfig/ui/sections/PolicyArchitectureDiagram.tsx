// web/run-manager/src/entities/runConfig/ui/sections/PolicyArchitectureDiagram.tsx
import { useEffect, useState } from "react";
import { formatParamCount } from "@/entities/runConfig/ui/sections/policy/convPreviewFormatting";
import { diagramMetrics } from "@/entities/runConfig/ui/sections/policyArchitectureDiagram/constants";
import { layoutGraph } from "@/entities/runConfig/ui/sections/policyArchitectureDiagram/layout";
import { buildArchitectureGraph } from "@/entities/runConfig/ui/sections/policyArchitectureDiagram/model";
import {
  fallbackGraph,
  toRenderedGraph,
} from "@/entities/runConfig/ui/sections/policyArchitectureDiagram/render";
import type {
  RenderedGraph,
  RenderedNode,
} from "@/entities/runConfig/ui/sections/policyArchitectureDiagram/types";
import type { PolicyArchitecturePreview } from "@/shared/api/contract";

export function PolicyArchitectureDiagram({ preview }: { preview: PolicyArchitecturePreview }) {
  const [layout, setLayout] = useState<RenderedGraph | null>(null);

  useEffect(() => {
    let cancelled = false;
    const graph = buildArchitectureGraph(preview);

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
    <div className="architecture-graph-frame">
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

        <ArchitectureSummaryBox totalParams={preview.total_params} />

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
    </div>
  );
}

function ArchitectureSummaryBox({ totalParams }: { totalParams: number }) {
  const box = diagramMetrics.summaryBox;
  const x = box.xInset;
  const y = box.yInset;
  return (
    <g className="architecture-summary">
      <rect height={box.height} width={box.width} x={x} y={y} />
      <text className="architecture-summary-label" x={x + box.paddingX} y={y + 17}>
        total params
      </text>
      <text className="architecture-summary-value" x={x + box.paddingX} y={y + 36}>
        {formatParamCount(totalParams)}
      </text>
    </g>
  );
}

function ArchitectureSvgNode({ node }: { node: RenderedNode }) {
  const detailY = node.y + 45;
  const paramY =
    detailY +
    node.visual.detailLines.length * diagramMetrics.node.detailLineHeight +
    diagramMetrics.node.paramTopGap;
  return (
    <g className={node.visual.tone === "muted" ? "architecture-node muted" : "architecture-node"}>
      <rect height={node.height} rx={0} width={node.width} x={node.x} y={node.y} />
      <text className="architecture-node-title" x={node.x + 14} y={node.y + 24}>
        {node.visual.label}
      </text>
      <text className="architecture-node-detail" x={node.x + 14} y={detailY}>
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
      {node.visual.paramLine ? (
        <text className="architecture-node-param" x={node.x + 14} y={paramY}>
          {node.visual.paramLine}
        </text>
      ) : null}
    </g>
  );
}
