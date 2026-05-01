import type { PolicyArchitecturePreview } from "@/shared/api/contract";

export function PolicyPreviewPanel({ preview }: { preview: PolicyArchitecturePreview | null }) {
  if (preview === null) {
    return <div className="preview-placeholder">Computing architecture preview...</div>;
  }

  return (
    <div className="policy-preview-grid">
      <section className="architecture-flow-panel">
        <h3>Architecture</h3>
        <ArchitectureDiagram preview={preview} />
      </section>

      <section className="parameter-panel">
        <h3>Trainable parameters</h3>
        <strong className="parameter-total">{formatParams(preview.total_params)}</strong>
        <div className="parameter-breakdown">
          {preview.parameter_groups.map((group) => (
            <div className="parameter-row" key={group.name}>
              <span>{group.name}</span>
              <strong>{formatParams(group.params)}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="conv-table-panel">
        <h3>CNN layers</h3>
        <table className="derived-table">
          <thead>
            <tr>
              <th>Layer</th>
              <th>Channels</th>
              <th>Kernel</th>
              <th>Output</th>
              <th>Params</th>
            </tr>
          </thead>
          <tbody>
            {preview.conv_layers.map((layer) => (
              <tr key={layer.name}>
                <th>{layer.name}</th>
                <td>
                  {layer.in_channels}
                  {" → "}
                  {layer.out_channels}
                </td>
                <td>
                  {layer.kernel_size} / {layer.stride}
                </td>
                <td>
                  {layer.output_height} x {layer.output_width}
                </td>
                <td>{formatParams(layer.params)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="shape-panel">
        <h3>Feature dimensions</h3>
        <div className="shape-summary-grid compact">
          <ShapeMetric label="Flatten" value={preview.flatten_dim.toLocaleString()} />
          <ShapeMetric label="Image features" value={preview.image_features_dim.toLocaleString()} />
          <ShapeMetric
            label="State MLP"
            value={`${preview.state_dim} → ${preview.state_features_dim}`}
          />
          <ShapeMetric label="Fusion input" value={preview.fusion_input_dim.toLocaleString()} />
          <ShapeMetric
            label="Extractor out"
            value={preview.extractor_output_dim.toLocaleString()}
          />
          <ShapeMetric label="Policy input" value={preview.policy_input_dim.toLocaleString()} />
        </div>
      </section>
    </div>
  );
}

function ArchitectureDiagram({ preview }: { preview: PolicyArchitecturePreview }) {
  const [imageLane, stateLane, trunkLane] = preview.architecture_lanes;
  const imageNodes = layoutBranch(imageLane?.nodes ?? [], 40, 44);
  const stateNodes = layoutBranch(stateLane?.nodes ?? [], 40, 176);
  const pipelineNodes = layoutFusionPipeline(
    (trunkLane?.nodes ?? []).filter((node) => node.id !== "concat"),
  );
  const mergePoint = {
    x: 478,
    y: pipelineNodes[0] === undefined ? 130 : middleY(pipelineNodes[0]),
  };

  return (
    <svg
      aria-label="Policy architecture diagram"
      className="architecture-svg"
      role="img"
      viewBox="0 0 1000 310"
    >
      <defs>
        <marker
          id="architecture-arrow"
          markerHeight="8"
          markerWidth="8"
          orient="auto"
          refX="7"
          refY="4"
          viewBox="0 0 8 8"
        >
          <path className="architecture-arrow-head" d="M0 0 8 4 0 8z" />
        </marker>
      </defs>
      <DiagramLaneLabel label={imageLane?.label ?? "Image branch"} x={40} y={24} />
      <DiagramLaneLabel label={stateLane?.label ?? "State branch"} x={40} y={156} />
      <DiagramLaneLabel label={trunkLane?.label ?? "Fusion and heads"} x={500} y={72} />

      {connectSequential(imageNodes)}
      {connectSequential(stateNodes)}
      {connectToMerge(imageNodes.at(-1), mergePoint, "image")}
      {connectToMerge(stateNodes.at(-1), mergePoint, "state")}
      {connectPointToNode(mergePoint, pipelineNodes[0], "concat")}
      {connectSequential(pipelineNodes)}

      {imageNodes.map((item) => (
        <DiagramNode item={item} key={item.node.id} />
      ))}
      {stateNodes.map((item) => (
        <DiagramNode item={item} key={item.node.id} />
      ))}
      <DiagramMergePoint point={mergePoint} />
      {pipelineNodes.map((item) => (
        <DiagramNode item={item} key={item.node.id} />
      ))}
    </svg>
  );
}

function DiagramLaneLabel({ label, x, y }: { label: string; x: number; y: number }) {
  return (
    <text className="architecture-lane-title" x={x} y={y}>
      {label}
    </text>
  );
}

function DiagramNode({ item }: { item: DiagramNodeLayout }) {
  const labelX = item.x + 12;
  const detailLines = wrapNodeDetail(formatPreviewText(item.node.detail), item.width);

  return (
    <g
      className={
        item.node.tone === "muted" ? "architecture-svg-node muted" : "architecture-svg-node"
      }
    >
      <rect height={item.height - 1} width={item.width - 1} x={item.x + 0.5} y={item.y + 0.5} />
      <text className="architecture-node-label" x={labelX} y={item.y + 22}>
        {item.node.label}
      </text>
      <text className="architecture-node-detail" x={labelX} y={item.y + 41}>
        {detailLines.map((line, index) => (
          <tspan dy={index === 0 ? 0 : 14} key={line} x={labelX}>
            {line}
          </tspan>
        ))}
      </text>
    </g>
  );
}

function DiagramMergePoint({ point }: { point: DiagramPoint }) {
  return (
    <g>
      <circle className="architecture-merge-dot" cx={point.x} cy={point.y} r={4} />
    </g>
  );
}

function layoutBranch(nodes: readonly ArchitectureNode[], startX: number, y: number) {
  return nodes.map((node, index) => layoutNode(node, startX + index * 146, y, 132, 88));
}

function layoutFusionPipeline(nodes: readonly ArchitectureNode[]) {
  const headNodes = nodes.filter((node) => isHeadNode(node));
  const pipelineNodes = nodes.filter((node) => !isHeadNode(node));
  const combinedHead = combineHeadNodes(headNodes);
  if (combinedHead !== null) {
    pipelineNodes.push(combinedHead);
  }
  return layoutPipeline(pipelineNodes, 500, 92);
}

function layoutPipeline(nodes: readonly ArchitectureNode[], startX: number, y: number) {
  let x = startX;
  return nodes.map((node) => {
    const item = layoutNode(node, x, y, pipelineNodeWidth(node), 96);
    x += item.width + 14;
    return item;
  });
}

function pipelineNodeWidth(node: ArchitectureNode) {
  if (node.id === "layer_norm") {
    return 96;
  }
  if (isHeadNode(node)) {
    return 120;
  }
  if (node.id === "lstm") {
    return 122;
  }
  return 118;
}

function combineHeadNodes(nodes: readonly ArchitectureNode[]): ArchitectureNode | null {
  if (nodes.length === 0) {
    return null;
  }
  if (nodes.length === 1) {
    return nodes[0] ?? null;
  }
  const policy = nodes.find((node) => node.id === "policy_head");
  const value = nodes.find((node) => node.id === "value_head");
  return {
    id: "heads",
    label: "Heads",
    detail: `pi ${compactHeadDetail(policy)}\nvf ${compactHeadDetail(value)}`,
    tone: "normal",
  };
}

function compactHeadDetail(node: ArchitectureNode | undefined) {
  if (node === undefined) {
    return "";
  }
  const detail = formatPreviewText(node.detail);
  const [, output] = detail.split("→");
  return (output ?? detail)
    .trim()
    .replace(/,\s*[^,]+$/, "")
    .replaceAll(", ", ",");
}

function isHeadNode(node: ArchitectureNode) {
  return node.id === "heads" || node.id === "policy_head" || node.id === "value_head";
}

function layoutNode(
  node: ArchitectureNode,
  x: number,
  y: number,
  width: number,
  height = 58,
): DiagramNodeLayout {
  return { height, node, width, x, y };
}

function connectSequential(nodes: readonly DiagramNodeLayout[]) {
  return nodes.slice(0, -1).map((node, index) => {
    const next = nodes[index + 1];
    if (next === undefined) {
      return null;
    }
    return connectorPath(
      `seq-${node.node.id}-${next.node.id}`,
      `M ${right(node)} ${middleY(node)} H ${next.x}`,
    );
  });
}

function connectToMerge(from: DiagramNodeLayout | undefined, merge: DiagramPoint, id: string) {
  if (from === undefined) {
    return null;
  }
  const elbowX = merge.x - 28;
  return connectorLine(
    `merge-${id}`,
    `M ${right(from)} ${middleY(from)} H ${elbowX} V ${merge.y} H ${merge.x}`,
  );
}

function connectPointToNode(from: DiagramPoint, to: DiagramNodeLayout | undefined, id: string) {
  if (to === undefined) {
    return null;
  }
  return connectorPath(`point-${id}`, `M ${from.x} ${from.y} H ${to.x}`);
}

function connectorPath(key: string, d: string) {
  return (
    <path
      className="architecture-connector"
      d={d}
      fill="none"
      key={key}
      markerEnd="url(#architecture-arrow)"
    />
  );
}

function connectorLine(key: string, d: string) {
  return <path className="architecture-connector" d={d} fill="none" key={key} />;
}

function right(node: DiagramNodeLayout) {
  return node.x + node.width;
}

function middleY(node: DiagramNodeLayout) {
  return node.y + node.height / 2;
}

type ArchitectureNode = PolicyArchitecturePreview["architecture_lanes"][number]["nodes"][number];

interface DiagramPoint {
  x: number;
  y: number;
}

interface DiagramNodeLayout {
  height: number;
  node: ArchitectureNode;
  width: number;
  x: number;
  y: number;
}

function ShapeMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="shape-metric">
      <span>{label}</span>
      <strong>{formatPreviewText(value)}</strong>
    </div>
  );
}

function formatPreviewText(value: string) {
  return value.replaceAll(" -> ", " → ");
}

function wrapNodeDetail(value: string, width: number) {
  const maxChars = Math.max(12, Math.floor((width - 24) / 6.2));
  const lines = value.split("\n").flatMap((line) => wrapTextLine(line, maxChars));

  if (lines.length <= 3) {
    return lines;
  }
  return [lines[0] ?? "", lines[1] ?? "", `${lines[2] ?? ""}…`];
}

function wrapTextLine(value: string, maxChars: number) {
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
      currentLine = word;
      continue;
    }
    lines.push(word.slice(0, maxChars - 1));
    currentLine = word.slice(maxChars - 1);
  }

  if (currentLine.length > 0) {
    lines.push(currentLine);
  }
  return lines;
}

function formatParams(value: number) {
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(2)}B`;
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return value.toLocaleString();
}
