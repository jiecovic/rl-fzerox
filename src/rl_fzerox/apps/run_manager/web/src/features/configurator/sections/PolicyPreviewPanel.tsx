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
  const imageNodes = layoutHorizontal(imageLane?.nodes ?? [], 40, 48);
  const stateNodes = layoutHorizontal(stateLane?.nodes ?? [], 40, 204);
  const { headNodes, trunkNodes } = layoutFusionAndHeads(
    (trunkLane?.nodes ?? []).filter((node) => node.id !== "concat"),
  );
  const mergePoint = {
    x: 590,
    y: trunkNodes[0] === undefined ? 155 : middleY(trunkNodes[0]),
  };

  return (
    <svg
      aria-label="Policy architecture diagram"
      className="architecture-svg"
      role="img"
      viewBox="0 0 980 550"
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
      <DiagramLaneLabel label={stateLane?.label ?? "State branch"} x={40} y={180} />
      <DiagramLaneLabel label={trunkLane?.label ?? "Fusion and heads"} x={660} y={102} />

      {connectSequential(imageNodes)}
      {connectSequential(stateNodes)}
      {connectToMerge(imageNodes.at(-1), mergePoint, "image")}
      {connectToMerge(stateNodes.at(-1), mergePoint, "state")}
      {connectPointToNode(mergePoint, trunkNodes[0], "concat")}
      {connectVertical(trunkNodes)}
      {connectHeadBranch(trunkNodes.at(-1), headNodes)}

      {imageNodes.map((item) => (
        <DiagramNode item={item} key={item.node.id} />
      ))}
      {stateNodes.map((item) => (
        <DiagramNode item={item} key={item.node.id} />
      ))}
      <DiagramMergePoint point={mergePoint} />
      {trunkNodes.map((item) => (
        <DiagramNode item={item} key={item.node.id} />
      ))}
      {headNodes.map((item) => (
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
  return (
    <foreignObject height={item.height} width={item.width} x={item.x} y={item.y}>
      <div
        className={
          item.node.tone === "muted" ? "architecture-svg-node muted" : "architecture-svg-node"
        }
      >
        <strong>{item.node.label}</strong>
        <span>{formatPreviewText(item.node.detail)}</span>
      </div>
    </foreignObject>
  );
}

function DiagramMergePoint({ point }: { point: DiagramPoint }) {
  return (
    <g>
      <circle className="architecture-merge-dot" cx={point.x} cy={point.y} r={4} />
    </g>
  );
}

function layoutHorizontal(nodes: readonly ArchitectureNode[], startX: number, y: number) {
  return nodes.map((node, index) => layoutNode(node, startX + index * 164, y, 146));
}

function layoutVertical(nodes: readonly ArchitectureNode[], x: number, startY: number) {
  return nodes.map((node, index) => layoutNode(node, x, startY + index * 96, 250, 70));
}

function layoutFusionAndHeads(nodes: readonly ArchitectureNode[]) {
  const headSourceNodes = nodes.filter((node) => isHeadNode(node));
  const trunkSourceNodes = nodes.filter((node) => !isHeadNode(node));
  const trunkNodes = layoutVertical(trunkSourceNodes, 660, 126);
  const headNodes =
    headSourceNodes.length === 1
      ? [layoutNode(headSourceNodes[0], 660, 424, 250, 70)]
      : headSourceNodes.map((node, index) => layoutNode(node, 500 + index * 230, 424, 210, 70));
  return { headNodes, trunkNodes };
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
  const elbowX = merge.x - 48;
  return connectorPath(
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

function connectVertical(nodes: readonly DiagramNodeLayout[]) {
  return nodes.slice(0, -1).map((node, index) => {
    const next = nodes[index + 1];
    if (next === undefined) {
      return null;
    }
    const x = node.x + node.width / 2;
    return connectorPath(
      `trunk-${node.node.id}-${next.node.id}`,
      `M ${x} ${bottom(node)} V ${next.y}`,
    );
  });
}

function connectHeadBranch(
  from: DiagramNodeLayout | undefined,
  heads: readonly DiagramNodeLayout[],
) {
  if (from === undefined || heads.length === 0) {
    return null;
  }
  const sourceX = from.x + from.width / 2;
  const sourceY = bottom(from);
  if (heads.length === 1) {
    const [head] = heads;
    return connectorPath("head-single", `M ${sourceX} ${sourceY} V ${head.y}`);
  }
  const branchY = sourceY + 28;
  return (
    <g>
      {connectorLine("head-stem", `M ${sourceX} ${sourceY} V ${branchY}`)}
      {heads.map((head) =>
        connectorPath(
          `head-${head.node.id}`,
          `M ${sourceX} ${branchY} H ${head.x + head.width / 2} V ${head.y}`,
        ),
      )}
    </g>
  );
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

function bottom(node: DiagramNodeLayout) {
  return node.y + node.height;
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
