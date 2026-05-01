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
                  {" -> "}
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
            value={`${preview.state_dim} -> ${preview.state_features_dim}`}
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
  const imageNodes = layoutHorizontal(imageLane?.nodes ?? [], 40, 44);
  const stateNodes = layoutHorizontal(stateLane?.nodes ?? [], 40, 186);
  const concatNode = trunkLane?.nodes.find((node) => node.id === "concat") ?? {
    id: "concat",
    label: "Concat",
    detail: preview.fusion_input_dim.toLocaleString(),
    tone: "normal",
  };
  const concat = layoutNode(concatNode, 560, 114, 132);
  const trunkNodes = layoutVertical(
    (trunkLane?.nodes ?? []).filter((node) => node.id !== "concat"),
    760,
    24,
  );

  return (
    <svg
      aria-label="Policy architecture diagram"
      className="architecture-svg"
      role="img"
      viewBox="0 0 980 384"
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
      <DiagramLaneLabel label={imageLane?.label ?? "Image branch"} x={40} y={20} />
      <DiagramLaneLabel label={stateLane?.label ?? "State branch"} x={40} y={162} />
      <DiagramLaneLabel label={trunkLane?.label ?? "Fusion and heads"} x={760} y={0} />

      {connectSequential(imageNodes)}
      {connectSequential(stateNodes)}
      {connectToMerge(imageNodes.at(-1), concat, 118, "image")}
      {connectToMerge(stateNodes.at(-1), concat, 168, "state")}
      {connectHorizontal(concat, trunkNodes[0], "concat")}
      {connectVertical(trunkNodes)}

      {imageNodes.map((item) => (
        <DiagramNode item={item} key={item.node.id} />
      ))}
      {stateNodes.map((item) => (
        <DiagramNode item={item} key={item.node.id} />
      ))}
      <DiagramNode item={concat} />
      {trunkNodes.map((item) => (
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
        <span>{item.node.detail}</span>
      </div>
    </foreignObject>
  );
}

function layoutHorizontal(nodes: readonly ArchitectureNode[], startX: number, y: number) {
  return nodes.map((node, index) => layoutNode(node, startX + index * 168, y, 138));
}

function layoutVertical(nodes: readonly ArchitectureNode[], x: number, startY: number) {
  return nodes.map((node, index) => layoutNode(node, x, startY + index * 86, 176));
}

function layoutNode(
  node: ArchitectureNode,
  x: number,
  y: number,
  width: number,
): DiagramNodeLayout {
  return { height: 58, node, width, x, y };
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

function connectToMerge(
  from: DiagramNodeLayout | undefined,
  merge: DiagramNodeLayout,
  mergeY: number,
  id: string,
) {
  if (from === undefined) {
    return null;
  }
  const elbowX = merge.x - 34;
  return connectorPath(
    `merge-${id}`,
    `M ${right(from)} ${middleY(from)} H ${elbowX} V ${mergeY} H ${merge.x}`,
  );
}

function connectHorizontal(from: DiagramNodeLayout, to: DiagramNodeLayout | undefined, id: string) {
  if (to === undefined) {
    return null;
  }
  return connectorPath(`horizontal-${id}`, `M ${right(from)} ${middleY(from)} H ${to.x}`);
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
      <strong>{value}</strong>
    </div>
  );
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
