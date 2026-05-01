import { PolicyArchitectureDiagram } from "@/features/configurator/sections/PolicyArchitectureDiagram";
import type { PolicyArchitecturePreview } from "@/shared/api/contract";

export function PolicyPreviewPanel({ preview }: { preview: PolicyArchitecturePreview | null }) {
  if (preview === null) {
    return <div className="preview-placeholder">Computing architecture preview...</div>;
  }

  return (
    <div className="policy-preview-grid">
      <section className="architecture-flow-panel">
        <h3>Architecture</h3>
        <PolicyArchitectureDiagram preview={preview} />
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
