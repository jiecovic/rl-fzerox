import { PolicyArchitectureDiagram } from "@/features/configurator/sections/PolicyArchitectureDiagram";
import {
  formatConvSpatial,
  formatFitMode,
  formatParamCount,
  formatPixelDrop,
} from "@/features/configurator/sections/policy/convPreviewFormatting";
import { CustomConvTableRows } from "@/features/configurator/sections/policy/LayerEditors";
import type { ManagedRunConfig, PolicyArchitecturePreview } from "@/shared/api/contract";

export function PolicyPreviewPanel({
  convProfile,
  customConvLayers,
  preview,
  setCustomConvLayers,
}: {
  convProfile: ManagedRunConfig["policy"]["conv_profile"];
  customConvLayers: ManagedRunConfig["policy"]["custom_conv_layers"];
  preview: PolicyArchitecturePreview | null;
  setCustomConvLayers: (value: ManagedRunConfig["policy"]["custom_conv_layers"]) => void;
}) {
  if (preview === null) {
    return <div className="preview-placeholder">Computing architecture preview...</div>;
  }

  return (
    <div className="policy-preview-grid">
      <section className="architecture-flow-panel">
        <h3>Architecture</h3>
        <div className="architecture-graph-shell">
          <div className="architecture-total-summary-box">
            <span className="architecture-total-label">total params</span>
            <strong className="architecture-total-value">
              {formatParamCount(preview.total_params)}
            </strong>
          </div>
          <PolicyArchitectureDiagram preview={preview} />
        </div>
      </section>

      <section className="conv-table-panel" id="policy-cnn-configurator">
        <h3>CNN layers</h3>
        <table
          className={
            convProfile === "custom"
              ? "derived-table conv-derived-table conv-derived-table-custom"
              : "derived-table conv-derived-table"
          }
        >
          <thead>
            <tr>
              <th>Layer</th>
              <th>Channels</th>
              <th>Kernel</th>
              <th>Stride</th>
              <th>Pad</th>
              <th>Input</th>
              <th>Output</th>
              <th>Fit</th>
              <th>Pixel drop</th>
              <th>Params</th>
              {convProfile === "custom" ? <th>Actions</th> : null}
            </tr>
          </thead>
          <tbody>
            {convProfile === "custom" ? (
              <CustomConvTableRows
                flattenDim={preview.flatten_dim}
                previewLayers={preview.conv_layers}
                value={customConvLayers}
                onChange={setCustomConvLayers}
              />
            ) : (
              preview.conv_layers.map((layer) => (
                <tr key={layer.name}>
                  <th>{layer.name}</th>
                  <td>
                    {layer.in_channels}
                    {" → "}
                    {layer.out_channels}
                  </td>
                  <td>{layer.kernel_size}</td>
                  <td>{layer.stride}</td>
                  <td>{layer.padding}</td>
                  <td>{formatConvSpatial(layer, "input")}</td>
                  <td>{formatConvSpatial(layer, "output")}</td>
                  <td>{formatFitMode(layer)}</td>
                  <td>{formatPixelDrop(layer)}</td>
                  <td>{formatParamCount(layer.params)}</td>
                </tr>
              ))
            )}
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
          <ShapeMetric
            label="Continuous action"
            value={preview.continuous_action_dims.toLocaleString()}
          />
          <ShapeMetric
            label="Discrete logits"
            value={preview.discrete_action_logits.toLocaleString()}
          />
        </div>
      </section>

      <section className="shape-panel">
        <h3>Action head</h3>
        <table className="derived-table">
          <thead>
            <tr>
              <th>Branch</th>
              <th>Kind</th>
              <th>Size</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {preview.action_branches.map((branch) => (
              <tr key={branch.name}>
                <th>{branch.name}</th>
                <td>{branch.kind}</td>
                <td>{branch.size}</td>
                <td>{branchStatus(branch)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function branchStatus(branch: PolicyArchitecturePreview["action_branches"][number]) {
  if (!branch.enabled) {
    return branch.mask_label ?? "masked";
  }
  if (branch.mask_label === null) {
    return "trainable";
  }
  return (
    <span className="derived-value-stack">
      <strong>trainable</strong>
      <span className="derived-value-note">{branch.mask_label}</span>
    </span>
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
