// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/PolicyPreviewPanel.tsx
import { PolicyArchitectureDiagram } from "@/features/configurator/sections/PolicyArchitectureDiagram";
import { formatParamCount } from "@/features/configurator/sections/policy/convPreviewFormatting";
import { CustomConvTableRows } from "@/features/configurator/sections/policy/LayerEditors";
import type { ManagedRunConfig, PolicyArchitecturePreview } from "@/shared/api/contract";

export function PolicyPreviewPanel({
  checkpointLocked = false,
  convProfile,
  customConvLayers,
  preview,
  convertPresetToCustom,
  setCustomConvLayers,
}: {
  checkpointLocked?: boolean;
  convProfile: ManagedRunConfig["policy"]["conv_profile"];
  customConvLayers: ManagedRunConfig["policy"]["custom_conv_layers"];
  preview: PolicyArchitecturePreview | null;
  convertPresetToCustom: (value: ManagedRunConfig["policy"]["custom_conv_layers"]) => void;
  setCustomConvLayers: (value: ManagedRunConfig["policy"]["custom_conv_layers"]) => void;
}) {
  if (preview === null) {
    return <div className="preview-placeholder">Computing architecture preview...</div>;
  }
  const isCustomProfile = convProfile === "custom";
  const canConvertPreset = !checkpointLocked && !isCustomProfile;
  const editorDisabledReason = checkpointLocked
    ? "Forked checkpoints keep the original CNN extractor."
    : "Preset CNNs are read-only. Use Edit as custom first.";
  const editableConvLayers = isCustomProfile
    ? customConvLayers
    : customConvLayersFromPreview(preview.conv_layers);

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
        <div className="conv-table-heading">
          <h3>CNN layers</h3>
          {canConvertPreset ? (
            <button
              className="secondary-button"
              type="button"
              onClick={() => convertPresetToCustom(editableConvLayers)}
            >
              Edit as custom
            </button>
          ) : null}
        </div>
        <table className="derived-table conv-derived-table conv-derived-table-custom">
          <thead>
            <tr>
              <th>Layer</th>
              <th>Channels</th>
              <th>Kernel</th>
              <th>Stride</th>
              <th>Pad</th>
              <th>Activation</th>
              <th>Input</th>
              <th>Output</th>
              <th>Fit</th>
              <th>Pixel drop</th>
              <th>Params</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <CustomConvTableRows
              disabled={checkpointLocked || !isCustomProfile}
              disabledReason={editorDisabledReason}
              flattenDim={preview.flatten_dim}
              previewLayers={preview.conv_layers}
              value={editableConvLayers}
              onChange={setCustomConvLayers}
            />
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

function customConvLayersFromPreview(
  layers: PolicyArchitecturePreview["conv_layers"],
): ManagedRunConfig["policy"]["custom_conv_layers"] {
  return layers.map((layer) => ({
    kind: layer.kind,
    out_channels: layer.out_channels,
    kernel_size: layer.kernel_size,
    stride: layer.stride,
    padding: layer.padding,
    post_activation: layer.post_activation,
    activation: layer.activation,
  }));
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
