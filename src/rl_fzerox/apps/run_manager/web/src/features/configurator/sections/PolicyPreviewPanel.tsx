// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/PolicyPreviewPanel.tsx
import { PolicyArchitectureDiagram } from "@/features/configurator/sections/PolicyArchitectureDiagram";
import { CustomConvTableRows } from "@/features/configurator/sections/policy/LayerEditors";
import type { ManagedRunConfig, PolicyArchitecturePreview } from "@/shared/api/contract";
import { Button } from "@/shared/ui/Button";

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
    return (
      <div className="border border-app-border bg-app-surface p-4 text-app-muted">
        Computing architecture preview...
      </div>
    );
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
    <div className="grid gap-3">
      <section className={previewPanelClass}>
        <PolicyArchitectureDiagram preview={preview} />
      </section>

      <section className={previewPanelClass} id="policy-cnn-configurator">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h3 className="m-0 text-[15px] font-semibold text-app-text">CNN layers</h3>
          {canConvertPreset ? (
            <Button className="h-8 px-3" onClick={() => convertPresetToCustom(editableConvLayers)}>
              Edit as custom
            </Button>
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

      <section className={previewPanelClass}>
        <h3 className={previewHeadingClass}>Feature dimensions</h3>
        <div className="grid grid-cols-3 gap-2.5 max-[900px]:grid-cols-2 max-[560px]:grid-cols-1">
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

      <section className={previewPanelClass}>
        <h3 className={previewHeadingClass}>Action head</h3>
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
    <div className="border border-app-border bg-app-surface p-2.5">
      <span className="mb-1 block text-xs text-app-muted">{label}</span>
      <strong className="font-bold tabular-nums text-app-text">{formatPreviewText(value)}</strong>
    </div>
  );
}

function formatPreviewText(value: string) {
  return value.replaceAll(" -> ", " → ");
}

const previewPanelClass = "col-span-full border border-app-border bg-app-surface p-3";
const previewHeadingClass = "mb-3 text-[15px] font-semibold text-app-text";
