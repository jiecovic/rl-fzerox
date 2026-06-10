// web/run-manager/src/entities/runConfig/ui/sections/policy/layerEditors/CustomConvTableRows.tsx
import { useRef } from "react";
import {
  formatConvSpatial,
  formatFitMode,
  formatFlattenSummary,
  formatParamCount,
  formatPixelDrop,
} from "@/entities/runConfig/ui/sections/policy/convPreviewFormatting";
import {
  createCustomCnnLayer,
  customCnnActivation,
  customCnnLayerKind,
  duplicateConvLayerFrom,
  isActivationLayerKind,
  isPoolingLayerKind,
  isResidualLayerKind,
  layerActivationLabel,
  layerLabel,
  withCustomCnnActivation,
  withCustomCnnKind,
  withCustomCnnNumericValue,
} from "@/entities/runConfig/ui/sections/policy/layerEditors/customCnnLayers";
import { syncLayerRowIds } from "@/entities/runConfig/ui/sections/policy/layerEditors/layerRows";
import type {
  CustomCnnActivation,
  CustomCnnLayerKind,
  CustomCnnNumericKey,
} from "@/entities/runConfig/ui/sections/policy/layerEditors/types";
import {
  customCnnRowClass,
  useCustomCnnLayerDrag,
} from "@/entities/runConfig/ui/sections/policy/layerEditors/useCustomCnnLayerDrag";
import type { ManagedRunConfig, PolicyArchitecturePreview } from "@/shared/api/contract";
import { IntegerTextInput } from "@/shared/ui/configFields/basicNumbers";
import { AddLayerIcon, RemoveLayerIcon } from "@/shared/ui/icons";
import { AppTooltip } from "@/shared/ui/Tooltip";

export function CustomConvTableRows({
  disabled = false,
  disabledReason = "CNN layers are locked.",
  flattenDim,
  previewLayers,
  value,
  onChange,
}: {
  disabled?: boolean;
  disabledReason?: string;
  flattenDim: number;
  previewLayers: PolicyArchitecturePreview["conv_layers"];
  value: ManagedRunConfig["policy"]["custom_conv_layers"];
  onChange: (value: ManagedRunConfig["policy"]["custom_conv_layers"]) => void;
}) {
  const rowIdsRef = useRef<string[]>([]);
  const { draggedLayerIndex, dragOverLayerIndex, layerDragProps } = useCustomCnnLayerDrag({
    disabled,
    value,
    onChange,
  });
  syncLayerRowIds(rowIdsRef.current, value.length, "custom-conv");

  function setLayerValue(index: number, key: CustomCnnNumericKey, nextValue: number) {
    onChange(
      value.map((layer, layerIndex) => {
        if (layerIndex !== index) {
          return layer;
        }
        return withCustomCnnNumericValue(layer, key, nextValue) ?? layer;
      }),
    );
  }

  function setLayerKind(index: number, kind: CustomCnnLayerKind) {
    onChange(
      value.map((layer, layerIndex) =>
        layerIndex === index ? withCustomCnnKind(layer, kind) : layer,
      ),
    );
  }

  function setLayerPostActivation(index: number, enabled: boolean) {
    onChange(
      value.map((layer, layerIndex) =>
        layerIndex === index ? { ...layer, post_activation: enabled } : layer,
      ),
    );
  }

  function setLayerActivation(index: number, activation: CustomCnnActivation) {
    onChange(
      value.map((layer, layerIndex) =>
        layerIndex === index ? withCustomCnnActivation(layer, activation) : layer,
      ),
    );
  }

  function addLayer(kind: CustomCnnLayerKind) {
    onChange([...value, createCustomCnnLayer(kind, value.at(-1))]);
  }

  function duplicateLastConvLayer() {
    onChange([...value, duplicateConvLayerFrom(value.at(-1))]);
  }

  function removeLayer(index: number) {
    if (value.length <= 1) {
      return;
    }
    onChange(value.filter((_, layerIndex) => layerIndex !== index));
  }

  return (
    <>
      {value.map((layer, index) => {
        const preview = previewLayers[index] ?? null;
        return (
          <tr
            className={customCnnRowClass(index, draggedLayerIndex, dragOverLayerIndex, disabled)}
            key={rowIdsRef.current[index]}
            {...layerDragProps(index)}
          >
            <th>
              <div className="custom-cnn-layer-cell">
                <span className="custom-cnn-layer-name">
                  <span className="custom-cnn-drag-mark" aria-hidden="true">
                    ::
                  </span>
                  <span>{preview?.name ?? layerLabel(index + 1, layer.kind)}</span>
                </span>
                <select
                  aria-label={`custom CNN layer ${index + 1} type`}
                  disabled={disabled}
                  value={layer.kind}
                  onChange={(event) =>
                    setLayerKind(index, customCnnLayerKind(event.currentTarget.value))
                  }
                >
                  <option value="conv">Conv</option>
                  <option value="residual_post">Res post</option>
                  <option value="residual_pre">Res pre</option>
                  <option value="maxpool">Max pool</option>
                  <option value="avgpool">Avg pool</option>
                  <option value="activation">Activation</option>
                </select>
              </div>
            </th>
            <td>
              <div className="custom-conv-channel-cell">
                <span>{preview === null ? "..." : `${preview.in_channels} ->`}</span>
                {isPoolingLayerKind(layer.kind) || isActivationLayerKind(layer.kind) ? (
                  <span className="custom-conv-static-value">
                    {preview === null ? "same" : preview.out_channels}
                  </span>
                ) : (
                  <IntegerTextInput
                    aria-label={`custom CNN layer ${index + 1} output channels`}
                    className="custom-conv-table-input"
                    disabled={disabled}
                    min={1}
                    value={layer.out_channels}
                    onChange={(nextValue) => setLayerValue(index, "out_channels", nextValue)}
                  />
                )}
              </div>
            </td>
            <td>
              {isActivationLayerKind(layer.kind) ? (
                <span className="custom-conv-static-value">-</span>
              ) : (
                <IntegerTextInput
                  aria-label={`custom CNN layer ${index + 1} kernel size`}
                  className="custom-conv-table-input"
                  disabled={disabled}
                  min={1}
                  value={layer.kernel_size}
                  onChange={(nextValue) => setLayerValue(index, "kernel_size", nextValue)}
                />
              )}
            </td>
            <td>
              {isActivationLayerKind(layer.kind) ? (
                <span className="custom-conv-static-value">-</span>
              ) : (
                <IntegerTextInput
                  aria-label={`custom CNN layer ${index + 1} stride`}
                  className="custom-conv-table-input"
                  disabled={disabled}
                  min={1}
                  value={layer.stride}
                  onChange={(nextValue) => setLayerValue(index, "stride", nextValue)}
                />
              )}
            </td>
            <td>
              {isActivationLayerKind(layer.kind) ? (
                <span className="custom-conv-static-value">-</span>
              ) : (
                <IntegerTextInput
                  aria-label={`custom CNN layer ${index + 1} padding`}
                  className="custom-conv-table-input"
                  min={0}
                  value={layer.padding}
                  disabled={disabled || isResidualLayerKind(layer.kind)}
                  onChange={(nextValue) => setLayerValue(index, "padding", nextValue)}
                />
              )}
            </td>
            <td>
              {layer.kind === "conv" ? (
                <select
                  aria-label={`custom CNN layer ${index + 1} activation`}
                  disabled={disabled}
                  value={layer.post_activation ? "relu" : "none"}
                  onChange={(event) =>
                    setLayerPostActivation(index, event.currentTarget.value === "relu")
                  }
                >
                  <option value="relu">ReLU</option>
                  <option value="none">None</option>
                </select>
              ) : layer.kind === "activation" ? (
                <select
                  aria-label={`custom CNN layer ${index + 1} activation`}
                  disabled={disabled}
                  value={layer.activation ?? "relu"}
                  onChange={(event) =>
                    setLayerActivation(index, customCnnActivation(event.currentTarget.value))
                  }
                >
                  <option value="relu">ReLU</option>
                  <option value="gelu">GELU</option>
                </select>
              ) : (
                <span className="custom-conv-static-value">{layerActivationLabel(layer)}</span>
              )}
            </td>
            <td>{formatConvSpatial(preview, "input")}</td>
            <td>{formatConvSpatial(preview, "output")}</td>
            <td>{formatFitMode(preview)}</td>
            <td>{formatPixelDrop(preview)}</td>
            <td>{preview === null ? "..." : formatParamCount(preview.params)}</td>
            <td className="conv-actions-cell">
              <AppTooltip
                content={
                  disabled
                    ? disabledReason
                    : value.length <= 1
                      ? "Keep at least one layer"
                      : "Remove layer"
                }
              >
                <span className="inline-flex">
                  <button
                    aria-label={`Remove custom CNN layer ${index + 1}`}
                    className="field-reset-button"
                    disabled={disabled || value.length <= 1}
                    type="button"
                    onClick={() => removeLayer(index)}
                  >
                    <RemoveLayerIcon />
                  </button>
                </span>
              </AppTooltip>
            </td>
          </tr>
        );
      })}
      <tr className="custom-conv-add-row">
        <th>{`L${value.length + 1}`}</th>
        <td colSpan={10}>
          <div className="custom-cnn-add-actions">
            {customCnnLayerOptions.map((option) => (
              <button
                className="secondary-button custom-conv-add-row-button"
                disabled={disabled}
                key={option.label}
                type="button"
                onClick={() =>
                  option.kind === "conv" ? duplicateLastConvLayer() : addLayer(option.kind)
                }
              >
                {option.label}
              </button>
            ))}
          </div>
        </td>
        <td className="conv-actions-cell">
          <AppTooltip content={disabled ? disabledReason : "Add layer"}>
            <span className="inline-flex">
              <button
                aria-label="Add custom CNN conv layer"
                className="field-reset-button layer-add-button"
                disabled={disabled}
                type="button"
                onClick={duplicateLastConvLayer}
              >
                <AddLayerIcon />
              </button>
            </span>
          </AppTooltip>
        </td>
      </tr>
      <tr className="custom-conv-flatten-row">
        <th>flatten</th>
        <td colSpan={11}>{formatFlattenSummary(previewLayers, flattenDim)}</td>
      </tr>
    </>
  );
}

const customCnnLayerOptions: readonly { kind: CustomCnnLayerKind; label: string }[] = [
  { kind: "conv", label: "Add conv layer" },
  { kind: "residual_post", label: "Add res post" },
  { kind: "residual_pre", label: "Add res pre" },
  { kind: "maxpool", label: "Add max pool" },
  { kind: "avgpool", label: "Add avg pool" },
  { kind: "activation", label: "Add activation" },
];
