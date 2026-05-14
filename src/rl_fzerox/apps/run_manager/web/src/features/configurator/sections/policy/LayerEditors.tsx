// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/policy/LayerEditors.tsx
import { type DragEvent, useRef, useState } from "react";
import { FieldLabel } from "@/features/configurator/fields/label";
import {
  formatConvSpatial,
  formatFitMode,
  formatFlattenSummary,
  formatParamCount,
  formatPixelDrop,
} from "@/features/configurator/sections/policy/convPreviewFormatting";
import type { ManagedRunConfig, PolicyArchitecturePreview } from "@/shared/api/contract";
import { AddLayerIcon, RemoveLayerIcon } from "@/shared/ui/icons";

type CustomConvLayer = ManagedRunConfig["policy"]["custom_conv_layers"][number];
type CustomCnnLayerKind = CustomConvLayer["kind"];
type CustomCnnNumericKey = "out_channels" | "kernel_size" | "stride" | "padding";

export function LayerListField({
  help,
  label,
  resetValue,
  value,
  onChange,
}: {
  help: string;
  label: string;
  resetValue: number[];
  value: number[];
  onChange: (value: number[]) => void;
}) {
  const rowIdsRef = useRef<string[]>([]);
  syncLayerRowIds(rowIdsRef.current, value.length, label);

  function setLayer(index: number, nextValue: number) {
    if (!Number.isSafeInteger(nextValue) || nextValue <= 0) {
      return;
    }
    onChange(value.map((layer, layerIndex) => (layerIndex === index ? nextValue : layer)));
  }

  function addLayer() {
    onChange([...value, value.at(-1) ?? 256]);
  }

  function removeLayer(index: number) {
    onChange(value.filter((_, layerIndex) => layerIndex !== index));
  }

  return (
    <div className="field-shell layer-list-field">
      <FieldLabel
        help={help}
        label={label}
        onReset={layerListResetHandler(value, resetValue, onChange)}
      />
      <div className="layer-list-editor">
        {value.length === 0 ? <span className="layer-list-empty">No hidden layers</span> : null}
        {value.map((layer, index) => (
          <div className="layer-list-row" key={rowIdsRef.current[index]}>
            <span className="layer-index">L{index + 1}</span>
            <input
              aria-label={`${label} layer ${index + 1}`}
              min={1}
              step={1}
              type="number"
              value={layer}
              onChange={(event) => setLayer(index, Number(event.target.value))}
            />
            <button
              aria-label={`Remove ${label} layer ${index + 1}`}
              className="field-reset-button tooltip-anchor"
              data-tooltip="Remove layer"
              type="button"
              onClick={() => removeLayer(index)}
            >
              <RemoveLayerIcon />
            </button>
          </div>
        ))}
        <div className="layer-list-row layer-add-row">
          <span className="layer-index">L{value.length + 1}</span>
          <button className="layer-add-placeholder" type="button" onClick={addLayer}>
            Add layer
          </button>
          <button
            aria-label={`Add ${label} layer`}
            className="field-reset-button layer-add-button tooltip-anchor"
            data-tooltip="Add layer"
            type="button"
            onClick={addLayer}
          >
            <AddLayerIcon />
          </button>
        </div>
      </div>
    </div>
  );
}

export function CustomConvTableRows({
  disabled = false,
  flattenDim,
  previewLayers,
  value,
  onChange,
}: {
  disabled?: boolean;
  flattenDim: number;
  previewLayers: PolicyArchitecturePreview["conv_layers"];
  value: ManagedRunConfig["policy"]["custom_conv_layers"];
  onChange: (value: ManagedRunConfig["policy"]["custom_conv_layers"]) => void;
}) {
  const rowIdsRef = useRef<string[]>([]);
  const [draggedLayerIndex, setDraggedLayerIndex] = useState<number | null>(null);
  const [dragOverLayerIndex, setDragOverLayerIndex] = useState<number | null>(null);
  syncLayerRowIds(rowIdsRef.current, value.length, "custom-conv");

  function setLayerValue(index: number, key: CustomCnnNumericKey, nextValue: number) {
    const minimum = key === "padding" ? 0 : 1;
    if (!Number.isSafeInteger(nextValue) || nextValue < minimum) {
      return;
    }
    onChange(
      value.map((layer, layerIndex) => {
        if (layerIndex !== index) {
          return layer;
        }
        const nextLayer = { ...layer, [key]: nextValue };
        if (nextLayer.kind === "residual" && key === "kernel_size") {
          if (nextValue % 2 === 0) {
            return layer;
          }
          return { ...nextLayer, padding: residualPadding(nextValue) };
        }
        return nextLayer;
      }),
    );
  }

  function setLayerKind(index: number, kind: CustomCnnLayerKind) {
    onChange(
      value.map((layer, layerIndex) => {
        if (layerIndex !== index) {
          return layer;
        }
        if (kind === "residual") {
          return normalizedResidualLayer(layer);
        }
        return { ...layer, kind };
      }),
    );
  }

  function addLayer(kind: CustomCnnLayerKind) {
    const previous = value.at(-1);
    const template: CustomConvLayer = {
      kind,
      out_channels: previous?.out_channels ?? 64,
      kernel_size: kind === "residual" ? 3 : (previous?.kernel_size ?? 3),
      stride: kind === "residual" ? 1 : (previous?.stride ?? 1),
      padding: kind === "residual" ? 1 : (previous?.padding ?? 0),
    };
    const nextLayer: CustomConvLayer =
      kind === "residual"
        ? normalizedResidualLayer(template)
        : {
            ...template,
            kind: "conv",
          };
    onChange([...value, nextLayer]);
  }

  function duplicateLastConvLayer() {
    const template: CustomConvLayer = value.at(-1) ?? {
      kind: "conv",
      out_channels: 64,
      kernel_size: 3,
      stride: 1,
      padding: 0,
    };
    onChange([...value, { ...template, kind: "conv" }]);
  }

  function removeLayer(index: number) {
    if (value.length <= 1) {
      return;
    }
    onChange(value.filter((_, layerIndex) => layerIndex !== index));
  }

  function moveLayer(fromIndex: number, toIndex: number) {
    if (disabled || fromIndex === toIndex || !value[fromIndex] || !value[toIndex]) {
      return;
    }
    const nextLayers = [...value];
    const [movedLayer] = nextLayers.splice(fromIndex, 1);
    nextLayers.splice(toIndex, 0, movedLayer);
    onChange(nextLayers);
  }

  function beginLayerDrag(event: DragEvent<HTMLTableRowElement>, index: number) {
    if (disabled || isInteractiveDragTarget(event.target)) {
      event.preventDefault();
      return;
    }
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", String(index));
    setDraggedLayerIndex(index);
  }

  function allowLayerDrop(event: DragEvent<HTMLTableRowElement>, index: number) {
    if (disabled || draggedLayerIndex === null || draggedLayerIndex === index) {
      return;
    }
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    setDragOverLayerIndex(index);
  }

  function completeLayerDrop(event: DragEvent<HTMLTableRowElement>, index: number) {
    event.preventDefault();
    const fromIndex = Number(event.dataTransfer.getData("text/plain"));
    if (Number.isSafeInteger(fromIndex)) {
      moveLayer(fromIndex, index);
    }
    setDraggedLayerIndex(null);
    setDragOverLayerIndex(null);
  }

  return (
    <>
      {value.map((layer, index) => {
        const preview = previewLayers[index] ?? null;
        return (
          <tr
            className={customCnnRowClass(index, draggedLayerIndex, dragOverLayerIndex)}
            draggable={!disabled}
            key={rowIdsRef.current[index]}
            onDragEnd={() => {
              setDraggedLayerIndex(null);
              setDragOverLayerIndex(null);
            }}
            onDragLeave={() => setDragOverLayerIndex(null)}
            onDragOver={(event) => allowLayerDrop(event, index)}
            onDragStart={(event) => beginLayerDrag(event, index)}
            onDrop={(event) => completeLayerDrop(event, index)}
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
                  <option value="residual">Res block</option>
                </select>
              </div>
            </th>
            <td>
              <div className="custom-conv-channel-cell">
                <span>{preview === null ? "…" : `${preview.in_channels} →`}</span>
                <input
                  aria-label={`custom CNN layer ${index + 1} output channels`}
                  className="custom-conv-table-input"
                  disabled={disabled}
                  min={1}
                  step={1}
                  type="number"
                  value={layer.out_channels}
                  onChange={(event) =>
                    setLayerValue(index, "out_channels", Number(event.target.value))
                  }
                />
              </div>
            </td>
            <td>
              <input
                aria-label={`custom CNN layer ${index + 1} kernel size`}
                className="custom-conv-table-input"
                disabled={disabled}
                min={1}
                step={1}
                type="number"
                value={layer.kernel_size}
                onChange={(event) =>
                  setLayerValue(index, "kernel_size", Number(event.target.value))
                }
              />
            </td>
            <td>
              <input
                aria-label={`custom CNN layer ${index + 1} stride`}
                className="custom-conv-table-input"
                disabled={disabled}
                min={1}
                step={1}
                type="number"
                value={layer.stride}
                onChange={(event) => setLayerValue(index, "stride", Number(event.target.value))}
              />
            </td>
            <td>
              <input
                aria-label={`custom CNN layer ${index + 1} padding`}
                className="custom-conv-table-input"
                min={0}
                step={1}
                type="number"
                value={layer.padding}
                disabled={disabled || layer.kind === "residual"}
                onChange={(event) => setLayerValue(index, "padding", Number(event.target.value))}
              />
            </td>
            <td>{formatConvSpatial(preview, "input")}</td>
            <td>{formatConvSpatial(preview, "output")}</td>
            <td>{formatFitMode(preview)}</td>
            <td>{formatPixelDrop(preview)}</td>
            <td>{preview === null ? "…" : formatParamCount(preview.params)}</td>
            <td className="conv-actions-cell">
              <button
                aria-label={`Remove custom CNN layer ${index + 1}`}
                className="field-reset-button tooltip-anchor"
                data-tooltip={
                  disabled
                    ? "Forked checkpoints keep the original CNN extractor."
                    : value.length <= 1
                      ? "Keep at least one layer"
                      : "Remove layer"
                }
                disabled={disabled || value.length <= 1}
                type="button"
                onClick={() => removeLayer(index)}
              >
                <RemoveLayerIcon />
              </button>
            </td>
          </tr>
        );
      })}
      <tr className="custom-conv-add-row">
        <th>{`L${value.length + 1}`}</th>
        <td colSpan={9}>
          <div className="custom-cnn-add-actions">
            <button
              className="secondary-button custom-conv-add-row-button"
              disabled={disabled}
              type="button"
              onClick={duplicateLastConvLayer}
            >
              Add conv layer
            </button>
            <button
              className="secondary-button custom-conv-add-row-button"
              disabled={disabled}
              type="button"
              onClick={() => addLayer("residual")}
            >
              Add res block
            </button>
          </div>
        </td>
        <td className="conv-actions-cell">
          <button
            aria-label="Add custom CNN conv layer"
            className="field-reset-button layer-add-button tooltip-anchor"
            data-tooltip={
              disabled ? "Forked checkpoints keep the original CNN extractor." : "Add layer"
            }
            disabled={disabled}
            type="button"
            onClick={duplicateLastConvLayer}
          >
            <AddLayerIcon />
          </button>
        </td>
      </tr>
      <tr className="custom-conv-flatten-row">
        <th>flatten</th>
        <td colSpan={10}>{formatFlattenSummary(previewLayers, flattenDim)}</td>
      </tr>
    </>
  );
}

function layerListResetHandler<T>(value: T[], resetValue: T[], onChange: (value: T[]) => void) {
  if (sameLayerList(value, resetValue)) {
    return undefined;
  }
  return () => onChange(resetValue);
}

function sameLayerList<T>(left: T[], right: T[]) {
  return (
    left.length === right.length && left.every((value, index) => deepEqual(value, right[index]))
  );
}

function deepEqual(left: unknown, right: unknown) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function normalizedResidualLayer(layer: CustomConvLayer): CustomConvLayer {
  const kernelSize =
    layer.kernel_size % 2 === 1 ? layer.kernel_size : Math.max(1, layer.kernel_size - 1);
  return {
    ...layer,
    kind: "residual",
    kernel_size: kernelSize,
    padding: residualPadding(kernelSize),
  };
}

function residualPadding(kernelSize: number) {
  return Math.floor(kernelSize / 2);
}

function layerLabel(index: number, kind: CustomCnnLayerKind) {
  return kind === "residual" ? `res${index}` : `conv${index}`;
}

function customCnnLayerKind(value: string): CustomCnnLayerKind {
  return value === "residual" ? "residual" : "conv";
}

function customCnnRowClass(
  index: number,
  draggedLayerIndex: number | null,
  dragOverLayerIndex: number | null,
) {
  const classes = ["custom-cnn-layer-row"];
  if (draggedLayerIndex === index) {
    classes.push("is-dragging");
  }
  if (dragOverLayerIndex === index && draggedLayerIndex !== index) {
    classes.push("is-drop-target");
  }
  return classes.join(" ");
}

function isInteractiveDragTarget(target: EventTarget | null) {
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLSelectElement ||
    target instanceof HTMLButtonElement
  );
}

function syncLayerRowIds(rowIds: string[], length: number, label: string) {
  while (rowIds.length < length) {
    rowIds.push(`${label}-${crypto.randomUUID()}`);
  }
  rowIds.length = length;
}
