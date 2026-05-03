import { useRef } from "react";
import { FieldLabel } from "@/features/configurator/fields/label";
import {
  formatConvSpatial,
  formatFitMode,
  formatFlattenSummary,
  formatParamCount,
  formatPixelDrop,
} from "@/features/configurator/sections/policy/convPreviewFormatting";
import type { ManagedRunConfig, PolicyArchitecturePreview } from "@/shared/api/contract";

type CustomConvLayer = ManagedRunConfig["policy"]["custom_conv_layers"][number];

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
  flattenDim,
  previewLayers,
  value,
  onChange,
}: {
  flattenDim: number;
  previewLayers: PolicyArchitecturePreview["conv_layers"];
  value: ManagedRunConfig["policy"]["custom_conv_layers"];
  onChange: (value: ManagedRunConfig["policy"]["custom_conv_layers"]) => void;
}) {
  const rowIdsRef = useRef<string[]>([]);
  syncLayerRowIds(rowIdsRef.current, value.length, "custom-conv");

  function setLayerValue<K extends keyof CustomConvLayer>(
    index: number,
    key: K,
    nextValue: number,
  ) {
    const minimum = key === "padding" ? 0 : 1;
    if (!Number.isSafeInteger(nextValue) || nextValue < minimum) {
      return;
    }
    onChange(
      value.map((layer, layerIndex) =>
        layerIndex === index ? { ...layer, [key]: nextValue } : layer,
      ),
    );
  }

  function addLayer() {
    const template = value.at(-1) ?? {
      out_channels: 64,
      kernel_size: 3,
      stride: 1,
      padding: 0,
    };
    onChange([...value, { ...template }]);
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
          <tr key={rowIdsRef.current[index]}>
            <th>{preview?.name ?? `conv${index + 1}`}</th>
            <td>
              <div className="custom-conv-channel-cell">
                <span>{preview === null ? "…" : `${preview.in_channels} →`}</span>
                <input
                  aria-label={`custom conv ${index + 1} output channels`}
                  className="custom-conv-table-input"
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
                aria-label={`custom conv ${index + 1} kernel size`}
                className="custom-conv-table-input"
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
                aria-label={`custom conv ${index + 1} stride`}
                className="custom-conv-table-input"
                min={1}
                step={1}
                type="number"
                value={layer.stride}
                onChange={(event) => setLayerValue(index, "stride", Number(event.target.value))}
              />
            </td>
            <td>
              <input
                aria-label={`custom conv ${index + 1} padding`}
                className="custom-conv-table-input"
                min={0}
                step={1}
                type="number"
                value={layer.padding}
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
                aria-label={`Remove custom conv layer ${index + 1}`}
                className="field-reset-button tooltip-anchor"
                data-tooltip={value.length <= 1 ? "Keep at least one layer" : "Remove layer"}
                disabled={value.length <= 1}
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
        <th>{`conv${value.length + 1}`}</th>
        <td colSpan={9}>
          <button
            className="secondary-button custom-conv-add-row-button"
            type="button"
            onClick={addLayer}
          >
            Add conv layer
          </button>
        </td>
        <td className="conv-actions-cell">
          <button
            aria-label="Add custom conv layer"
            className="field-reset-button layer-add-button tooltip-anchor"
            data-tooltip="Add layer"
            type="button"
            onClick={addLayer}
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

function syncLayerRowIds(rowIds: string[], length: number, label: string) {
  while (rowIds.length < length) {
    rowIds.push(`${label}-${crypto.randomUUID()}`);
  }
  rowIds.length = length;
}

function AddLayerIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="14" viewBox="0 0 20 20" width="14">
      <path d="M10 4v12M4 10h12" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </svg>
  );
}

function RemoveLayerIcon() {
  return (
    <svg aria-hidden="true" fill="none" height="12" viewBox="0 0 20 20" width="12">
      <path d="M5 10h10" stroke="currentColor" strokeLinecap="round" strokeWidth="1.8" />
    </svg>
  );
}
