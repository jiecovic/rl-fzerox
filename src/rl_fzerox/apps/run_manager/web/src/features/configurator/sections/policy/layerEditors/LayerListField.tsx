// src/rl_fzerox/apps/run_manager/web/src/features/configurator/sections/policy/layerEditors/LayerListField.tsx
import { useRef } from "react";

import { IntegerTextInput } from "@/features/configurator/fields/basicNumbers";
import { FieldLabel } from "@/features/configurator/fields/label";
import {
  layerListResetHandler,
  syncLayerRowIds,
} from "@/features/configurator/sections/policy/layerEditors/layerRows";
import { AddLayerIcon, RemoveLayerIcon } from "@/shared/ui/icons";
import { AppTooltip } from "@/shared/ui/Tooltip";

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
            <IntegerTextInput
              aria-label={`${label} layer ${index + 1}`}
              min={1}
              value={layer}
              onChange={(nextValue) => setLayer(index, nextValue)}
            />
            <AppTooltip content="Remove layer">
              <button
                aria-label={`Remove ${label} layer ${index + 1}`}
                className="field-reset-button"
                type="button"
                onClick={() => removeLayer(index)}
              >
                <RemoveLayerIcon />
              </button>
            </AppTooltip>
          </div>
        ))}
        <div className="layer-list-row layer-add-row">
          <span className="layer-index">L{value.length + 1}</span>
          <button className="layer-add-placeholder" type="button" onClick={addLayer}>
            Add layer
          </button>
          <AppTooltip content="Add layer">
            <button
              aria-label={`Add ${label} layer`}
              className="field-reset-button layer-add-button"
              type="button"
              onClick={addLayer}
            >
              <AddLayerIcon />
            </button>
          </AppTooltip>
        </div>
      </div>
    </div>
  );
}
