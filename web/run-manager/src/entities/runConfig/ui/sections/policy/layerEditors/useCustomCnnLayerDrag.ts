// web/run-manager/src/entities/runConfig/ui/sections/policy/layerEditors/useCustomCnnLayerDrag.ts
import type { ComponentProps, DragEvent } from "react";
import { useState } from "react";

import type { ManagedRunConfig } from "@/shared/api/contract";

type CustomConvLayers = ManagedRunConfig["policy"]["custom_conv_layers"];

interface CustomCnnLayerDragProps {
  disabled: boolean;
  value: CustomConvLayers;
  onChange: (value: CustomConvLayers) => void;
}

export function useCustomCnnLayerDrag({ disabled, value, onChange }: CustomCnnLayerDragProps) {
  const [draggedLayerIndex, setDraggedLayerIndex] = useState<number | null>(null);
  const [dragOverLayerIndex, setDragOverLayerIndex] = useState<number | null>(null);

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

  function layerDragProps(
    index: number,
  ): Pick<
    ComponentProps<"tr">,
    "draggable" | "onDragEnd" | "onDragLeave" | "onDragOver" | "onDragStart" | "onDrop"
  > {
    return {
      draggable: !disabled,
      onDragEnd: () => {
        setDraggedLayerIndex(null);
        setDragOverLayerIndex(null);
      },
      onDragLeave: () => setDragOverLayerIndex(null),
      onDragOver: (event) => allowLayerDrop(event, index),
      onDragStart: (event) => beginLayerDrag(event, index),
      onDrop: (event) => completeLayerDrop(event, index),
    };
  }

  return {
    draggedLayerIndex,
    dragOverLayerIndex,
    layerDragProps,
  };
}

export function customCnnRowClass(
  index: number,
  draggedLayerIndex: number | null,
  dragOverLayerIndex: number | null,
  disabled: boolean,
) {
  const classes = ["custom-cnn-layer-row"];
  if (disabled) {
    classes.push("is-disabled");
  }
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
    target instanceof HTMLButtonElement ||
    target instanceof HTMLInputElement ||
    target instanceof HTMLSelectElement
  );
}
