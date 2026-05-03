import type { PolicyArchitecturePreview } from "@/shared/api/contract";

type ConvLayerPreview = PolicyArchitecturePreview["conv_layers"][number];

export function formatConvSpatial(layer: ConvLayerPreview | null, axis: "input" | "output") {
  if (layer === null) {
    return "…";
  }
  if (axis === "input") {
    return `${layer.input_height} x ${layer.input_width}`;
  }
  return `${layer.output_height} x ${layer.output_width}`;
}

export function formatFitMode(layer: ConvLayerPreview | null) {
  if (layer === null) {
    return "…";
  }
  return layer.dropped_height === 0 && layer.dropped_width === 0 ? "exact" : "floored";
}

export function formatPixelDrop(layer: ConvLayerPreview | null) {
  if (layer === null) {
    return "…";
  }
  const droppedParts = [
    pluralizedCount(layer.dropped_height, "row"),
    pluralizedCount(layer.dropped_width, "col"),
  ].filter((part) => part !== null);
  return droppedParts.length === 0 ? "none" : droppedParts.join(", ");
}

export function formatFlattenSummary(
  layers: PolicyArchitecturePreview["conv_layers"],
  flattenDim: number,
) {
  const lastLayer = layers.at(-1);
  if (lastLayer === undefined) {
    return flattenDim.toLocaleString();
  }
  return `${lastLayer.out_channels} x ${lastLayer.output_height} x ${lastLayer.output_width} = ${flattenDim.toLocaleString()}`;
}

export function formatParamCount(value: number) {
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

function pluralizedCount(value: number, singular: string) {
  if (value === 0) {
    return null;
  }
  return `${value} ${singular}${value === 1 ? "" : "s"}`;
}
