// web/run-manager/src/entities/runConfig/ui/sections/policy/layerEditors/customCnnLayers.ts
import type {
  CustomCnnActivation,
  CustomCnnLayerKind,
  CustomCnnNumericKey,
  CustomConvLayer,
} from "@/entities/runConfig/ui/sections/policy/layerEditors/types";

export function customCnnLayerKind(value: string): CustomCnnLayerKind {
  if (value === "residual") {
    return "residual_post";
  }
  if (
    value === "activation" ||
    value === "avgpool" ||
    value === "maxpool" ||
    value === "residual_post" ||
    value === "residual_pre"
  ) {
    return value;
  }
  return "conv";
}

export function customCnnActivation(value: string): CustomCnnActivation {
  return value === "gelu" ? "gelu" : "relu";
}

export function layerActivationLabel(layer: CustomConvLayer) {
  if (layer.kind === "activation") {
    return layer.activation ?? "relu";
  }
  if (layer.kind === "residual_pre") {
    return "pre";
  }
  if (layer.kind === "residual_post") {
    return "post";
  }
  return "none";
}

export function layerLabel(index: number, kind: CustomCnnLayerKind) {
  if (kind === "residual_pre") {
    return `res-pre${index}`;
  }
  if (kind === "residual_post") {
    return `res${index}`;
  }
  if (kind === "maxpool") {
    return `pool${index}`;
  }
  if (kind === "avgpool") {
    return `avgpool${index}`;
  }
  if (kind === "activation") {
    return `act${index}`;
  }
  return `conv${index}`;
}

export function withCustomCnnNumericValue(
  layer: CustomConvLayer,
  key: CustomCnnNumericKey,
  nextValue: number,
) {
  const minimum = key === "padding" ? 0 : 1;
  if (!Number.isSafeInteger(nextValue) || nextValue < minimum) {
    return null;
  }
  const nextLayer = { ...layer, [key]: nextValue };
  if (isResidualLayerKind(nextLayer.kind) && key === "kernel_size") {
    if (nextValue % 2 === 0) {
      return null;
    }
    return { ...nextLayer, padding: residualPadding(nextValue) };
  }
  return nextLayer;
}

export function withCustomCnnKind(layer: CustomConvLayer, kind: CustomCnnLayerKind) {
  if (isResidualLayerKind(kind)) {
    return normalizedResidualLayer(layer, kind);
  }
  if (isPoolingLayerKind(kind)) {
    return normalizedPoolLayer(layer, kind);
  }
  if (isActivationLayerKind(kind)) {
    return normalizedActivationLayer(layer);
  }
  return withoutStandaloneActivation({ ...layer, kind });
}

export function withCustomCnnActivation(layer: CustomConvLayer, activation: CustomCnnActivation) {
  return { ...normalizedActivationLayer(layer), activation };
}

export function createCustomCnnLayer(kind: CustomCnnLayerKind, previous?: CustomConvLayer) {
  const template: CustomConvLayer = {
    kind,
    out_channels: previous?.out_channels ?? 64,
    kernel_size: isActivationLayerKind(kind)
      ? 1
      : isResidualLayerKind(kind)
        ? 3
        : isPoolingLayerKind(kind)
          ? 2
          : (previous?.kernel_size ?? 2),
    stride: isActivationLayerKind(kind)
      ? 1
      : isResidualLayerKind(kind)
        ? 1
        : isPoolingLayerKind(kind)
          ? 2
          : (previous?.stride ?? 1),
    padding: isActivationLayerKind(kind)
      ? 0
      : isResidualLayerKind(kind)
        ? 1
        : isPoolingLayerKind(kind)
          ? 0
          : (previous?.padding ?? 0),
    post_activation: true,
    ...(isActivationLayerKind(kind) ? { activation: "relu" as const } : {}),
  };
  return withCustomCnnKind(template, kind);
}

export function duplicateConvLayerFrom(previous?: CustomConvLayer) {
  const template: CustomConvLayer = previous ?? {
    kind: "conv",
    out_channels: 64,
    kernel_size: 3,
    stride: 1,
    padding: 0,
    post_activation: true,
  };
  return { ...withoutStandaloneActivation(template), kind: "conv" as const };
}

export function isResidualLayerKind(
  kind: CustomCnnLayerKind,
): kind is Extract<CustomCnnLayerKind, "residual_post" | "residual_pre"> {
  return kind === "residual_pre" || kind === "residual_post";
}

export function isPoolingLayerKind(
  kind: CustomCnnLayerKind,
): kind is Extract<CustomCnnLayerKind, "avgpool" | "maxpool"> {
  return kind === "maxpool" || kind === "avgpool";
}

export function isActivationLayerKind(
  kind: CustomCnnLayerKind,
): kind is Extract<CustomCnnLayerKind, "activation"> {
  return kind === "activation";
}

function normalizedResidualLayer(
  layer: CustomConvLayer,
  kind: Extract<CustomCnnLayerKind, "residual_post" | "residual_pre">,
): CustomConvLayer {
  const kernelSize =
    layer.kernel_size % 2 === 1 ? layer.kernel_size : Math.max(1, layer.kernel_size - 1);
  return {
    ...withoutStandaloneActivation(layer),
    kind,
    kernel_size: kernelSize,
    padding: residualPadding(kernelSize),
  };
}

function normalizedPoolLayer(
  layer: CustomConvLayer,
  kind: Extract<CustomCnnLayerKind, "avgpool" | "maxpool">,
): CustomConvLayer {
  return {
    ...withoutStandaloneActivation(layer),
    kind,
    kernel_size: Math.max(1, layer.kernel_size),
    stride: Math.max(1, layer.stride),
    padding: Math.max(0, layer.padding),
  };
}

function normalizedActivationLayer(layer: CustomConvLayer): CustomConvLayer {
  return {
    ...layer,
    kind: "activation",
    kernel_size: 1,
    stride: 1,
    padding: 0,
    post_activation: true,
    activation: layer.activation ?? "relu",
  };
}

function withoutStandaloneActivation(layer: CustomConvLayer): CustomConvLayer {
  const { activation: _activation, ...rest } = layer;
  return rest;
}

function residualPadding(kernelSize: number) {
  return Math.floor(kernelSize / 2);
}
