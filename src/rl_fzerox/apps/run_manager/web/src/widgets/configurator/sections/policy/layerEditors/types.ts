// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/policy/layerEditors/types.ts
import type { ManagedRunConfig } from "@/shared/api/contract";

export type CustomConvLayer = ManagedRunConfig["policy"]["custom_conv_layers"][number];
export type CustomCnnLayerKind = CustomConvLayer["kind"];
export type CustomCnnActivation = NonNullable<CustomConvLayer["activation"]>;
export type CustomCnnNumericKey = "kernel_size" | "out_channels" | "padding" | "stride";
