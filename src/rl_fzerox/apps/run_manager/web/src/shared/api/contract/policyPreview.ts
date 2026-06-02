// src/rl_fzerox/apps/run_manager/web/src/shared/api/contract/policyPreview.ts
import { z } from "zod";

import { customCnnLayerKindSchema } from "@/shared/api/contract/config";
import { customCnnActivationSchema, stateComponentNameSchema } from "@/shared/api/contract/enums";

const shapePreviewSchema = z.object({
  height: z.number().int().positive(),
  width: z.number().int().positive(),
  channels: z.number().int().positive(),
});

const stateFeaturePreviewSchema = z.object({
  component: stateComponentNameSchema,
  name: z.string(),
  dropout_prob: z.number().min(0).max(1),
});

const convLayerPreviewSchema = z.object({
  name: z.string(),
  kind: customCnnLayerKindSchema,
  in_channels: z.number().int().nonnegative(),
  out_channels: z.number().int().positive(),
  kernel_size: z.number().int().positive(),
  stride: z.number().int().positive(),
  padding: z.number().int().nonnegative(),
  post_activation: z.boolean(),
  activation: customCnnActivationSchema.nullable().optional(),
  input_height: z.number().int().positive(),
  input_width: z.number().int().positive(),
  output_height: z.number().int().positive(),
  output_width: z.number().int().positive(),
  dropped_height: z.number().int().nonnegative(),
  dropped_width: z.number().int().nonnegative(),
  params: z.number().int().nonnegative(),
});

const parameterGroupPreviewSchema = z.object({
  name: z.string(),
  params: z.number().int().nonnegative(),
});

const actionBranchPreviewSchema = z.object({
  name: z.string(),
  kind: z.string(),
  size: z.number().int().positive(),
  enabled: z.boolean(),
  mask_label: z.string().nullable().optional(),
});

const architectureNodePreviewSchema = z.object({
  id: z.string(),
  label: z.string(),
  detail: z.string(),
  params: z.number().int().nonnegative().nullable().optional(),
  tone: z.string(),
});

const architectureLanePreviewSchema = z.object({
  id: z.string(),
  label: z.string(),
  nodes: z.array(architectureNodePreviewSchema),
});

export const policyArchitecturePreviewSchema = z.object({
  image_shape: shapePreviewSchema,
  state_dim: z.number().int().nonnegative(),
  state_features: z.array(stateFeaturePreviewSchema),
  conv_layers: z.array(convLayerPreviewSchema),
  flatten_dim: z.number().int().nonnegative(),
  image_features_dim: z.number().int().nonnegative(),
  state_features_dim: z.number().int().nonnegative(),
  fusion_input_dim: z.number().int().nonnegative(),
  extractor_output_dim: z.number().int().nonnegative(),
  policy_input_dim: z.number().int().nonnegative(),
  action_branches: z.array(actionBranchPreviewSchema),
  continuous_action_dims: z.number().int().nonnegative(),
  discrete_action_logits: z.number().int().nonnegative(),
  parameter_groups: z.array(parameterGroupPreviewSchema),
  total_params: z.number().int().nonnegative(),
  architecture_lanes: z.array(architectureLanePreviewSchema),
});

export type PolicyArchitecturePreview = z.infer<typeof policyArchitecturePreviewSchema>;
