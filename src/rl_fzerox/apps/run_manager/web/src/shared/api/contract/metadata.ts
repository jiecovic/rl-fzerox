// src/rl_fzerox/apps/run_manager/web/src/shared/api/contract/metadata.ts
import { z } from "zod";

import {
  auxiliaryStateTargetNameSchema,
  observationPresetSchema,
  rendererSchema,
  stateComponentNameSchema,
} from "@/shared/api/contract/enums";

const selectOptionSchema = z.object({
  value: z.string(),
  label: z.string(),
});

const observationPresetInfoSchema = z.object({
  value: observationPresetSchema,
  label: z.string(),
  height: z.number().int().positive(),
  width: z.number().int().positive(),
});

const observationResolutionBoundsSchema = z.object({
  min_dimension: z.number().int().positive(),
  max_height: z.number().int().positive(),
  max_width: z.number().int().positive(),
});

const observationSourceGeometryInfoSchema = z.object({
  renderer: rendererSchema,
  height: z.number().int().positive(),
  width: z.number().int().positive(),
});

const trackCupInfoSchema = z.object({
  id: z.string(),
  label: z.string(),
  order: z.number().int().nonnegative(),
  course_ids: z.array(z.string()),
});

const builtInCourseInfoSchema = z.object({
  id: z.string(),
  ref: z.string(),
  display_name: z.string(),
  cup: z.string(),
  cup_label: z.string(),
  course_index: z.number().int().nonnegative(),
  default_selected: z.boolean(),
});

const vehicleInfoSchema = z.object({
  id: z.string(),
  display_name: z.string(),
  character_index: z.number().int().nonnegative(),
  machine_select_slot: z.number().int().nonnegative(),
  menu_row: z.number().int().nonnegative(),
  menu_column: z.number().int().nonnegative(),
});

const engineSettingPresetInfoSchema = z.object({
  id: z.string(),
  display_name: z.string(),
  raw_value: z.number().int().min(0).max(100),
});

const stateComponentInfoSchema = z.object({
  name: z.string(),
  low: z.number(),
  high: z.number(),
  default_enabled: z.boolean().default(true),
  auxiliary_target_name: auxiliaryStateTargetNameSchema.nullable().default(null),
  auxiliary_supports_grounded_only: z.boolean().default(false),
});

const stateComponentSchema = z.object({
  name: stateComponentNameSchema,
  label: z.string(),
  features: z.array(stateComponentInfoSchema),
});

export const configMetadataSchema = z.object({
  observation_presets: z.array(observationPresetInfoSchema),
  observation_resolution_bounds: observationResolutionBoundsSchema,
  observation_source_geometries: z.array(observationSourceGeometryInfoSchema),
  camera_settings: z.array(selectOptionSchema),
  race_modes: z.array(selectOptionSchema),
  gp_difficulties: z.array(selectOptionSchema),
  track_sampling_modes: z.array(selectOptionSchema),
  track_cups: z.array(trackCupInfoSchema),
  built_in_courses: z.array(builtInCourseInfoSchema),
  vehicles: z.array(vehicleInfoSchema),
  engine_setting_presets: z.array(engineSettingPresetInfoSchema),
  steering_modes: z.array(selectOptionSchema),
  drive_modes: z.array(selectOptionSchema),
  lean_output_modes: z.array(selectOptionSchema),
  lean_modes: z.array(selectOptionSchema),
  stack_modes: z.array(selectOptionSchema),
  resize_filters: z.array(selectOptionSchema),
  progress_sources: z.array(selectOptionSchema),
  action_history_controls: z.array(selectOptionSchema),
  state_components: z.array(stateComponentSchema),
  conv_profiles: z.array(selectOptionSchema),
  activation_functions: z.array(selectOptionSchema).default([
    { value: "relu", label: "relu" },
    { value: "gelu", label: "gelu" },
    { value: "tanh", label: "tanh" },
  ]),
  net_arch_presets: z.array(selectOptionSchema),
});

export type ConfigMetadata = z.infer<typeof configMetadataSchema>;
