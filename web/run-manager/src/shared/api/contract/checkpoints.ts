// web/run-manager/src/shared/api/contract/checkpoints.ts
import { z } from "zod";

import { managedRunConfigSchema } from "@/shared/api/contract/config";
import { evaluationResultSummarySchema } from "@/shared/api/contract/evaluations";
import {
  engineTuningRuntimeStateSchema,
  managedRunSummarySchema,
} from "@/shared/api/contract/runs";

const checkpointBundleFileSchema = z.object({
  path: z.string(),
  role: z.string(),
  sha256: z.string(),
  size_bytes: z.number().int().nonnegative(),
});

const checkpointBundleManifestSchema = z
  .object({
    checkpoint: z
      .object({
        id: z.string(),
        name: z.string(),
        version: z.string(),
        source_artifact: z.string(),
        local_num_timesteps: z.number().int().nonnegative().nullable(),
        lineage_num_timesteps: z.number().int().nonnegative().nullable(),
      })
      .passthrough(),
    compatibility: z
      .object({
        training_algorithm: z.string().nullable(),
      })
      .passthrough(),
    exported_at: z.string(),
    files: z.array(checkpointBundleFileSchema),
    format_name: z.string(),
    schema_version: z.number().int().positive(),
  })
  .passthrough();

export const checkpointCatalogBundleSchema = z.object({
  url: z.string(),
  filename: z.string(),
  size_bytes: z.number().int().nonnegative(),
  sha256: z.string(),
});

export const checkpointCatalogEntrySchema = z.object({
  id: z.string(),
  version: z.string(),
  name: z.string(),
  bundle: checkpointCatalogBundleSchema,
  manifest: checkpointBundleManifestSchema,
  installed_checkpoint_id: z.string().nullable(),
});

export const publishedCheckpointSchema = z.object({
  id: z.string(),
  checkpoint_id: z.string(),
  version: z.string(),
  name: z.string(),
  run_id: z.string(),
  run: managedRunSummarySchema.nullable(),
  config: managedRunConfigSchema,
  import_dir: z.string(),
  source_run_id: z.string().nullable(),
  source_run_name: z.string().nullable(),
  source_artifact: z.enum(["latest", "best"]),
  local_num_timesteps: z.number().int().nonnegative().nullable(),
  lineage_num_timesteps: z.number().int().nonnegative().nullable(),
  source_bundle_sha256: z.string().nullable(),
  has_evaluation_metrics: z.boolean(),
  has_engine_tuning_state: z.boolean(),
  evaluation_summary: evaluationResultSummarySchema.nullable(),
  engine_tuning_state: engineTuningRuntimeStateSchema.nullable(),
  exported_at: z.string(),
  imported_at: z.string(),
  updated_at: z.string(),
});

export const checkpointCatalogResponseSchema = z.object({
  catalog: z.object({
    format_name: z.string(),
    schema_version: z.number().int().positive(),
    updated_at: z.string(),
  }),
  entries: z.array(checkpointCatalogEntrySchema),
  installed_checkpoints: z.array(publishedCheckpointSchema),
});

export const installCheckpointResponseSchema = z.object({
  status: z.enum(["installed", "already_installed"]),
  checkpoint: publishedCheckpointSchema,
});

export const deleteCheckpointResponseSchema = z.object({
  deleted: z.boolean(),
});

export type CheckpointCatalogEntry = z.infer<typeof checkpointCatalogEntrySchema>;
export type CheckpointCatalogResponse = z.infer<typeof checkpointCatalogResponseSchema>;
export type DeleteCheckpointResponse = z.infer<typeof deleteCheckpointResponseSchema>;
export type InstallCheckpointResponse = z.infer<typeof installCheckpointResponseSchema>;
export type PublishedCheckpoint = z.infer<typeof publishedCheckpointSchema>;
