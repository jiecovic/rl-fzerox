// web/run-manager/src/shared/api/contract/checkpoints.ts
import { z } from "zod";

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
  source_artifact: z.string(),
  local_num_timesteps: z.number().int().nonnegative().nullable(),
  lineage_num_timesteps: z.number().int().nonnegative().nullable(),
  source_bundle_sha256: z.string().nullable(),
  has_evaluation_metrics: z.boolean(),
  has_engine_tuning_state: z.boolean(),
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

export type CheckpointCatalogEntry = z.infer<typeof checkpointCatalogEntrySchema>;
export type CheckpointCatalogResponse = z.infer<typeof checkpointCatalogResponseSchema>;
export type InstallCheckpointResponse = z.infer<typeof installCheckpointResponseSchema>;
export type PublishedCheckpoint = z.infer<typeof publishedCheckpointSchema>;
