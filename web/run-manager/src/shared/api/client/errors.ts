// web/run-manager/src/shared/api/client/errors.ts
export const API_SCHEMA_MISMATCH_MESSAGE = "Run-manager backend is outdated. Restart run-manager.";

export class ApiSchemaMismatchError extends Error {
  constructor() {
    super(API_SCHEMA_MISMATCH_MESSAGE);
    this.name = "ApiSchemaMismatchError";
  }
}
