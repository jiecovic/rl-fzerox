// web/run-manager/src/shared/api/client/errors.ts
export const API_SCHEMA_MISMATCH_MESSAGE =
  "Run-manager API response does not match the frontend contract.";

export class ApiSchemaMismatchError extends Error {
  constructor(details?: string) {
    super(details ? `${API_SCHEMA_MISMATCH_MESSAGE} ${details}` : API_SCHEMA_MISMATCH_MESSAGE);
    this.name = "ApiSchemaMismatchError";
  }
}
