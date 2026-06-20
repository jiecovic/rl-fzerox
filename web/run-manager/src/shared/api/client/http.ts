// web/run-manager/src/shared/api/client/http.ts
import type { ZodIssue, ZodType } from "zod";

import { ApiSchemaMismatchError } from "@/shared/api/client/errors";

export interface RequestOptions {
  signal?: AbortSignal;
}

export async function getJson(url: string, options: RequestOptions = {}): Promise<unknown> {
  const response = await fetch(url, { signal: options.signal });
  return parseJson(response);
}

export function parseApiPayload<T>(schema: ZodType<T>, payload: unknown): T {
  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    throw new ApiSchemaMismatchError(formatSchemaIssues(parsed.error.issues));
  }
  return parsed.data;
}

function formatSchemaIssues(issues: ZodIssue[]): string {
  const visibleIssues = issues.slice(0, 3).map((issue) => {
    const path = issue.path.length > 0 ? issue.path.join(".") : "<root>";
    return `${path}: ${issue.message}`;
  });
  const suffix =
    issues.length > visibleIssues.length ? ` (+${issues.length - visibleIssues.length} more)` : "";
  return `First issue${visibleIssues.length === 1 ? "" : "s"}: ${visibleIssues.join("; ")}${suffix}`;
}

export async function parseJson(response: Response): Promise<unknown> {
  const payload = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(errorMessageFromPayload(response, payload));
  }
  return payload;
}

export async function responseErrorMessage(response: Response): Promise<string> {
  return errorMessageFromPayload(response, await readJsonResponse(response));
}

async function readJsonResponse(response: Response): Promise<unknown> {
  const text = await response.text();
  if (text.length === 0) {
    return null;
  }
  try {
    return JSON.parse(text) as unknown;
  } catch (caught) {
    if (!response.ok) {
      return null;
    }
    throw caught;
  }
}

function isApiErrorPayload(payload: unknown): payload is { error?: unknown } {
  return typeof payload === "object" && payload !== null && "error" in payload;
}

function isFastApiErrorPayload(payload: unknown): payload is { detail?: unknown } {
  return typeof payload === "object" && payload !== null && "detail" in payload;
}

function errorMessageFromPayload(response: Response, payload: unknown) {
  if (isApiErrorPayload(payload) && typeof payload.error === "string") {
    return payload.error;
  }
  if (isFastApiErrorPayload(payload) && typeof payload.detail === "string") {
    return payload.detail;
  }
  return response.statusText || `request failed with status ${response.status}`;
}
