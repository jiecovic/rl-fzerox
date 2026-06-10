// web/run-manager/src/shared/api/client/bundles.ts

import { parseApiPayload, parseJson, responseErrorMessage } from "@/shared/api/client/http";
import { type ManagedRun, type ManagedRunDetail, runResponseSchema } from "@/shared/api/contract";

interface RunBundleWritableFile {
  close: () => Promise<void> | void;
  write: (data: Blob) => Promise<void> | void;
}

interface RunBundleFileHandle {
  createWritable: () => Promise<RunBundleWritableFile>;
}

interface RunBundleSavePickerHost {
  showSaveFilePicker: (options: {
    suggestedName: string;
    types: Array<{ accept: Record<string, string[]>; description: string }>;
  }) => Promise<RunBundleFileHandle>;
}

export async function exportRunBundle(run: ManagedRun): Promise<void> {
  const response = await fetch(`/api/runs/${encodeURIComponent(run.id)}/export`);
  if (!response.ok) {
    throw new Error(await responseErrorMessage(response));
  }
  const blob = await response.blob();
  await saveRunBundleBlob(blob, exportFilename(response, run));
}

export async function importRunBundle(file: File): Promise<ManagedRunDetail> {
  const form = new FormData();
  form.append("bundle", file);
  const response = await fetch("/api/run-imports", {
    method: "POST",
    body: form,
  });
  const payload = parseApiPayload(runResponseSchema, await parseJson(response));
  return payload.run;
}

function exportFilename(response: Response, run: ManagedRun) {
  const header = response.headers.get("content-disposition");
  const match = header?.match(/filename="?([^";]+)"?/i);
  return match?.[1] ?? `${run.id}.zip`;
}

async function saveRunBundleBlob(blob: Blob, filename: string) {
  const pickerHost = saveFilePickerHost(window);
  if (pickerHost === null) {
    downloadBlob(blob, filename);
    return;
  }
  try {
    const handle = await pickerHost.showSaveFilePicker({
      suggestedName: filename,
      types: [{ accept: { "application/zip": [".zip"] }, description: "Run export bundle" }],
    });
    const writable = await handle.createWritable();
    try {
      await writable.write(blob);
    } finally {
      await writable.close();
    }
  } catch (caught) {
    if (caught instanceof DOMException && caught.name === "AbortError") {
      return;
    }
    throw caught;
  }
}

function saveFilePickerHost(candidate: unknown): RunBundleSavePickerHost | null {
  return hasSaveFilePicker(candidate) ? candidate : null;
}

function hasSaveFilePicker(candidate: unknown): candidate is RunBundleSavePickerHost {
  if (
    typeof candidate !== "object" ||
    candidate === null ||
    !("showSaveFilePicker" in candidate) ||
    typeof candidate.showSaveFilePicker !== "function"
  ) {
    return false;
  }
  return true;
}

function downloadBlob(blob: Blob, filename: string) {
  const href = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = href;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(href);
}
