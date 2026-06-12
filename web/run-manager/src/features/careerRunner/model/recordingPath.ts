// web/run-manager/src/features/careerRunner/model/recordingPath.ts
export function defaultCareerRecordingPath(saveGameId: string | null): string {
  const owner = saveGameId ?? "new-save";
  const timestamp = new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-");
  return `local/recordings/career/${owner}/${timestamp}.mkv`;
}
