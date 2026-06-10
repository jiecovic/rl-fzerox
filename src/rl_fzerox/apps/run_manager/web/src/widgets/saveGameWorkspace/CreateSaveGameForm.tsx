// src/rl_fzerox/apps/run_manager/web/src/widgets/saveGameWorkspace/CreateSaveGameForm.tsx
import type { SaveGameSession } from "@/app/workspace/types";
import { Button } from "@/shared/ui/Button";
import { FieldInput, FieldShell } from "@/shared/ui/Field";
import { FloatingNotice } from "@/shared/ui/FloatingNotice";

export function CreateSaveGameForm({
  error,
  isCreating,
  onCreateSaveGame,
  onPatchSession,
  session,
}: {
  error: string | null;
  isCreating: boolean;
  onCreateSaveGame: () => void;
  onPatchSession: (
    sessionId: SaveGameSession["sessionId"],
    patch: Partial<Omit<SaveGameSession, "sessionId">>,
  ) => void;
  session: SaveGameSession;
}) {
  return (
    <section className="grid gap-4 border border-app-border bg-app-surface p-5">
      {error !== null ? <FloatingNotice tone="error">{error}</FloatingNotice> : null}
      <div className="grid gap-4 md:grid-cols-[minmax(280px,1fr)_auto] md:items-end">
        <FieldShell>
          <span>Name</span>
          <FieldInput
            aria-label="Save game name"
            value={session.nameText}
            onChange={(event) =>
              onPatchSession(session.sessionId, { nameText: event.currentTarget.value })
            }
          />
        </FieldShell>
        <Button disabled={isCreating} variant="primary" onClick={onCreateSaveGame}>
          {isCreating ? "Creating" : "Create"}
        </Button>
      </div>
    </section>
  );
}
