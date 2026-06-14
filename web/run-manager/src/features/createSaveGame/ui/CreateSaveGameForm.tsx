// web/run-manager/src/features/createSaveGame/ui/CreateSaveGameForm.tsx
import type { SaveGameSession } from "@/app/workspace/types";
import { Button } from "@/shared/ui/Button";
import { FieldInput, FieldShell } from "@/shared/ui/Field";

export function CreateSaveGameForm({
  isCreating,
  onCreateSaveGame,
  onPatchSession,
  session,
}: {
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
