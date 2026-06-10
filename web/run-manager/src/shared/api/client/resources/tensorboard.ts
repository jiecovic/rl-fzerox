// web/run-manager/src/shared/api/client/resources/tensorboard.ts
import { parseApiPayload, parseJson } from "@/shared/api/client/http";
import {
  rebuildTensorboardViewsResponseSchema,
  type TensorboardViewGroup,
} from "@/shared/api/contract";

export async function rebuildTensorboardViews(): Promise<TensorboardViewGroup[]> {
  const response = await fetch("/api/tensorboard-views/rebuild", { method: "POST" });
  const payload = parseApiPayload(rebuildTensorboardViewsResponseSchema, await parseJson(response));
  return payload.tensorboard_views;
}
