import type { ElkNode } from "elkjs/lib/elk.bundled.js";

import { elkLayoutOptions } from "./constants";

export async function layoutGraph(elkGraph: ElkNode) {
  const { default: ELK } = await import("elkjs/lib/elk.bundled.js");
  const elk = new ELK({ defaultLayoutOptions: elkLayoutOptions });
  return elk.layout(elkGraph);
}
