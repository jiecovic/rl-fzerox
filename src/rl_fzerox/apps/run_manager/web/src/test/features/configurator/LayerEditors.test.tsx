// src/rl_fzerox/apps/run_manager/web/src/test/features/configurator/LayerEditors.test.tsx
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CustomConvTableRows } from "@/features/configurator/sections/policy/LayerEditors";
import type { ManagedRunConfig } from "@/shared/api/contract";

type CustomConvLayer = ManagedRunConfig["policy"]["custom_conv_layers"][number];

describe("CustomConvTableRows", () => {
  afterEach(() => {
    cleanup();
  });

  it("adds residual blocks with same-padding defaults", async () => {
    const user = userEvent.setup();
    const layers: CustomConvLayer[] = [
      {
        kind: "conv",
        out_channels: 16,
        kernel_size: 6,
        stride: 3,
        padding: 0,
      },
    ];
    const onChange = vi.fn();

    render(
      <table>
        <tbody>
          <CustomConvTableRows
            flattenDim={0}
            previewLayers={[]}
            value={layers}
            onChange={onChange}
          />
        </tbody>
      </table>,
    );

    await user.click(screen.getByRole("button", { name: "Add res block" }));

    expect(onChange).toHaveBeenCalledWith([
      layers[0],
      {
        kind: "residual",
        out_channels: 16,
        kernel_size: 3,
        stride: 1,
        padding: 1,
      },
    ]);
  });

  it("reorders layers with drag and drop", () => {
    const layers: CustomConvLayer[] = [
      {
        kind: "conv",
        out_channels: 16,
        kernel_size: 6,
        stride: 3,
        padding: 0,
      },
      {
        kind: "residual",
        out_channels: 16,
        kernel_size: 3,
        stride: 1,
        padding: 1,
      },
    ];
    const onChange = vi.fn();

    render(
      <table>
        <tbody>
          <CustomConvTableRows
            flattenDim={0}
            previewLayers={[]}
            value={layers}
            onChange={onChange}
          />
        </tbody>
      </table>,
    );

    const firstRow = screen.getByText("conv1").closest("tr");
    const secondRow = screen.getByText("res2").closest("tr");
    if (!(firstRow instanceof HTMLTableRowElement) || !(secondRow instanceof HTMLTableRowElement)) {
      throw new Error("expected CNN layer table rows");
    }
    const dataTransfer = fakeDataTransfer();

    fireEvent.dragStart(firstRow, { dataTransfer });
    fireEvent.dragOver(secondRow, { dataTransfer });
    fireEvent.drop(secondRow, { dataTransfer });

    expect(onChange).toHaveBeenCalledWith([layers[1], layers[0]]);
  });
});

function fakeDataTransfer() {
  const values = new Map<string, string>();
  return {
    dropEffect: "move",
    effectAllowed: "move",
    getData: (key: string) => values.get(key) ?? "",
    setData: (key: string, value: string) => {
      values.set(key, value);
    },
  };
}
