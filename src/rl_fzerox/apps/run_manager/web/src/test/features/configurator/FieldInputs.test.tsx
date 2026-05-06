// src/rl_fzerox/apps/run_manager/web/src/test/features/configurator/FieldInputs.test.tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import {
  OptionalNumberField,
  OptionalRangePairField,
  RangePairField,
} from "@/features/configurator/fields";

describe("Configurator field inputs", () => {
  it("does not coerce a cleared optional number input to zero", () => {
    const onChange = vi.fn();

    render(<OptionalNumberField help="test" label="Penalty" value={0.25} onChange={onChange} />);

    const input = screen.getByRole("spinbutton", { name: "Penalty" });
    fireEvent.change(input, { target: { value: "" } });

    expect(onChange).not.toHaveBeenCalledWith(0);
    expect(input).toHaveValue(null);
  });

  it("does not coerce a cleared range minimum input to zero", () => {
    const onChange = vi.fn();

    render(
      <RangePairField
        help="test"
        label="Window"
        max={100}
        min={1}
        step={1}
        valueMax={20}
        valueMin={10}
        onChange={onChange}
      />,
    );

    const input = screen.getByRole("spinbutton", { name: "Window minimum" });
    fireEvent.change(input, { target: { value: "" } });

    expect(onChange).not.toHaveBeenCalledWith({ max: 20, min: 0 });
    expect(input).toHaveValue(null);
  });

  it("does not coerce a cleared optional range maximum input to zero", () => {
    const onChange = vi.fn();

    render(
      <OptionalRangePairField
        defaultMax={20}
        defaultMin={10}
        help="test"
        label="Soft limit"
        max={100}
        min={1}
        step={1}
        valueMax={20}
        valueMin={10}
        onChange={onChange}
      />,
    );

    const input = screen.getByRole("spinbutton", { name: "Soft limit maximum" });
    fireEvent.change(input, { target: { value: "" } });

    expect(onChange).not.toHaveBeenCalledWith({ max: 0, min: 10 });
    expect(input).toHaveValue(null);
  });
});
