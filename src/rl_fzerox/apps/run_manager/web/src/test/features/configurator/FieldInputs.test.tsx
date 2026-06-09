// src/rl_fzerox/apps/run_manager/web/src/test/features/configurator/FieldInputs.test.tsx

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  OptionalNumberField,
  OptionalRangePairField,
  RangeNumberField,
  RangePairField,
} from "@/features/configurator/fields";
import { cleanup, fireEvent, render, screen } from "@/test/render";

describe("Configurator field inputs", () => {
  afterEach(() => {
    cleanup();
  });

  it("drafts range number edits until blur", () => {
    const onChange = vi.fn();

    render(
      <RangeNumberField
        help="test"
        label="Clip range"
        max={0.5}
        min={0.01}
        rangeStep={0.01}
        numberStep="0.001"
        value={0.2}
        onChange={onChange}
      />,
    );

    const input = screen.getByRole("textbox", { name: "Clip range" });
    fireEvent.change(input, { target: { value: "0.17" } });

    expect(onChange).not.toHaveBeenCalled();
    fireEvent.blur(input);

    expect(onChange).toHaveBeenLastCalledWith(0.17);
  });

  it("does not coerce a cleared range number input to zero", () => {
    const onChange = vi.fn();

    render(
      <RangeNumberField
        help="test"
        label="Clip range"
        max={0.5}
        min={0.01}
        rangeStep={0.01}
        numberStep="0.001"
        value={0.2}
        onChange={onChange}
      />,
    );

    const input = screen.getByRole("textbox", { name: "Clip range" });
    fireEvent.change(input, { target: { value: "" } });

    expect(onChange).not.toHaveBeenCalledWith(0);
    expect(input).toHaveValue("");
  });

  it("does not coerce a cleared optional number input to zero", () => {
    const onChange = vi.fn();

    render(<OptionalNumberField help="test" label="Penalty" value={0.25} onChange={onChange} />);

    const input = screen.getByRole("textbox", { name: "Penalty" });
    fireEvent.change(input, { target: { value: "" } });

    expect(onChange).not.toHaveBeenCalledWith(0);
    expect(input).toHaveValue("");
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

    const input = screen.getByRole("textbox", { name: "Window minimum" });
    fireEvent.change(input, { target: { value: "" } });

    expect(onChange).not.toHaveBeenCalledWith({ max: 20, min: 0 });
    expect(input).toHaveValue("");
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

    const input = screen.getByRole("textbox", { name: "Soft limit maximum" });
    fireEvent.change(input, { target: { value: "" } });

    expect(onChange).not.toHaveBeenCalledWith({ max: 0, min: 10 });
    expect(input).toHaveValue("");
  });
});
