// web/run-manager/src/test/shared/ui/ConfigDisclosure.test.tsx

import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ConfigDisclosure } from "@/shared/ui/config/ConfigDisclosure";
import { render, screen } from "@/test/render";

describe("ConfigDisclosure", () => {
  it("unmounts uncontrolled body content when collapsed", async () => {
    const user = userEvent.setup();

    render(
      <ConfigDisclosure title="Advanced settings">
        <button type="button">Heavy control</button>
      </ConfigDisclosure>,
    );

    expect(screen.getByRole("button", { name: "Heavy control" })).toBeInTheDocument();

    await user.click(screen.getByText("Advanced settings"));

    expect(screen.queryByRole("button", { name: "Heavy control" })).not.toBeInTheDocument();

    await user.click(screen.getByText("Advanced settings"));

    expect(screen.getByRole("button", { name: "Heavy control" })).toBeInTheDocument();
  });

  it("follows controlled open state and reports toggle requests", async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();
    const { rerender } = render(
      <ConfigDisclosure open={false} title="Runtime settings" onToggle={onToggle}>
        <button type="button">Runtime control</button>
      </ConfigDisclosure>,
    );

    expect(screen.queryByRole("button", { name: "Runtime control" })).not.toBeInTheDocument();

    await user.click(screen.getByText("Runtime settings"));

    expect(onToggle).toHaveBeenLastCalledWith(true);
    expect(screen.queryByRole("button", { name: "Runtime control" })).not.toBeInTheDocument();

    rerender(
      <ConfigDisclosure open title="Runtime settings" onToggle={onToggle}>
        <button type="button">Runtime control</button>
      </ConfigDisclosure>,
    );

    expect(screen.getByRole("button", { name: "Runtime control" })).toBeInTheDocument();
  });
});
