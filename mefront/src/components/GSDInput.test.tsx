import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { GSDDialog } from "./GSDInput";

afterEach(() => {
	cleanup();
});

describe("GSDDialog", () => {
	it("uses the model resolution as the recommended initial value", () => {
		render(
			<GSDDialog modelDefault={1} onConfirm={vi.fn()} onCancel={vi.fn()} />,
		);

		expect(
			(screen.getByLabelText("1 ピクセルあたりの距離") as HTMLInputElement)
				.value,
		).toBe("1");
		expect(screen.getByRole("button", { name: "1 m (推奨)" })).not.toBeNull();
	});

	it("updates the resolution input when a suggested value is selected", () => {
		render(
			<GSDDialog modelDefault={1} onConfirm={vi.fn()} onCancel={vi.fn()} />,
		);

		fireEvent.click(screen.getByRole("button", { name: "30 cm" }));

		expect(
			(screen.getByLabelText("1 ピクセルあたりの距離") as HTMLInputElement)
				.value,
		).toBe("0.3");
	});

	it("confirms only a valid resolution and does not offer skip", () => {
		const onConfirm = vi.fn();
		render(
			<GSDDialog modelDefault={1} onConfirm={onConfirm} onCancel={vi.fn()} />,
		);

		const input = screen.getByLabelText("1 ピクセルあたりの距離");
		fireEvent.change(input, { target: { value: "" } });
		expect(
			(
				screen.getByRole("button", {
					name: "この解像度で推論",
				}) as HTMLButtonElement
			).disabled,
		).toBe(true);
		expect(screen.queryByRole("button", { name: "スキップ" })).toBeNull();

		fireEvent.change(input, { target: { value: "0.5" } });
		fireEvent.click(screen.getByRole("button", { name: "この解像度で推論" }));

		expect(onConfirm).toHaveBeenCalledWith(0.5);
	});
});
