import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DetectionCanvas } from "./DetectionCanvas";

const context = {
	beginPath: vi.fn(),
	clearRect: vi.fn(),
	closePath: vi.fn(),
	drawImage: vi.fn(),
	restore: vi.fn(),
	save: vi.fn(),
	scale: vi.fn(),
	stroke: vi.fn(),
	translate: vi.fn(),
} as unknown as CanvasRenderingContext2D;

afterEach(() => {
	cleanup();
	vi.restoreAllMocks();
});

describe("DetectionCanvas", () => {
	it("prevents page scrolling while zooming with the mouse wheel", () => {
		vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
			context,
		);
		const addEventListener = vi.spyOn(
			HTMLCanvasElement.prototype,
			"addEventListener",
		);
		const imageSource = document.createElement("canvas");

		const { container } = render(
			<DetectionCanvas
				imageSource={imageSource}
				detections={[]}
				imageWidth={100}
				imageHeight={100}
			/>,
		);
		const canvas = container.querySelector("canvas");
		expect(canvas).not.toBeNull();
		if (!canvas) throw new Error("Detection canvas was not rendered");

		const wheelEvent = new WheelEvent("wheel", {
			bubbles: true,
			cancelable: true,
			deltaY: -1,
		});
		canvas.dispatchEvent(wheelEvent);

		expect(wheelEvent.defaultPrevented).toBe(true);
		expect(addEventListener).toHaveBeenCalledWith(
			"wheel",
			expect.any(Function),
			{ passive: false },
		);
	});
});
