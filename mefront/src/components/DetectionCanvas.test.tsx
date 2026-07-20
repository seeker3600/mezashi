import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DetectionCanvas } from "./DetectionCanvas";

const strokeStyles: string[] = [];
const context = {
	beginPath: vi.fn(),
	clearRect: vi.fn(),
	closePath: vi.fn(),
	drawImage: vi.fn(),
	fill: vi.fn(),
	lineTo: vi.fn(),
	moveTo: vi.fn(),
	restore: vi.fn(),
	save: vi.fn(),
	scale: vi.fn(),
	stroke: vi.fn(),
	translate: vi.fn(),
} as unknown as CanvasRenderingContext2D;

Object.defineProperty(context, "strokeStyle", {
	set(value: string) {
		strokeStyles.push(value);
	},
});

afterEach(() => {
	cleanup();
	vi.restoreAllMocks();
	strokeStyles.length = 0;
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

	it("draws overview direction markers with an outline and a high-contrast foreground", () => {
		vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
			context,
		);
		const imageSource = document.createElement("canvas");

		render(
			<DetectionCanvas
				imageSource={imageSource}
				detections={[
					{
						cx: 50,
						cy: 50,
						width: 20,
						height: 10,
						angle: 0,
						confidence: 0.9,
						classId: 0,
						className: "ship",
					},
				]}
				imageWidth={100}
				imageHeight={100}
				showBoxes={false}
				showLabels={false}
				showDirection
			/>,
		);

		expect(strokeStyles).toContain("rgba(17,24,39,0.9)");
		expect(strokeStyles).toContain("rgba(253,224,71,0.9)");
	});
});
