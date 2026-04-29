import { describe, expect, it } from "vitest";
import { mapDetectionFromAugmentedTile } from "../inference";
import type { Detection } from "../types";

const baseDetection: Detection = {
	classId: 0,
	className: "ship",
	confidence: 0.9,
	cx: 100,
	cy: 200,
	width: 50,
	height: 30,
	angle: 0.5,
};

describe("mapDetectionFromAugmentedTile", () => {
	const size = 640;

	it("maps horizontal flip back to original coordinates", () => {
		const mapped = mapDetectionFromAugmentedTile(
			baseDetection,
			"flipHorizontal",
			size,
		);
		expect(mapped.cx).toBe(size - baseDetection.cx);
		expect(mapped.cy).toBe(baseDetection.cy);
		expect(mapped.angle).toBeCloseTo(Math.PI - baseDetection.angle);
	});

	it("maps vertical flip back to original coordinates", () => {
		const mapped = mapDetectionFromAugmentedTile(
			baseDetection,
			"flipVertical",
			size,
		);
		expect(mapped.cx).toBe(baseDetection.cx);
		expect(mapped.cy).toBe(size - baseDetection.cy);
		expect(mapped.angle).toBeCloseTo(-baseDetection.angle);
	});

	it("maps both-axis flip back to original coordinates", () => {
		const mapped = mapDetectionFromAugmentedTile(
			baseDetection,
			"flipBoth",
			size,
		);
		expect(mapped.cx).toBe(size - baseDetection.cx);
		expect(mapped.cy).toBe(size - baseDetection.cy);
		expect(mapped.angle).toBeCloseTo(baseDetection.angle - Math.PI);
	});
});
