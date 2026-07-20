import { describe, expect, it } from "vitest";
import {
	aggregateDirectionsByGrid,
	getDetailScaleThreshold,
	getDirectionDisplayMode,
} from "../directionDisplay";
import type { Detection } from "../types";

function makeDetection(overrides: Partial<Detection> = {}): Detection {
	return {
		classId: 0,
		className: "ship",
		confidence: 0.9,
		cx: 50,
		cy: 50,
		width: 100,
		height: 20,
		angle: 0,
		...overrides,
	};
}

describe("getDetailScaleThreshold", () => {
	it("requires a higher scale when detection long axes are shorter", () => {
		const longAxisThreshold = getDetailScaleThreshold(
			[makeDetection({ width: 100 })],
			0.5,
		);
		const shortAxisThreshold = getDetailScaleThreshold(
			[makeDetection({ width: 20 })],
			0.5,
		);

		expect(shortAxisThreshold).toBeGreaterThan(longAxisThreshold);
	});

	it("uses the configured target ratio instead of the smallest outlier", () => {
		const threshold = getDetailScaleThreshold(
			[
				makeDetection({ width: 100 }),
				makeDetection({ width: 100 }),
				makeDetection({ width: 100 }),
				makeDetection({ width: 100 }),
				makeDetection({ width: 1 }),
			],
			1,
		);

		expect(threshold).toBe(1);
	});
});

describe("getDirectionDisplayMode", () => {
	it("keeps the previous mode inside the hysteresis range", () => {
		expect(getDirectionDisplayMode(1, 1, "overview")).toBe("overview");
		expect(getDirectionDisplayMode(1, 1, "detail")).toBe("detail");
	});

	it("switches mode outside the hysteresis range", () => {
		expect(getDirectionDisplayMode(1.2, 1, "overview")).toBe("detail");
		expect(getDirectionDisplayMode(0.8, 1, "detail")).toBe("overview");
	});
});

describe("aggregateDirectionsByGrid", () => {
	it("averages 0 and 180 degree OBB axes into the same direction", () => {
		const groups = aggregateDirectionsByGrid(
			[
				makeDetection({ cx: 10, cy: 10, angle: 0 }),
				makeDetection({ cx: 90, cy: 90, angle: Math.PI - 0.01 }),
			],
			100,
		);

		expect(groups).toHaveLength(1);
		expect(groups[0].count).toBe(2);
		expect(Math.sin(groups[0].angle)).toBeCloseTo(0, 2);
	});

	it("keeps detections in separate grid cells separate", () => {
		const groups = aggregateDirectionsByGrid(
			[makeDetection({ cx: 10, cy: 10 }), makeDetection({ cx: 110, cy: 10 })],
			100,
		);

		expect(groups).toHaveLength(2);
		expect(groups.map((group) => group.count)).toEqual([1, 1]);
	});
});
