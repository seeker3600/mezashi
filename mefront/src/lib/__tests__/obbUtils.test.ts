import { describe, expect, it } from "vitest";
import { getOBBCorners } from "../obbUtils";
import type { Detection } from "../types";

function makeDetection(overrides: Partial<Detection> = {}): Detection {
	return {
		classId: 0,
		className: "plane",
		confidence: 0.9,
		cx: 100,
		cy: 100,
		width: 40,
		height: 20,
		angle: 0,
		...overrides,
	};
}

describe("getOBBCorners", () => {
	it("should return 4 corner points", () => {
		const corners = getOBBCorners(makeDetection());
		expect(corners).toHaveLength(4);
	});

	it("should return correct corners for axis-aligned box", () => {
		const d = makeDetection({
			cx: 50,
			cy: 50,
			width: 20,
			height: 10,
			angle: 0,
		});
		const corners = getOBBCorners(d);
		// With angle=0: corners are at (-w/2,-h/2), (w/2,-h/2), (w/2,h/2), (-w/2,h/2)
		expect(corners[0]).toEqual([40, 45]); // cx-w/2, cy-h/2
		expect(corners[1]).toEqual([60, 45]); // cx+w/2, cy-h/2
		expect(corners[2]).toEqual([60, 55]); // cx+w/2, cy+h/2
		expect(corners[3]).toEqual([40, 55]); // cx-w/2, cy+h/2
	});

	it("should rotate corners when angle is non-zero", () => {
		const d = makeDetection({
			cx: 0,
			cy: 0,
			width: 2,
			height: 2,
			angle: Math.PI / 4,
		});
		const corners = getOBBCorners(d);
		// With 45-degree rotation of a 2x2 box centered at origin
		const sqrt2 = Math.SQRT2;
		for (const [x, y] of corners) {
			// All corners should be at distance sqrt(1^2+1^2) = sqrt(2) from center
			expect(Math.sqrt(x * x + y * y)).toBeCloseTo(sqrt2, 5);
		}
	});

	it("should handle 90-degree rotation", () => {
		const d = makeDetection({
			cx: 0,
			cy: 0,
			width: 4,
			height: 2,
			angle: Math.PI / 2,
		});
		const corners = getOBBCorners(d);
		// 90 degrees: width becomes height and vice versa
		// Original corners: (-2,-1), (2,-1), (2,1), (-2,1)
		// After 90° rotation: (1,-2), (1,2), (-1,2), (-1,-2)
		expect(corners[0][0]).toBeCloseTo(1, 5);
		expect(corners[0][1]).toBeCloseTo(-2, 5);
		expect(corners[1][0]).toBeCloseTo(1, 5);
		expect(corners[1][1]).toBeCloseTo(2, 5);
	});
});
