import { describe, expect, it } from "vitest";
import {
	convexPolygonIoU,
	getOBBCorners,
	intersectConvexPolygons,
	polygonArea,
} from "../obbUtils";
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

// ---------------------------------------------------------------------------
// polygonArea
// ---------------------------------------------------------------------------

describe("polygonArea", () => {
	it("should compute area of a unit square", () => {
		const square: [number, number][] = [
			[0, 0],
			[1, 0],
			[1, 1],
			[0, 1],
		];
		expect(polygonArea(square)).toBeCloseTo(1, 10);
	});

	it("should compute area of a rectangle", () => {
		const rect: [number, number][] = [
			[0, 0],
			[4, 0],
			[4, 3],
			[0, 3],
		];
		expect(polygonArea(rect)).toBeCloseTo(12, 10);
	});

	it("should return 0 for fewer than 3 vertices", () => {
		expect(polygonArea([])).toBe(0);
		expect(
			polygonArea([
				[0, 0],
				[1, 1],
			]),
		).toBe(0);
	});

	it("should handle CW and CCW winding identically (absolute area)", () => {
		const ccw: [number, number][] = [
			[0, 0],
			[2, 0],
			[2, 2],
			[0, 2],
		];
		const cw: [number, number][] = [...ccw].reverse();
		expect(polygonArea(ccw)).toBeCloseTo(polygonArea(cw), 10);
	});
});

// ---------------------------------------------------------------------------
// intersectConvexPolygons
// ---------------------------------------------------------------------------

describe("intersectConvexPolygons", () => {
	it("should return full overlap for identical rectangles", () => {
		const rect: [number, number][] = [
			[0, 0],
			[2, 0],
			[2, 2],
			[0, 2],
		];
		const inter = intersectConvexPolygons(rect, rect);
		expect(polygonArea(inter)).toBeCloseTo(4, 5);
	});

	it("should return empty for non-overlapping rectangles", () => {
		const a: [number, number][] = [
			[0, 0],
			[1, 0],
			[1, 1],
			[0, 1],
		];
		const b: [number, number][] = [
			[5, 5],
			[6, 5],
			[6, 6],
			[5, 6],
		];
		const inter = intersectConvexPolygons(a, b);
		expect(polygonArea(inter)).toBeCloseTo(0, 10);
	});

	it("should compute partial overlap of two offset squares", () => {
		// Two 2x2 squares, offset by 1 in both x and y
		const a: [number, number][] = [
			[0, 0],
			[2, 0],
			[2, 2],
			[0, 2],
		];
		const b: [number, number][] = [
			[1, 1],
			[3, 1],
			[3, 3],
			[1, 3],
		];
		const inter = intersectConvexPolygons(a, b);
		// Intersection should be 1x1 square from (1,1) to (2,2) = area 1
		expect(polygonArea(inter)).toBeCloseTo(1, 5);
	});

	it("should handle rotated rectangles", () => {
		// Axis-aligned 2x2 square centered at origin
		const square: [number, number][] = [
			[-1, -1],
			[1, -1],
			[1, 1],
			[-1, 1],
		];
		// 45-degree rotated 2x2 square centered at origin (diamond)
		const s = Math.SQRT2;
		const diamond: [number, number][] = [
			[0, -s],
			[s, 0],
			[0, s],
			[-s, 0],
		];
		const inter = intersectConvexPolygons(square, diamond);
		// The intersection is an octagon; area = 8(√2 − 1) ≈ 3.3137
		expect(polygonArea(inter)).toBeCloseTo(8 * (Math.SQRT2 - 1), 4);
	});
});

// ---------------------------------------------------------------------------
// convexPolygonIoU
// ---------------------------------------------------------------------------

describe("convexPolygonIoU", () => {
	it("should return 1 for identical rectangles", () => {
		const rect: [number, number][] = [
			[0, 0],
			[10, 0],
			[10, 5],
			[0, 5],
		];
		expect(convexPolygonIoU(rect, rect)).toBeCloseTo(1, 5);
	});

	it("should return 0 for non-overlapping rectangles", () => {
		const a: [number, number][] = [
			[0, 0],
			[1, 0],
			[1, 1],
			[0, 1],
		];
		const b: [number, number][] = [
			[10, 10],
			[11, 10],
			[11, 11],
			[10, 11],
		];
		expect(convexPolygonIoU(a, b)).toBe(0);
	});

	it("should compute correct IoU for partially overlapping squares", () => {
		// Two 2x2 squares offset by 1 in x and y
		const a: [number, number][] = [
			[0, 0],
			[2, 0],
			[2, 2],
			[0, 2],
		];
		const b: [number, number][] = [
			[1, 1],
			[3, 1],
			[3, 3],
			[1, 3],
		];
		// Intersection = 1, union = 4 + 4 - 1 = 7
		expect(convexPolygonIoU(a, b)).toBeCloseTo(1 / 7, 5);
	});

	it("should correctly handle OBB corners with different angles", () => {
		// Same center, same size, angle 0 vs angle PI/2
		const d0 = makeDetection({
			cx: 0,
			cy: 0,
			width: 100,
			height: 10,
			angle: 0,
		});
		const d90 = makeDetection({
			cx: 0,
			cy: 0,
			width: 100,
			height: 10,
			angle: Math.PI / 2,
		});
		const corners0 = getOBBCorners(d0);
		const corners90 = getOBBCorners(d90);
		const iou = convexPolygonIoU(corners0, corners90);
		// Cross-shaped intersection: 10x10 = 100
		// Each box area: 100*10 = 1000, union = 1000 + 1000 - 100 = 1900
		// IoU ≈ 100/1900 ≈ 0.0526
		expect(iou).toBeCloseTo(100 / 1900, 3);
	});

	it("should return high IoU for OBBs with same angle and slight offset", () => {
		const d1 = makeDetection({
			cx: 0,
			cy: 0,
			width: 100,
			height: 10,
			angle: 0.3,
		});
		const d2 = makeDetection({
			cx: 2,
			cy: 2,
			width: 100,
			height: 10,
			angle: 0.3,
		});
		const iou = convexPolygonIoU(getOBBCorners(d1), getOBBCorners(d2));
		// Same angle, slight offset: should still be high IoU
		expect(iou).toBeGreaterThan(0.5);
	});
});
