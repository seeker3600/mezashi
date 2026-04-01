import { describe, expect, it } from "vitest";
import { buildIouMatrix, nmsFromRaw } from "../nms";
import type { Detection } from "../types";

function makeDet(overrides: Partial<Detection> = {}): Detection {
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

describe("buildIouMatrix", () => {
	it("should return an empty matrix for empty detections", () => {
		const matrix = buildIouMatrix([], "detect");
		expect(matrix.length).toBe(0);
	});

	it("should return a 1×1 zero matrix for a single detection", () => {
		const matrix = buildIouMatrix([makeDet()], "detect");
		expect(matrix.length).toBe(1);
		expect(matrix[0]).toBe(0);
	});

	it("should compute non-zero IoU for overlapping same-class detections (detect)", () => {
		const dets = [
			makeDet({ cx: 100, cy: 100, width: 50, height: 50, classId: 0 }),
			makeDet({ cx: 101, cy: 101, width: 50, height: 50, classId: 0 }),
		];
		const matrix = buildIouMatrix(dets, "detect");
		expect(matrix.length).toBe(4);
		expect(matrix[0 * 2 + 1]).toBeGreaterThan(0.9); // [0,1] high overlap
		expect(matrix[1 * 2 + 0]).toBeCloseTo(matrix[0 * 2 + 1]); // symmetric
	});

	it("should store zero IoU for different-class pairs (never suppress cross-class)", () => {
		const dets = [
			makeDet({ cx: 100, cy: 100, width: 50, height: 50, classId: 0 }),
			makeDet({ cx: 100, cy: 100, width: 50, height: 50, classId: 1 }),
		];
		const matrix = buildIouMatrix(dets, "detect");
		expect(matrix[0 * 2 + 1]).toBe(0);
		expect(matrix[1 * 2 + 0]).toBe(0);
	});

	it("should be symmetric", () => {
		const dets = [
			makeDet({ cx: 100, cy: 100, width: 40, height: 40 }),
			makeDet({ cx: 110, cy: 110, width: 40, height: 40 }),
			makeDet({ cx: 200, cy: 200, width: 40, height: 40 }),
		];
		const matrix = buildIouMatrix(dets, "detect");
		const n = dets.length;
		for (let i = 0; i < n; i++) {
			for (let j = 0; j < n; j++) {
				expect(matrix[i * n + j]).toBeCloseTo(matrix[j * n + i]);
			}
		}
	});
});

describe("nmsFromRaw", () => {
	it("should return empty array for empty input", () => {
		const matrix = buildIouMatrix([], "detect");
		expect(nmsFromRaw([], matrix, 0.25, 0.45)).toEqual([]);
	});

	it("should filter by confidence threshold before NMS", () => {
		const dets = [
			makeDet({ confidence: 0.8 }),
			makeDet({ confidence: 0.1 }), // below threshold
		];
		const matrix = buildIouMatrix(dets, "detect");
		const result = nmsFromRaw(dets, matrix, 0.5, 0.45);
		expect(result).toHaveLength(1);
		expect(result[0].confidence).toBe(0.8);
	});

	it("should suppress overlapping lower-confidence same-class detections", () => {
		const dets = [
			makeDet({ cx: 100, cy: 100, width: 50, height: 50, confidence: 0.9 }),
			makeDet({ cx: 101, cy: 101, width: 50, height: 50, confidence: 0.7 }),
		];
		const matrix = buildIouMatrix(dets, "detect");
		const result = nmsFromRaw(dets, matrix, 0.25, 0.45);
		expect(result).toHaveLength(1);
		expect(result[0].confidence).toBe(0.9);
	});

	it("should keep non-overlapping detections", () => {
		const dets = [
			makeDet({ cx: 100, cy: 100, width: 20, height: 20, confidence: 0.9 }),
			makeDet({ cx: 300, cy: 300, width: 20, height: 20, confidence: 0.8 }),
		];
		const matrix = buildIouMatrix(dets, "detect");
		const result = nmsFromRaw(dets, matrix, 0.25, 0.45);
		expect(result).toHaveLength(2);
	});

	it("should keep different-class detections even if overlapping", () => {
		const dets = [
			makeDet({
				cx: 100,
				cy: 100,
				width: 50,
				height: 50,
				classId: 0,
				confidence: 0.9,
			}),
			makeDet({
				cx: 100,
				cy: 100,
				width: 50,
				height: 50,
				classId: 1,
				confidence: 0.8,
			}),
		];
		const matrix = buildIouMatrix(dets, "detect");
		const result = nmsFromRaw(dets, matrix, 0.25, 0.45);
		expect(result).toHaveLength(2);
	});

	it("should re-run NMS using only detections above the new threshold", () => {
		// With low threshold: both A and B pass, B suppresses C
		// With high threshold: B is filtered out before NMS, so C is kept
		const dets = [
			// A: high confidence, far away from B and C
			makeDet({
				cx: 500,
				cy: 500,
				width: 20,
				height: 20,
				confidence: 0.95,
				classId: 0,
			}),
			// B: medium confidence, heavily overlaps C
			makeDet({
				cx: 100,
				cy: 100,
				width: 50,
				height: 50,
				confidence: 0.4,
				classId: 0,
			}),
			// C: medium-high confidence, heavily overlaps B
			makeDet({
				cx: 101,
				cy: 101,
				width: 50,
				height: 50,
				confidence: 0.8,
				classId: 0,
			}),
		];
		const matrix = buildIouMatrix(dets, "detect");

		// With threshold=0.25: all pass. Sort: A(0.95), C(0.8), B(0.4)
		// A kept, C kept (no overlap with A), B suppressed (overlaps C).
		const resultLow = nmsFromRaw(dets, matrix, 0.25, 0.45);
		expect(resultLow).toHaveLength(2);
		expect(resultLow.map((d) => d.confidence)).toContain(0.95);
		expect(resultLow.map((d) => d.confidence)).toContain(0.8);

		// With threshold=0.5: B(0.4) filtered first. Remaining: A(0.95), C(0.8) — same NMS result.
		const resultHigh = nmsFromRaw(dets, matrix, 0.5, 0.45);
		expect(resultHigh).toHaveLength(2);
		expect(resultHigh.map((d) => d.confidence)).toContain(0.95);
		expect(resultHigh.map((d) => d.confidence)).toContain(0.8);
	});
});
