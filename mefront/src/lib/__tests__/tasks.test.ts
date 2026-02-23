import { describe, expect, it } from "vitest";
import type { TaskHandler } from "../tasks";
import { getTaskHandler } from "../tasks";

describe("getTaskHandler", () => {
	it("should return a handler for 'obb'", () => {
		const handler = getTaskHandler("obb");
		expect(handler).toBeDefined();
		expect(handler.columnsPerDetection).toBe(7);
	});

	it("should return a handler for 'detect'", () => {
		const handler = getTaskHandler("detect");
		expect(handler).toBeDefined();
		expect(handler.columnsPerDetection).toBe(6);
	});

	it("should throw for unknown task", () => {
		expect(() => getTaskHandler("unknown" as "obb")).toThrow(
			"未知のタスクタイプです",
		);
	});
});

describe("obb handler", () => {
	const handler: TaskHandler = getTaskHandler("obb");
	const labels = ["plane", "ship"];

	it("should parse a 7-column detection with angle", () => {
		// [cx, cy, w, h, confidence, classId, angle]
		const data = new Float32Array([100, 200, 40, 20, 0.9, 0, 0.5]);
		const det = handler.parseDetection(data, 0, labels);
		if (det == null) throw new Error("expected detection");
		expect(det.cx).toBeCloseTo(100);
		expect(det.cy).toBeCloseTo(200);
		expect(det.width).toBeCloseTo(40);
		expect(det.height).toBeCloseTo(20);
		expect(det.confidence).toBeCloseTo(0.9);
		expect(det.classId).toBe(0);
		expect(det.className).toBe("plane");
		expect(det.angle).toBeCloseTo(0.5);
	});

	it("should return null for low-confidence detection", () => {
		const data = new Float32Array([100, 200, 40, 20, 0.01, 0, 0.5]);
		const det = handler.parseDetection(data, 0, labels);
		expect(det).toBeNull();
	});

	it("should fallback className when classId is out of bounds", () => {
		const data = new Float32Array([0, 0, 10, 10, 0.9, 99, 0]);
		const det = handler.parseDetection(data, 0, labels);
		if (det == null) throw new Error("expected detection");
		expect(det.className).toBe("class_99");
	});

	it("nms should suppress overlapping same-class detections", () => {
		const dets = [
			makeDet({
				cx: 100,
				cy: 100,
				width: 40,
				height: 40,
				confidence: 0.9,
				classId: 0,
			}),
			makeDet({
				cx: 102,
				cy: 102,
				width: 40,
				height: 40,
				confidence: 0.8,
				classId: 0,
			}),
		];
		const result = handler.nms(dets, 0.45);
		expect(result).toHaveLength(1);
		expect(result[0].confidence).toBe(0.9);
	});

	it("nms should keep different-class detections", () => {
		const dets = [
			makeDet({
				cx: 100,
				cy: 100,
				width: 40,
				height: 40,
				confidence: 0.9,
				classId: 0,
			}),
			makeDet({
				cx: 102,
				cy: 102,
				width: 40,
				height: 40,
				confidence: 0.8,
				classId: 1,
				className: "ship",
			}),
		];
		const result = handler.nms(dets, 0.45);
		expect(result).toHaveLength(2);
	});
});

describe("detect handler", () => {
	const handler: TaskHandler = getTaskHandler("detect");
	const labels = ["car", "person"];

	it("should parse a 6-column detection with angle = 0", () => {
		// [cx, cy, w, h, confidence, classId]
		const data = new Float32Array([50, 60, 30, 40, 0.85, 1]);
		const det = handler.parseDetection(data, 0, labels);
		if (det == null) throw new Error("expected detection");
		expect(det.cx).toBeCloseTo(50);
		expect(det.cy).toBeCloseTo(60);
		expect(det.width).toBeCloseTo(30);
		expect(det.height).toBeCloseTo(40);
		expect(det.confidence).toBeCloseTo(0.85);
		expect(det.classId).toBe(1);
		expect(det.className).toBe("person");
		expect(det.angle).toBe(0);
	});

	it("should return null for low-confidence detection", () => {
		const data = new Float32Array([50, 60, 30, 40, 0.01, 1]);
		const det = handler.parseDetection(data, 0, labels);
		expect(det).toBeNull();
	});

	it("nms should use exact AABB IoU (not max-dimension approximation)", () => {
		// Two boxes: same center, one tall+thin, one short+wide — low AABB IoU
		const dets = [
			makeDet({
				cx: 100,
				cy: 100,
				width: 10,
				height: 100,
				confidence: 0.9,
				classId: 0,
			}),
			makeDet({
				cx: 100,
				cy: 100,
				width: 100,
				height: 10,
				confidence: 0.8,
				classId: 0,
			}),
		];
		const result = handler.nms(dets, 0.45);
		// AABB IoU = (10*10) / (10*100 + 100*10 - 100) = 100/1900 ≈ 0.053 — both kept
		expect(result).toHaveLength(2);
	});

	it("nms should suppress nearly identical boxes", () => {
		const dets = [
			makeDet({
				cx: 100,
				cy: 100,
				width: 50,
				height: 50,
				confidence: 0.9,
				classId: 0,
			}),
			makeDet({
				cx: 101,
				cy: 101,
				width: 50,
				height: 50,
				confidence: 0.8,
				classId: 0,
			}),
		];
		const result = handler.nms(dets, 0.45);
		expect(result).toHaveLength(1);
	});
});

function makeDet(
	overrides: Partial<import("../types").Detection> = {},
): import("../types").Detection {
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
