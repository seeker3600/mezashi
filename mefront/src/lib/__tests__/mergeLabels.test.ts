import { describe, expect, it } from "vitest";
import { applyLabelMerge, buildMergeMap } from "../mergeLabels";
import type { Detection, ModelMetadata } from "../types";

function makeMetadata(overrides: Partial<ModelMetadata> = {}): ModelMetadata {
	return {
		name: "TestModel",
		task: "obb",
		onnxUrl: "/models/test.onnx",
		inputSize: 640,
		labels: ["large vehicle", "small vehicle", "plane"],
		license: { name: "MIT" },
		...overrides,
	};
}

function makeDet(overrides: Partial<Detection> = {}): Detection {
	return {
		classId: 0,
		className: "large vehicle",
		confidence: 0.9,
		cx: 100,
		cy: 100,
		width: 40,
		height: 20,
		angle: 0,
		...overrides,
	};
}

describe("buildMergeMap", () => {
	it("should return null when metadata has no merge field", () => {
		const metadata = makeMetadata();
		expect(buildMergeMap(metadata)).toBeNull();
	});

	it("should return null when merge is empty object", () => {
		const metadata = makeMetadata({ merge: {} });
		expect(buildMergeMap(metadata)).toBeNull();
	});

	it("should return null when merge labels don't match any labels", () => {
		const metadata = makeMetadata({
			merge: { vehicle: ["truck", "bus"] },
		});
		expect(buildMergeMap(metadata)).toBeNull();
	});

	it("should build a merge map for matching labels", () => {
		const metadata = makeMetadata({
			merge: { vehicle: ["large vehicle", "small vehicle"] },
		});
		const map = buildMergeMap(metadata);
		expect(map).not.toBeNull();
		// "large vehicle" is index 0, "small vehicle" is index 1
		// merged classId = min(0, 1) = 0
		expect(map?.get(0)).toEqual({ classId: 0, className: "vehicle" });
		expect(map?.get(1)).toEqual({ classId: 0, className: "vehicle" });
		// "plane" (index 2) should not be in the map
		expect(map?.has(2)).toBe(false);
	});

	it("should use the minimum classId as the merged classId", () => {
		const metadata = makeMetadata({
			labels: ["plane", "large vehicle", "small vehicle"],
			merge: { vehicle: ["large vehicle", "small vehicle"] },
		});
		const map = buildMergeMap(metadata);
		expect(map).not.toBeNull();
		// "large vehicle" is index 1, "small vehicle" is index 2
		// merged classId = min(1, 2) = 1
		expect(map?.get(1)).toEqual({ classId: 1, className: "vehicle" });
		expect(map?.get(2)).toEqual({ classId: 1, className: "vehicle" });
	});

	it("should handle multiple merge groups", () => {
		const metadata = makeMetadata({
			labels: ["large vehicle", "small vehicle", "fighter", "bomber"],
			merge: {
				vehicle: ["large vehicle", "small vehicle"],
				aircraft: ["fighter", "bomber"],
			},
		});
		const map = buildMergeMap(metadata);
		expect(map).not.toBeNull();
		expect(map?.get(0)).toEqual({ classId: 0, className: "vehicle" });
		expect(map?.get(1)).toEqual({ classId: 0, className: "vehicle" });
		expect(map?.get(2)).toEqual({ classId: 2, className: "aircraft" });
		expect(map?.get(3)).toEqual({ classId: 2, className: "aircraft" });
	});

	it("should skip source labels not found in labels array", () => {
		const metadata = makeMetadata({
			merge: { vehicle: ["large vehicle", "unknown"] },
		});
		const map = buildMergeMap(metadata);
		expect(map).not.toBeNull();
		// Only "large vehicle" (index 0) matches
		expect(map?.get(0)).toEqual({ classId: 0, className: "vehicle" });
		expect(map?.size).toBe(1);
	});
});

describe("applyLabelMerge", () => {
	const mergeMetadata = makeMetadata({
		merge: { vehicle: ["large vehicle", "small vehicle"] },
	});

	function getMergeMap() {
		const map = buildMergeMap(mergeMetadata);
		if (!map) throw new Error("expected merge map");
		return map;
	}

	it("should remap classId and className for merged detections", () => {
		const mergeMap = getMergeMap();

		const detections = [
			makeDet({ classId: 0, className: "large vehicle" }),
			makeDet({ classId: 1, className: "small vehicle" }),
			makeDet({ classId: 2, className: "plane" }),
		];

		const result = applyLabelMerge(detections, mergeMap);

		expect(result[0].classId).toBe(0);
		expect(result[0].className).toBe("vehicle");
		expect(result[1].classId).toBe(0);
		expect(result[1].className).toBe("vehicle");
		// "plane" should be unchanged
		expect(result[2].classId).toBe(2);
		expect(result[2].className).toBe("plane");
	});

	it("should not mutate original detections", () => {
		const mergeMap = getMergeMap();

		const original = makeDet({ classId: 0, className: "large vehicle" });
		const detections = [original];
		applyLabelMerge(detections, mergeMap);

		expect(original.classId).toBe(0);
		expect(original.className).toBe("large vehicle");
	});

	it("should preserve other detection properties", () => {
		const mergeMap = getMergeMap();

		const det = makeDet({
			classId: 1,
			className: "small vehicle",
			confidence: 0.85,
			cx: 200,
			cy: 300,
			width: 60,
			height: 30,
			angle: 0.5,
		});
		const result = applyLabelMerge([det], mergeMap);

		expect(result[0].confidence).toBe(0.85);
		expect(result[0].cx).toBe(200);
		expect(result[0].cy).toBe(300);
		expect(result[0].width).toBe(60);
		expect(result[0].height).toBe(30);
		expect(result[0].angle).toBe(0.5);
	});
});
