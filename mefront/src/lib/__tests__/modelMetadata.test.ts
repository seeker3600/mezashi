import { describe, expect, it } from "vitest";
import { validateModelMetadata } from "../modelMetadata";

function validMetadata(overrides: Record<string, unknown> = {}) {
	return {
		name: "TestModel",
		task: "obb",
		onnxUrl: "/models/test.onnx",
		inputSize: 640,
		labels: ["plane"],
		license: { name: "MIT" },
		...overrides,
	};
}

describe("validateModelMetadata", () => {
	it("should accept valid metadata with task=obb", () => {
		expect(() => validateModelMetadata(validMetadata())).not.toThrow();
	});

	it("should accept valid metadata with task=detect", () => {
		expect(() =>
			validateModelMetadata(validMetadata({ task: "detect" })),
		).not.toThrow();
	});

	it("should reject missing task", () => {
		const { task: _, ...noTask } = validMetadata();
		expect(() => validateModelMetadata(noTask)).toThrow('"task"');
	});

	it("should reject invalid task value", () => {
		expect(() =>
			validateModelMetadata(validMetadata({ task: "segment" })),
		).toThrow('"task"');
	});

	it("should reject non-string task", () => {
		expect(() => validateModelMetadata(validMetadata({ task: 123 }))).toThrow(
			'"task"',
		);
	});

	it("should still reject missing name", () => {
		expect(() => validateModelMetadata(validMetadata({ name: "" }))).toThrow(
			'"name"',
		);
	});

	it("should still reject missing onnxUrl", () => {
		expect(() => validateModelMetadata(validMetadata({ onnxUrl: "" }))).toThrow(
			'"onnxUrl"',
		);
	});

	it("should accept metadata with valid merge field", () => {
		expect(() =>
			validateModelMetadata(
				validMetadata({
					labels: ["large vehicle", "small vehicle"],
					merge: { vehicle: ["large vehicle", "small vehicle"] },
				}),
			),
		).not.toThrow();
	});

	it("should accept metadata without merge field", () => {
		expect(() => validateModelMetadata(validMetadata())).not.toThrow();
	});

	it("should reject merge with non-object value", () => {
		expect(() =>
			validateModelMetadata(validMetadata({ merge: "invalid" })),
		).toThrow('"merge"');
	});

	it("should reject merge with empty array value", () => {
		expect(() =>
			validateModelMetadata(validMetadata({ merge: { vehicle: [] } })),
		).toThrow('"merge.vehicle"');
	});

	it("should reject merge with non-string array elements", () => {
		expect(() =>
			validateModelMetadata(validMetadata({ merge: { vehicle: [123] } })),
		).toThrow('"merge.vehicle"');
	});

	it("should reject merge with label not in labels array", () => {
		expect(() =>
			validateModelMetadata(validMetadata({ merge: { vehicle: ["unknown"] } })),
		).toThrow('"unknown"');
	});
});
