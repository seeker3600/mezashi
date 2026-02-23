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
});
