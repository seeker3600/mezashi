import { describe, expect, it } from "vitest";
import { CLASS_NAMES, CONFIDENCE_THRESHOLD, MODEL_INPUT_SIZE } from "../labels";

describe("labels", () => {
	it("should have 15 class names matching DOTA dataset", () => {
		expect(CLASS_NAMES).toHaveLength(15);
	});

	it("should include expected class names", () => {
		expect(CLASS_NAMES).toContain("plane");
		expect(CLASS_NAMES).toContain("ship");
		expect(CLASS_NAMES).toContain("small vehicle");
		expect(CLASS_NAMES).toContain("swimming pool");
	});

	it("MODEL_INPUT_SIZE should be 1024", () => {
		expect(MODEL_INPUT_SIZE).toBe(1024);
	});

	it("CONFIDENCE_THRESHOLD should be a valid probability", () => {
		expect(CONFIDENCE_THRESHOLD).toBeGreaterThan(0);
		expect(CONFIDENCE_THRESHOLD).toBeLessThan(1);
	});
});
