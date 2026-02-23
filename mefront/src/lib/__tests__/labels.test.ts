import { describe, expect, it } from "vitest";
import { CONFIDENCE_THRESHOLD } from "../labels";

describe("labels", () => {
	it("CONFIDENCE_THRESHOLD should be a valid probability", () => {
		expect(CONFIDENCE_THRESHOLD).toBeGreaterThan(0);
		expect(CONFIDENCE_THRESHOLD).toBeLessThan(1);
	});
});
