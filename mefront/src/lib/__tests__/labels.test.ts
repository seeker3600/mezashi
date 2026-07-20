import { describe, expect, it } from "vitest";
import {
	CONFIDENCE_THRESHOLD,
	DIRECTION_DETAIL_MIN_SHAFT_LENGTH,
	DIRECTION_DETAIL_TARGET_RATIO,
	DIRECTION_MARKER_STYLE,
	DIRECTION_MARKER_STYLES,
	DIRECTION_MODE_HYSTERESIS,
	DIRECTION_OVERVIEW_GRID_SPACING,
} from "../labels";

describe("labels", () => {
	it("CONFIDENCE_THRESHOLD should be a valid probability", () => {
		expect(CONFIDENCE_THRESHOLD).toBeGreaterThan(0);
		expect(CONFIDENCE_THRESHOLD).toBeLessThan(1);
	});

	it("should default direction markers to arrows", () => {
		expect(DIRECTION_MARKER_STYLES).toEqual({
			ARROW: "arrow",
			LINE: "line",
		});
		expect(DIRECTION_MARKER_STYLE).toBe(DIRECTION_MARKER_STYLES.ARROW);
	});

	it("should provide valid direction display settings", () => {
		expect(DIRECTION_DETAIL_MIN_SHAFT_LENGTH).toBeGreaterThan(0);
		expect(DIRECTION_DETAIL_TARGET_RATIO).toBeGreaterThan(0);
		expect(DIRECTION_DETAIL_TARGET_RATIO).toBeLessThanOrEqual(1);
		expect(DIRECTION_MODE_HYSTERESIS).toBeGreaterThan(0);
		expect(DIRECTION_MODE_HYSTERESIS).toBeLessThan(1);
		expect(DIRECTION_OVERVIEW_GRID_SPACING).toBeGreaterThan(0);
	});
});
