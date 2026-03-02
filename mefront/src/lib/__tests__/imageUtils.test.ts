import { describe, expect, it } from "vitest";
import { computeGeoTIFFShrinkScale } from "../imageUtils";
import { MIN_GSD_METERS } from "../labels";

describe("computeGeoTIFFShrinkScale", () => {
	const threshold = 1280;

	it("should return 1.0 when image fits within threshold", () => {
		const scale = computeGeoTIFFShrinkScale(
			1000,
			800,
			{ x: 0.1, y: 0.1 },
			threshold,
		);
		expect(scale).toBe(1.0);
	});

	it("should return 1.0 when image exactly matches threshold", () => {
		const scale = computeGeoTIFFShrinkScale(
			1280,
			1000,
			{ x: 0.1, y: 0.1 },
			threshold,
		);
		expect(scale).toBe(1.0);
	});

	it("should shrink to fit threshold when GSD allows it", () => {
		// 5cm/px image, 5120x5120 → fitScale = 1280/5120 = 0.25
		// minScale = 0.05 / 0.5 = 0.1
		// max(0.25, 0.1) = 0.25 → fits in threshold
		const scale = computeGeoTIFFShrinkScale(
			5120,
			5120,
			{ x: 0.05, y: 0.05 },
			threshold,
		);
		expect(scale).toBeCloseTo(0.25);
	});

	it("should clamp to MIN_GSD when shrinking to threshold would exceed 50cm/px", () => {
		// 30cm/px image, 10000x10000
		// fitScale = 1280/10000 = 0.128
		// minScale = 0.3 / 0.5 = 0.6
		// max(0.128, 0.6) = 0.6 → clamped by GSD constraint
		const scale = computeGeoTIFFShrinkScale(
			10000,
			10000,
			{ x: 0.3, y: 0.3 },
			threshold,
		);
		expect(scale).toBeCloseTo(0.6);
	});

	it("should return 1.0 when GSD is already at 50cm/px", () => {
		const scale = computeGeoTIFFShrinkScale(
			5000,
			5000,
			{ x: 0.5, y: 0.5 },
			threshold,
		);
		expect(scale).toBe(1.0);
	});

	it("should return 1.0 when GSD is coarser than 50cm/px (e.g. Sentinel-2 at 10m)", () => {
		const scale = computeGeoTIFFShrinkScale(
			10980,
			10980,
			{ x: 10, y: 10 },
			threshold,
		);
		expect(scale).toBe(1.0);
	});

	it("should use the larger pixelScale dimension for GSD calculation", () => {
		// x=0.1, y=0.4 → currentGSD = 0.4
		// 5000x5000, fitScale = 1280/5000 = 0.256
		// minScale = 0.4 / 0.5 = 0.8
		// max(0.256, 0.8) = 0.8
		const scale = computeGeoTIFFShrinkScale(
			5000,
			5000,
			{ x: 0.1, y: 0.4 },
			threshold,
		);
		expect(scale).toBeCloseTo(0.8);
	});

	it("should use the larger image dimension for fit calculation", () => {
		// 2560x1000 → maxDim=2560, fitScale=1280/2560=0.5
		// GSD=0.1, minScale=0.1/0.5=0.2
		// max(0.5, 0.2)=0.5
		const scale = computeGeoTIFFShrinkScale(
			2560,
			1000,
			{ x: 0.1, y: 0.1 },
			threshold,
		);
		expect(scale).toBeCloseTo(0.5);
	});

	it("should handle very high resolution imagery (1cm/px)", () => {
		// 1cm/px, 50000x50000
		// fitScale = 1280/50000 = 0.0256
		// minScale = 0.01 / 0.5 = 0.02
		// max(0.0256, 0.02) = 0.0256 → fits
		const scale = computeGeoTIFFShrinkScale(
			50000,
			50000,
			{ x: 0.01, y: 0.01 },
			threshold,
		);
		expect(scale).toBeCloseTo(0.0256);
	});

	it("should handle very high resolution with large image where GSD is the constraint", () => {
		// 1cm/px, 100000x100000
		// fitScale = 1280/100000 = 0.0128
		// minScale = 0.01 / 0.5 = 0.02
		// max(0.0128, 0.02) = 0.02 → GSD constrained, will still need tiling
		const scale = computeGeoTIFFShrinkScale(
			100000,
			100000,
			{ x: 0.01, y: 0.01 },
			threshold,
		);
		expect(scale).toBeCloseTo(0.02);
	});

	it("ensures resulting GSD does not exceed MIN_GSD_METERS", () => {
		const pixelScale = { x: 0.2, y: 0.2 };
		const scale = computeGeoTIFFShrinkScale(
			10000,
			10000,
			pixelScale,
			threshold,
		);
		const resultingGSD = pixelScale.x / scale;
		expect(resultingGSD).toBeLessThanOrEqual(MIN_GSD_METERS + 1e-9);
	});
});
