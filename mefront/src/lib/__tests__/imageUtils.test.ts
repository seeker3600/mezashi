import { describe, expect, it } from "vitest";
import { computeGeoTIFFShrinkScale } from "../imageUtils";

describe("computeGeoTIFFShrinkScale", () => {
	it("should return 1.0 when image GSD equals targetGSD", () => {
		const scale = computeGeoTIFFShrinkScale({ x: 1.0, y: 1.0 }, 1.0);
		expect(scale).toBe(1.0);
	});

	it("should upscale when image GSD is coarser than targetGSD", () => {
		// 10m/px image, model expects 1m/px → scale = 10/1 = 10.0
		const scale = computeGeoTIFFShrinkScale({ x: 10, y: 10 }, 1.0);
		expect(scale).toBe(10.0);
	});

	it("should downscale when image GSD is finer than targetGSD", () => {
		// 0.3m/px image, model expects 1.0m/px → scale = 0.3/1.0 = 0.3
		const scale = computeGeoTIFFShrinkScale({ x: 0.3, y: 0.3 }, 1.0);
		expect(scale).toBeCloseTo(0.3);
	});

	it("should downscale for very high resolution imagery", () => {
		// 0.1m/px image, model expects 1.0m/px → scale = 0.1/1.0 = 0.1
		const scale = computeGeoTIFFShrinkScale({ x: 0.1, y: 0.1 }, 1.0);
		expect(scale).toBeCloseTo(0.1);
	});

	it("should use the larger pixelScale dimension for GSD calculation", () => {
		// x=0.1, y=0.4 → currentGSD=0.4, model expects 1.0 → scale=0.4
		const scale = computeGeoTIFFShrinkScale({ x: 0.1, y: 0.4 }, 1.0);
		expect(scale).toBeCloseTo(0.4);
	});

	it("should return 1.0 when GSD exactly matches targetGSD (0.5m)", () => {
		const scale = computeGeoTIFFShrinkScale({ x: 0.5, y: 0.5 }, 0.5);
		expect(scale).toBe(1.0);
	});

	it("should downscale for sub-metre imagery with 0.3m target", () => {
		// 0.1m/px image, model expects 0.3m/px → scale = 0.1/0.3 ≈ 0.333
		const scale = computeGeoTIFFShrinkScale({ x: 0.1, y: 0.1 }, 0.3);
		expect(scale).toBeCloseTo(1 / 3);
	});

	it("should not downscale when image GSD equals targetGSD (0.3m)", () => {
		const scale = computeGeoTIFFShrinkScale({ x: 0.3, y: 0.3 }, 0.3);
		expect(scale).toBe(1.0);
	});
});
