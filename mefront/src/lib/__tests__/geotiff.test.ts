import { describe, expect, it } from "vitest";
import { isGeoTIFFFile, pixelToGeo } from "../geotiff";
import type { GeoTIFFMeta } from "../types";

describe("isGeoTIFFFile", () => {
	it("should return true for .tif files", () => {
		const file = new File([], "image.tif");
		expect(isGeoTIFFFile(file)).toBe(true);
	});

	it("should return true for .tiff files", () => {
		const file = new File([], "image.tiff");
		expect(isGeoTIFFFile(file)).toBe(true);
	});

	it("should return true for .TIF files (case insensitive)", () => {
		const file = new File([], "IMAGE.TIF");
		expect(isGeoTIFFFile(file)).toBe(true);
	});

	it("should return false for .jpg files", () => {
		const file = new File([], "photo.jpg");
		expect(isGeoTIFFFile(file)).toBe(false);
	});

	it("should return false for .png files", () => {
		const file = new File([], "photo.png");
		expect(isGeoTIFFFile(file)).toBe(false);
	});
});

describe("pixelToGeo", () => {
	it("should transform pixel (0,0) to tie point", () => {
		const meta: GeoTIFFMeta = {
			tiePoint: { x: 500000, y: 4000000 },
			pixelScale: { x: 0.5, y: 0.5 },
			epsg: 32654,
		};
		const result = pixelToGeo(0, 0, meta);
		expect(result.x).toBe(500000);
		expect(result.y).toBe(4000000);
	});

	it("should correctly transform a pixel offset", () => {
		const meta: GeoTIFFMeta = {
			tiePoint: { x: 100, y: 200 },
			pixelScale: { x: 1, y: 1 },
			epsg: 4326,
		};
		const result = pixelToGeo(10, 20, meta);
		expect(result.x).toBe(110); // 100 + 10*1
		expect(result.y).toBe(180); // 200 - 20*1 (y decreases with pixel row)
	});

	it("should handle non-integer pixel scales", () => {
		const meta: GeoTIFFMeta = {
			tiePoint: { x: 0, y: 0 },
			pixelScale: { x: 0.25, y: 0.25 },
			epsg: null,
		};
		const result = pixelToGeo(4, 4, meta);
		expect(result.x).toBeCloseTo(1.0);
		expect(result.y).toBeCloseTo(-1.0);
	});
});
