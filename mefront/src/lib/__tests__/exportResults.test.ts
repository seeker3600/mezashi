import { describe, expect, it } from "vitest";
import {
	buildGeoJSONForClass,
	buildPixelResultJSON,
	mergeGeoTIFFDetections,
} from "../exportResults";
import type { Detection, GeoTIFFMeta } from "../types";

function makeDetection(overrides: Partial<Detection> = {}): Detection {
	return {
		classId: 0,
		className: "plane",
		confidence: 0.95,
		cx: 100,
		cy: 200,
		width: 40,
		height: 20,
		angle: 0,
		...overrides,
	};
}

describe("buildPixelResultJSON", () => {
	it("should return correct structure", () => {
		const dets = [makeDetection()];
		const result = buildPixelResultJSON(dets, 800, 600) as {
			imageWidth: number;
			imageHeight: number;
			detections: Array<{
				class: string;
				classId: number;
				confidence: number;
				bbox: {
					cx: number;
					cy: number;
					width: number;
					height: number;
					angle: number;
				};
				corners: [number, number][];
			}>;
		};
		expect(result.imageWidth).toBe(800);
		expect(result.imageHeight).toBe(600);
		expect(result.detections).toHaveLength(1);
		expect(result.detections[0].class).toBe("plane");
		expect(result.detections[0].corners).toHaveLength(4);
	});

	it("should handle empty detections", () => {
		const result = buildPixelResultJSON([], 800, 600) as {
			detections: unknown[];
		};
		expect(result.detections).toHaveLength(0);
	});
});

describe("buildGeoJSONForClass", () => {
	const meta: GeoTIFFMeta = {
		tiePoint: { x: 0, y: 100 },
		pixelScale: { x: 1, y: 1 },
		epsg: 32654,
	};

	it("should create a valid GeoJSON FeatureCollection", () => {
		const dets = [makeDetection({ className: "ship", classId: 1 })];
		const geojson = buildGeoJSONForClass(dets, "ship", meta) as {
			type: string;
			crs: { properties: { name: string } };
			features: Array<{
				type: string;
				geometry: {
					type: string;
					coordinates: number[][][];
				};
				properties: { class: string };
			}>;
		};

		expect(geojson.type).toBe("FeatureCollection");
		expect(geojson.features).toHaveLength(1);
		expect(geojson.features[0].type).toBe("Feature");
		expect(geojson.features[0].geometry.type).toBe("Polygon");
		expect(geojson.features[0].properties.class).toBe("ship");
	});

	it("should include CRS when EPSG is available", () => {
		const dets = [makeDetection()];
		const geojson = buildGeoJSONForClass(dets, "plane", meta) as {
			crs: { properties: { name: string } };
		};
		expect(geojson.crs.properties.name).toContain("EPSG::32654");
	});

	it("should not include CRS when EPSG is null", () => {
		const metaNoEpsg: GeoTIFFMeta = { ...meta, epsg: null };
		const dets = [makeDetection()];
		const geojson = buildGeoJSONForClass(dets, "plane", metaNoEpsg) as {
			crs?: unknown;
		};
		expect(geojson.crs).toBeUndefined();
	});

	it("should filter detections by class name", () => {
		const dets = [
			makeDetection({ className: "plane", classId: 0 }),
			makeDetection({ className: "ship", classId: 1 }),
		];
		const geojson = buildGeoJSONForClass(dets, "plane", meta) as {
			features: unknown[];
		};
		expect(geojson.features).toHaveLength(1);
	});

	it("should create closed polygon rings", () => {
		const dets = [makeDetection()];
		const geojson = buildGeoJSONForClass(dets, "plane", meta) as {
			features: Array<{
				geometry: { coordinates: number[][][] };
			}>;
		};
		const ring = geojson.features[0].geometry.coordinates[0];
		// Ring should be closed (first == last)
		expect(ring[ring.length - 1]).toEqual(ring[0]);
		// Should have 5 points (4 corners + closing)
		expect(ring).toHaveLength(5);
	});
});

describe("mergeGeoTIFFDetections", () => {
	const meta1: GeoTIFFMeta = {
		tiePoint: { x: 0, y: 100 },
		pixelScale: { x: 1, y: 1 },
		epsg: 32654,
	};

	const meta2: GeoTIFFMeta = {
		tiePoint: { x: 50, y: 100 },
		pixelScale: { x: 1, y: 1 },
		epsg: 32654,
	};

	it("should merge detections without duplicates when no overlap", () => {
		const dets1 = [makeDetection({ cx: 10, cy: 10, width: 10, height: 10 })];
		const dets2 = [makeDetection({ cx: 200, cy: 200, width: 10, height: 10 })];
		const merged = mergeGeoTIFFDetections(dets1, meta1, dets2, meta2);
		expect(merged).toHaveLength(2);
	});

	it("should remove duplicates when detections overlap significantly", () => {
		// Same location, same metadata - should be considered duplicate
		const dets1 = [
			makeDetection({
				cx: 100,
				cy: 100,
				width: 40,
				height: 20,
				confidence: 0.9,
			}),
		];
		const dets2 = [
			makeDetection({
				cx: 50,
				cy: 100,
				width: 40,
				height: 20,
				confidence: 0.8,
			}),
		];
		// In geo coordinates: det1 at (100, 0), det2 at (100, 0) - same location
		const merged = mergeGeoTIFFDetections(dets1, meta1, dets2, meta2);
		expect(merged).toHaveLength(1);
		// Should keep the higher confidence detection
		expect(merged[0].confidence).toBe(0.9);
	});

	it("should replace lower confidence duplicate with higher confidence one", () => {
		const dets1 = [
			makeDetection({
				cx: 100,
				cy: 100,
				width: 40,
				height: 20,
				confidence: 0.8,
			}),
		];
		const dets2 = [
			makeDetection({
				cx: 50,
				cy: 100,
				width: 40,
				height: 20,
				confidence: 0.95,
			}),
		];
		const merged = mergeGeoTIFFDetections(dets1, meta1, dets2, meta2);
		expect(merged).toHaveLength(1);
		expect(merged[0].confidence).toBe(0.95);
	});

	it("should only compare detections of the same class", () => {
		const dets1 = [
			makeDetection({ className: "plane", classId: 0, cx: 100, cy: 100 }),
		];
		const dets2 = [
			makeDetection({ className: "ship", classId: 1, cx: 50, cy: 100 }),
		];
		const merged = mergeGeoTIFFDetections(dets1, meta1, dets2, meta2);
		// Different classes, should not be considered duplicates
		expect(merged).toHaveLength(2);
	});

	it("should handle empty detection arrays", () => {
		const merged1 = mergeGeoTIFFDetections([], meta1, [], meta2);
		expect(merged1).toHaveLength(0);

		const dets = [makeDetection()];
		const merged2 = mergeGeoTIFFDetections(dets, meta1, [], meta2);
		expect(merged2).toHaveLength(1);

		const merged3 = mergeGeoTIFFDetections([], meta1, dets, meta2);
		expect(merged3).toHaveLength(1);
	});
});
