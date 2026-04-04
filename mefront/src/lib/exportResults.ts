import { pixelToGeo } from "./geotiff";
import { convexPolygonIoU, getOBBCorners } from "./obbUtils";
import type { Detection, DetectionSet, GeoTIFFMeta } from "./types";

/**
 * Trigger a file download in the browser.
 */
function downloadBlob(blob: Blob, filename: string): void {
	const url = URL.createObjectURL(blob);
	const a = document.createElement("a");
	a.href = url;
	a.download = filename;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	URL.revokeObjectURL(url);
}

/**
 * Build a JSON result for regular images (pixel coordinates).
 */
export function buildPixelResultJSON(
	detections: Detection[],
	imageWidth: number,
	imageHeight: number,
): object {
	return {
		imageWidth,
		imageHeight,
		detections: detections.map((d) => ({
			class: d.className,
			classId: d.classId,
			confidence: Math.round(d.confidence * 1000) / 1000,
			bbox: {
				cx: Math.round(d.cx * 10) / 10,
				cy: Math.round(d.cy * 10) / 10,
				width: Math.round(d.width * 10) / 10,
				height: Math.round(d.height * 10) / 10,
				angle: Math.round(d.angle * 1000) / 1000,
			},
			corners: getOBBCorners(d).map(([x, y]) => [
				Math.round(x * 10) / 10,
				Math.round(y * 10) / 10,
			]),
		})),
	};
}

/**
 * Download detection results as JSON (for normal images).
 */
export function downloadResultJSON(
	detections: Detection[],
	imageWidth: number,
	imageHeight: number,
): void {
	const result = buildPixelResultJSON(detections, imageWidth, imageHeight);
	const json = JSON.stringify(result, null, 2);
	const blob = new Blob([json], { type: "application/json" });
	downloadBlob(blob, "detections.json");
}

/**
 * Build a GeoJSON FeatureCollection for a single class.
 */
export function buildGeoJSONForClass(
	detections: Detection[],
	className: string,
	meta: GeoTIFFMeta,
): object {
	const features = detections
		.filter((d) => d.className === className)
		.map((d) => {
			const corners = getOBBCorners(d);
			// Use detection's own geoMeta if available, otherwise use the provided meta
			const detectionMeta = d.geoMeta ?? meta;
			const geoCorners = corners.map(([px, py]) =>
				pixelToGeo(px, py, detectionMeta),
			);
			// Close the ring for GeoJSON polygon
			const ring = [
				...geoCorners.map((c) => [c.x, c.y]),
				[geoCorners[0].x, geoCorners[0].y],
			];

			return {
				type: "Feature" as const,
				properties: {
					class: d.className,
					classId: d.classId,
					confidence: Math.round(d.confidence * 1000) / 1000,
				},
				geometry: {
					type: "Polygon" as const,
					coordinates: [ring],
				},
			};
		});

	return {
		type: "FeatureCollection",
		...(meta.epsg != null
			? {
					crs: {
						type: "name",
						properties: { name: `urn:ogc:def:crs:EPSG::${meta.epsg}` },
					},
				}
			: {}),
		features,
	};
}

/**
 * Download GeoJSON files (one per class) for GeoTIFF results.
 */
export function downloadGeoJSON(
	detections: Detection[],
	meta: GeoTIFFMeta,
): void {
	// Get unique class names that have detections
	const classNames = [...new Set(detections.map((d) => d.className))];

	for (const className of classNames) {
		const geojson = buildGeoJSONForClass(detections, className, meta);
		const json = JSON.stringify(geojson, null, 2);
		const blob = new Blob([json], { type: "application/geo+json" });
		const safeName = className.replace(/\s+/g, "_");
		downloadBlob(blob, `${safeName}.geojson`);
	}
}

/**
 * Merge detections from two GeoTIFF images and remove duplicates.
 * Uses IoU (Intersection over Union) in geographic coordinates to identify duplicates.
 */
export function mergeGeoTIFFDetections(
	detections1: Detection[],
	meta1: GeoTIFFMeta,
	detections2: Detection[],
	meta2: GeoTIFFMeta,
	iouThreshold = 0.5,
): (Detection & { geoMeta: GeoTIFFMeta })[] {
	// First, convert all detections to geo coordinates for comparison
	// Use each detection's own geoMeta if it has one, otherwise use meta1/meta2
	const geoDetections1 = detections1.map((d) => ({
		detection: d,
		geoCorners: getOBBCorners(d).map(([px, py]) =>
			pixelToGeo(px, py, d.geoMeta ?? meta1),
		),
	}));

	const geoDetections2 = detections2.map((d) => ({
		detection: d,
		geoCorners: getOBBCorners(d).map(([px, py]) =>
			pixelToGeo(px, py, d.geoMeta ?? meta2),
		),
	}));

	// Start with all detections from first image, preserving their geoMeta if they have one
	const merged: (Detection & { geoMeta: GeoTIFFMeta })[] = detections1.map(
		(d) => ({
			...d,
			geoMeta: d.geoMeta ?? meta1,
		}),
	);

	// Check each detection from second image against first image detections
	for (let i = 0; i < geoDetections2.length; i++) {
		let isDuplicate = false;

		for (let j = 0; j < geoDetections1.length; j++) {
			// Only compare same class
			if (
				geoDetections2[i].detection.classId !==
				geoDetections1[j].detection.classId
			) {
				continue;
			}

			// Calculate IoU in geo coordinates
			const iou = computePolygonIoU(
				geoDetections2[i].geoCorners,
				geoDetections1[j].geoCorners,
			);

			if (iou > iouThreshold) {
				isDuplicate = true;
				// Keep the detection with higher confidence
				if (
					geoDetections2[i].detection.confidence >
					geoDetections1[j].detection.confidence
				) {
					// Replace with detection from image 2, preserving its geoMeta
					merged[j] = { ...detections2[i], geoMeta: meta2 };
				}
				break;
			}
		}

		// If not a duplicate, add to merged results with its geoMeta
		if (!isDuplicate) {
			merged.push({ ...detections2[i], geoMeta: meta2 });
		}
	}

	return merged;
}

/**
 * Calculate IoU (Intersection over Union) for two polygons using their corner points
 * and the Sutherland-Hodgman polygon clipping algorithm.
 */
function computePolygonIoU(
	corners1: { x: number; y: number }[],
	corners2: { x: number; y: number }[],
): number {
	const a: [number, number][] = corners1.map((c) => [c.x, c.y]);
	const b: [number, number][] = corners2.map((c) => [c.x, c.y]);
	return convexPolygonIoU(a, b);
}

/**
 * Download merged GeoJSON from two GeoTIFF detections.
 */
export function downloadMergedGeoJSON(
	detections: Detection[],
	meta: GeoTIFFMeta,
): void {
	// Get unique class names that have detections
	const classNames = [...new Set(detections.map((d) => d.className))];

	for (const className of classNames) {
		const geojson = buildGeoJSONForClass(detections, className, meta);
		const json = JSON.stringify(geojson, null, 2);
		const blob = new Blob([json], { type: "application/geo+json" });
		const safeName = className.replace(/\s+/g, "_");
		downloadBlob(blob, `merged_${safeName}.geojson`);
	}
}

/**
 * Merge detections from N GeoTIFF detection sets, removing duplicates.
 * Iteratively merges each set into the accumulated result.
 * Returns the merged detections and the GeoTIFFMeta of the last set.
 */
export function mergeDetectionSets(
	sets: DetectionSet[],
	iouThreshold = 0.5,
): { detections: Detection[]; meta: GeoTIFFMeta } | null {
	const geoSets = sets.filter(
		(s): s is DetectionSet & { geoMeta: GeoTIFFMeta } =>
			s.isGeoTIFF && s.geoMeta != null,
	);
	if (geoSets.length === 0) return null;
	if (geoSets.length === 1) {
		return { detections: geoSets[0].detections, meta: geoSets[0].geoMeta };
	}

	// Start with first image's detections, ensuring they have their geoMeta
	let merged = geoSets[0].detections.map((d) => ({
		...d,
		geoMeta: d.geoMeta ?? geoSets[0].geoMeta,
	}));
	let mergedMeta = geoSets[0].geoMeta;

	for (let i = 1; i < geoSets.length; i++) {
		merged = mergeGeoTIFFDetections(
			merged,
			mergedMeta,
			geoSets[i].detections,
			geoSets[i].geoMeta,
			iouThreshold,
		);
		mergedMeta = geoSets[i].geoMeta;
	}

	return { detections: merged, meta: mergedMeta };
}
