import { pixelToGeo } from "./geotiff";
import { getOBBCorners } from "./obbUtils";
import type { Detection, GeoTIFFMeta } from "./types";

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
			const geoCorners = corners.map(([px, py]) => pixelToGeo(px, py, meta));
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
): Detection[] {
	// First, convert all detections to geo coordinates for comparison
	const geoDetections1 = detections1.map((d) => ({
		detection: d,
		geoCorners: getOBBCorners(d).map(([px, py]) => pixelToGeo(px, py, meta1)),
	}));

	const geoDetections2 = detections2.map((d) => ({
		detection: d,
		geoCorners: getOBBCorners(d).map(([px, py]) => pixelToGeo(px, py, meta2)),
	}));

	// Start with all detections from first image
	const merged: Detection[] = [...detections1];

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
					merged[j] = detections2[i];
				}
				break;
			}
		}

		// If not a duplicate, add to merged results
		if (!isDuplicate) {
			merged.push(detections2[i]);
		}
	}

	return merged;
}

/**
 * Calculate IoU (Intersection over Union) for two polygons using their corner points.
 * This is an approximation using axis-aligned bounding boxes in geo coordinates.
 */
function computePolygonIoU(
	corners1: { x: number; y: number }[],
	corners2: { x: number; y: number }[],
): number {
	// Get axis-aligned bounding boxes
	const box1 = getAABB(corners1);
	const box2 = getAABB(corners2);

	// Calculate intersection
	const ix1 = Math.max(box1.minX, box2.minX);
	const iy1 = Math.max(box1.minY, box2.minY);
	const ix2 = Math.min(box1.maxX, box2.maxX);
	const iy2 = Math.min(box1.maxY, box2.maxY);

	const iw = Math.max(0, ix2 - ix1);
	const ih = Math.max(0, iy2 - iy1);
	const intersection = iw * ih;

	// Calculate union
	const area1 = (box1.maxX - box1.minX) * (box1.maxY - box1.minY);
	const area2 = (box2.maxX - box2.minX) * (box2.maxY - box2.minY);
	const union = area1 + area2 - intersection;

	return union > 0 ? intersection / union : 0;
}

/**
 * Get axis-aligned bounding box from polygon corners.
 */
function getAABB(corners: { x: number; y: number }[]): {
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
} {
	let minX = Number.POSITIVE_INFINITY;
	let minY = Number.POSITIVE_INFINITY;
	let maxX = Number.NEGATIVE_INFINITY;
	let maxY = Number.NEGATIVE_INFINITY;

	for (const corner of corners) {
		minX = Math.min(minX, corner.x);
		minY = Math.min(minY, corner.y);
		maxX = Math.max(maxX, corner.x);
		maxY = Math.max(maxY, corner.y);
	}

	return { minX, minY, maxX, maxY };
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
