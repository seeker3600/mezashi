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
