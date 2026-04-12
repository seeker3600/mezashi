import { fromArrayBuffer } from "geotiff";
import type { GeoTIFFMeta } from "./types";

/**
 * Check if a file is a GeoTIFF by examining its extension.
 */
export function isGeoTIFFFile(file: File): boolean {
	const name = file.name.toLowerCase();
	return name.endsWith(".tif") || name.endsWith(".tiff");
}

/**
 * Parse a GeoTIFF file and return an ImageData-like object plus geo metadata.
 */
export async function parseGeoTIFF(file: File): Promise<{
	imageData: ImageData;
	meta: GeoTIFFMeta;
}> {
	const buffer = await file.arrayBuffer();
	const tiff = await fromArrayBuffer(buffer);
	const image = await tiff.getImage();

	const width = image.getWidth();
	const height = image.getHeight();

	// Read raster data
	const rasters = await image.readRasters();
	const numBands = rasters.length;

	// Build RGBA ImageData
	const rgba = new Uint8ClampedArray(width * height * 4);
	for (let i = 0; i < width * height; i++) {
		rgba[i * 4] = (rasters[0] as Uint8Array | Uint16Array | Float32Array)[i]; // R
		rgba[i * 4 + 1] =
			numBands >= 3
				? (rasters[1] as Uint8Array | Uint16Array | Float32Array)[i]
				: (rasters[0] as Uint8Array | Uint16Array | Float32Array)[i]; // G
		rgba[i * 4 + 2] =
			numBands >= 3
				? (rasters[2] as Uint8Array | Uint16Array | Float32Array)[i]
				: (rasters[0] as Uint8Array | Uint16Array | Float32Array)[i]; // B
		rgba[i * 4 + 3] = 255; // A
	}

	const imageData = new ImageData(rgba, width, height);

	// Extract geo metadata
	const tiePoints = await image.getTiePoints();
	const fileDir = image.getFileDirectory();
	const pixelScaleRaw = fileDir.getValue("ModelPixelScale") as
		| number[]
		| undefined;
	const geoKeys = image.getGeoKeys();

	const tiePoint =
		tiePoints.length > 0
			? {
					x: (tiePoints[0] as unknown as { x: number; y: number }).x,
					y: (tiePoints[0] as unknown as { x: number; y: number }).y,
				}
			: { x: 0, y: 0 };

	const scale = pixelScaleRaw
		? { x: pixelScaleRaw[0], y: pixelScaleRaw[1] }
		: { x: 1, y: 1 };

	// Try to get EPSG code
	const epsg =
		(geoKeys?.ProjectedCSTypeGeoKey as number | undefined) ??
		(geoKeys?.GeographicTypeGeoKey as number | undefined) ??
		null;

	// Geographic CRS (e.g. WGS84 / EPSG:4326) has no ProjectedCSTypeGeoKey;
	// its pixelScale is in degrees, not metres.
	const isGeographic =
		(geoKeys?.ProjectedCSTypeGeoKey as number | undefined) === undefined;

	const meta: GeoTIFFMeta = { tiePoint, pixelScale: scale, epsg, isGeographic };

	return { imageData, meta };
}

/**
 * Convert an ImageData to an HTMLCanvasElement for drawing.
 */
export function imageDataToCanvas(imageData: ImageData): HTMLCanvasElement {
	const canvas = document.createElement("canvas");
	canvas.width = imageData.width;
	canvas.height = imageData.height;
	const ctx = canvas.getContext("2d");
	if (!ctx) throw new Error("Cannot get 2d context");
	ctx.putImageData(imageData, 0, 0);
	return canvas;
}

/**
 * Convert pixel coordinates to geo coordinates using GeoTIFF metadata.
 */
export function pixelToGeo(
	px: number,
	py: number,
	meta: GeoTIFFMeta,
): { x: number; y: number } {
	return {
		x: meta.tiePoint.x + px * meta.pixelScale.x,
		y: meta.tiePoint.y - py * meta.pixelScale.y,
	};
}

/**
 * Metres per degree of latitude/longitude at the equator.
 * Used to convert geographic CRS pixel scales (degrees/px) to metres/px.
 */
const METRES_PER_DEGREE = 111_320;

/**
 * Return the pixel scale of a GeoTIFF in metres per pixel.
 * Geographic CRS (isGeographic=true) stores pixelScale in degrees; this
 * function converts those to metres using the equatorial approximation.
 * Projected CRS already stores metres so pixelScale is returned as-is.
 */
export function pixelScaleMeters(meta: GeoTIFFMeta): { x: number; y: number } {
	if (meta.isGeographic) {
		return {
			x: meta.pixelScale.x * METRES_PER_DEGREE,
			y: meta.pixelScale.y * METRES_PER_DEGREE,
		};
	}
	return meta.pixelScale;
}
