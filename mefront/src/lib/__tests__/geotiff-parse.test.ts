import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fromArrayBuffer } from "geotiff";
import { beforeAll, describe, expect, it } from "vitest";
import { parseGeoTIFF } from "../geotiff";

// jsdom does not provide ImageData; polyfill for testing
beforeAll(() => {
	if (typeof globalThis.ImageData === "undefined") {
		(globalThis as any).ImageData = class ImageData {
			data: Uint8ClampedArray;
			width: number;
			height: number;
			constructor(data: Uint8ClampedArray, width: number, height: number) {
				this.data = data;
				this.width = width;
				this.height = height;
			}
		};
	}
});

/**
 * parseGeoTIFF の内部ロジックを直接テストする。
 * jsdom の File.arrayBuffer() 非対応を回避するため、
 * ArrayBuffer を直接 geotiff に渡して解析フローの問題を再現する。
 */
describe("parseGeoTIFF with real Sentinel-2 file", () => {
	const filePath = resolve(__dirname, "../../../samples/Sentinel-2.tiff");

	it("should parse the sample GeoTIFF via geotiff library without throwing", async () => {
		const buf = readFileSync(filePath);
		const arrayBuffer = buf.buffer.slice(
			buf.byteOffset,
			buf.byteOffset + buf.byteLength,
		);

		const tiff = await fromArrayBuffer(arrayBuffer);
		const image = await tiff.getImage();

		const width = image.getWidth();
		const height = image.getHeight();
		expect(width).toBeGreaterThan(0);
		expect(height).toBeGreaterThan(0);

		// This is where parseGeoTIFF reads raster data
		const rasters = await image.readRasters();
		const numBands = rasters.length;
		expect(numBands).toBeGreaterThan(0);

		// Build RGBA same as parseGeoTIFF does
		const rgba = new Uint8ClampedArray(width * height * 4);
		for (let i = 0; i < width * height; i++) {
			rgba[i * 4] = (rasters[0] as any)[i]; // R
			rgba[i * 4 + 1] =
				numBands >= 3 ? (rasters[1] as any)[i] : (rasters[0] as any)[i];
			rgba[i * 4 + 2] =
				numBands >= 3 ? (rasters[2] as any)[i] : (rasters[0] as any)[i];
			rgba[i * 4 + 3] = 255;
		}

		// Try to access tiePoints and fileDir same as parseGeoTIFF
		const tiePoints = await image.getTiePoints();
		const fileDir = image.getFileDirectory();
		const pixelScaleRaw = fileDir.getValue("ModelPixelScale");
		const geoKeys = image.getGeoKeys();

		expect(tiePoints).toBeDefined();
		expect(fileDir).toBeDefined();
	});

	it("should parse via parseGeoTIFF function end-to-end", async () => {
		const buf = readFileSync(filePath);
		// Polyfill File.arrayBuffer for jsdom
		const file = new File([buf], "Sentinel-2.tiff");
		if (!file.arrayBuffer) {
			(file as any).arrayBuffer = () =>
				Promise.resolve(
					buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
				);
		}

		const { imageData, meta } = await parseGeoTIFF(file);

		expect(imageData.width).toBe(1793);
		expect(imageData.height).toBe(2222);
		expect(imageData.data.length).toBe(1793 * 2222 * 4);
		expect(meta).toBeDefined();
		expect(meta.tiePoint).toBeDefined();
		expect(meta.pixelScale).toBeDefined();
		expect(meta.epsg).not.toBeUndefined();
	});
});
