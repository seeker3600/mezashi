import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { fromArrayBuffer } from "geotiff";
import { beforeAll, describe, expect, it } from "vitest";
import { parseGeoTIFF } from "../geotiff";

const __dirname = dirname(fileURLToPath(import.meta.url));
type RasterBand = ArrayLike<number>;

// jsdom does not provide ImageData; polyfill for testing
beforeAll(() => {
	if (typeof globalThis.ImageData === "undefined") {
		Object.defineProperty(globalThis, "ImageData", {
			configurable: true,
			writable: true,
			value: class ImageData {
				data: Uint8ClampedArray;
				width: number;
				height: number;
				constructor(data: Uint8ClampedArray, width: number, height: number) {
					this.data = data;
					this.width = width;
					this.height = height;
				}
			},
		});
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
		const redBand = rasters[0] as RasterBand;
		const greenBand = (numBands >= 3 ? rasters[1] : rasters[0]) as RasterBand;
		const blueBand = (numBands >= 3 ? rasters[2] : rasters[0]) as RasterBand;

		// Build RGBA same as parseGeoTIFF does
		const rgba = new Uint8ClampedArray(width * height * 4);
		for (let i = 0; i < width * height; i++) {
			rgba[i * 4] = redBand[i] ?? 0; // R
			rgba[i * 4 + 1] = greenBand[i] ?? 0;
			rgba[i * 4 + 2] = blueBand[i] ?? 0;
			rgba[i * 4 + 3] = 255;
		}

		// Try to access tiePoints and fileDir same as parseGeoTIFF
		const tiePoints = await image.getTiePoints();
		const fileDir = image.getFileDirectory();

		expect(tiePoints).toBeDefined();
		expect(fileDir).toBeDefined();
	});

	it("should parse via parseGeoTIFF function end-to-end", async () => {
		const buf = readFileSync(filePath);
		// Polyfill File.arrayBuffer for jsdom
		const file = new File([buf], "Sentinel-2.tiff");
		if (!file.arrayBuffer) {
			Object.defineProperty(file, "arrayBuffer", {
				configurable: true,
				value: () =>
					Promise.resolve(
						buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
					),
			});
		}

		const { imageData, meta } = await parseGeoTIFF(file);

		expect(imageData.width).toBe(1793);
		expect(imageData.height).toBe(2222);
		expect(imageData.data.length).toBe(1793 * 2222 * 4);
		expect(meta).toBeDefined();
		expect(meta.tiePoint).toBeDefined();
		expect(meta.pixelScale).toBeDefined();
		expect(meta.epsg).not.toBeUndefined();
		// Sentinel-2 downloaded from Copernicus Browser is typically WGS84 (geographic)
		expect(meta.isGeographic).toBe(true);
	});
});
