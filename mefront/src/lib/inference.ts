import * as ort from "onnxruntime-web";
import { pixelScaleMeters } from "./geotiff";
import {
	computeGeoTIFFShrinkScale,
	createShrunkCanvas,
	prepareTile,
} from "./imageUtils";
import { TILE_OVERLAP } from "./labels";
import { applyLabelMerge, buildMergeMap } from "./mergeLabels";
import type { TaskHandler } from "./tasks";
import { getTaskHandler } from "./tasks";
import type { Detection, GeoTIFFMeta, ModelMetadata } from "./types";

// Use WASM backend (works in all browsers, no WebGL/WebGPU required)
ort.env.wasm.wasmPaths = import.meta.env.DEV
	? "/node_modules/onnxruntime-web/dist/"
	: "/";
ort.env.wasm.numThreads = 1;

let sessionPromise: Promise<ort.InferenceSession> | null = null;
let sessionUrl: string | null = null;

/**
 * Load (or return cached) ONNX inference session.
 * If the URL has changed since the last call, the session is recreated.
 */
export function loadModel(url: string): Promise<ort.InferenceSession> {
	if (!sessionPromise || sessionUrl !== url) {
		sessionUrl = url;
		sessionPromise = ort.InferenceSession.create(url, {
			executionProviders: ["webgpu", "webgl", "wasm"],
		});
	}
	return sessionPromise;
}

/**
 * Run inference on a single preprocessed tile.
 * Returns raw detections in tile-local pixel coordinates.
 */
async function runTile(
	session: ort.InferenceSession,
	inputData: Float32Array,
	inputSize: number,
	classNames: readonly string[],
	handler: TaskHandler,
): Promise<Detection[]> {
	const tensor = new ort.Tensor("float32", inputData, [
		1,
		3,
		inputSize,
		inputSize,
	]);
	const results = await session.run({ images: tensor });
	const output = results.output0;
	if (!output) return [];

	const data = output.data as Float32Array;
	const cols = handler.columnsPerDetection;
	const numDetections = output.dims[1];
	const detections: Detection[] = [];

	for (let i = 0; i < numDetections; i++) {
		const det = handler.parseDetection(data, i * cols, classNames);
		if (det) detections.push(det);
	}

	return detections;
}

/**
 * Map detections from tile-local model coordinates back to original image coordinates.
 */
function mapDetectionsToOriginal(
	detections: Detection[],
	scale: number,
	padX: number,
	padY: number,
	tileOffsetX: number,
	tileOffsetY: number,
): Detection[] {
	return detections.map((d) => ({
		...d,
		cx: (d.cx - padX) / scale + tileOffsetX,
		cy: (d.cy - padY) / scale + tileOffsetY,
		width: d.width / scale,
		height: d.height / scale,
	}));
}

/**
 * Run full inference on an image element, using slice inference for large images.
 * Returns detections in original image pixel coordinates.
 *
 * GSD が指定された場合、画像を expectedResolution に合わせて縮小する。
 * 縮小後もなお SLICE_THRESHOLD を超える場合はタイル化して推論する。
 */
export async function runInference(
	img: HTMLCanvasElement | HTMLImageElement,
	imgWidth: number,
	imgHeight: number,
	metadata: ModelMetadata,
	onProgress?: (done: number, total: number) => void,
	geoMeta?: GeoTIFFMeta,
	pixelSizeMeters?: number,
): Promise<Detection[]> {
	const { onnxUrl, inputSize, labels, task } = metadata;
	const session = await loadModel(onnxUrl);
	const handler = getTaskHandler(task);

	// GeoTIFF / 手動 GSD 指定: 大きい画像は GSD 制約の範囲内で縮小
	let src = img;
	let w = imgWidth;
	let h = imgHeight;
	let shrinkScale = 1.0;

	if (geoMeta) {
		shrinkScale = computeGeoTIFFShrinkScale(
			pixelScaleMeters(geoMeta),
			metadata.expectedResolution,
		);
	} else if (pixelSizeMeters !== undefined) {
		// 非 GeoTIFF で地上解像度が指定された場合も同じロジックで縮小
		shrinkScale = computeGeoTIFFShrinkScale(
			{ x: pixelSizeMeters, y: pixelSizeMeters },
			metadata.expectedResolution,
		);
	}

	if (shrinkScale !== 1.0) {
		if (shrinkScale < 1.0) {
			// Floor: shrunk image must be at least inputSize px in its larger dimension.
			// This guards against CRS unit mismatch (e.g. geographic GeoTIFFs where
			// pixelScale is in degrees rather than metres), which would otherwise
			// produce a near-zero scale and a 0-px canvas.
			shrinkScale = Math.max(
				shrinkScale,
				Math.min(1.0, inputSize / Math.max(w, h)),
			);
		}
		const rescaled = createShrunkCanvas(img, w, h, shrinkScale);
		src = rescaled;
		w = rescaled.width;
		h = rescaled.height;
	}

	let detections: Detection[];

	// Decide whether to use slice inference
	if (w <= inputSize && h <= inputSize) {
		// Single pass
		const { input, scale, padX, padY } = prepareTile(
			src,
			0,
			0,
			w,
			h,
			inputSize,
		);
		onProgress?.(0, 1);
		const dets = await runTile(session, input, inputSize, labels, handler);
		onProgress?.(1, 1);
		detections = mapDetectionsToOriginal(dets, scale, padX, padY, 0, 0);
	} else {
		// Slice inference for large images
		const tileSize = inputSize;
		const stride = Math.round(tileSize * (1 - TILE_OVERLAP));

		const tilesX = Math.max(1, Math.ceil((w - tileSize) / stride) + 1);
		const tilesY = Math.max(1, Math.ceil((h - tileSize) / stride) + 1);
		const totalTiles = tilesX * tilesY;

		const allDetections: Detection[] = [];
		let done = 0;

		for (let ty = 0; ty < tilesY; ty++) {
			for (let tx = 0; tx < tilesX; tx++) {
				const sx = Math.min(tx * stride, w - tileSize);
				const sy = Math.min(ty * stride, h - tileSize);
				const sw = Math.min(tileSize, w - sx);
				const sh = Math.min(tileSize, h - sy);

				const { input, scale, padX, padY } = prepareTile(
					src,
					sx,
					sy,
					sw,
					sh,
					inputSize,
				);

				const tileDets = await runTile(
					session,
					input,
					inputSize,
					labels,
					handler,
				);
				const mapped = mapDetectionsToOriginal(
					tileDets,
					scale,
					padX,
					padY,
					sx,
					sy,
				);
				allDetections.push(...mapped);

				done++;
				onProgress?.(done, totalTiles);
			}
		}

		detections = allDetections;
	}

	// Apply label merge if configured
	const mergeMap = buildMergeMap(metadata);
	if (mergeMap) {
		detections = applyLabelMerge(detections, mergeMap);
	}

	// 縮小した場合、検出座標を元の画像座標に変換
	if (shrinkScale !== 1.0) {
		for (const d of detections) {
			d.cx /= shrinkScale;
			d.cy /= shrinkScale;
			d.width /= shrinkScale;
			d.height /= shrinkScale;
		}
	}

	return detections;
}
