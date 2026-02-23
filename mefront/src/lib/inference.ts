import * as ort from "onnxruntime-web";
import { prepareTile } from "./imageUtils";
import type { TaskHandler } from "./tasks";
import { getTaskHandler } from "./tasks";
import type { Detection, ModelMetadata } from "./types";

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

/** Threshold for when to use slice inference (pixels) */
const SLICE_THRESHOLD = 1280;
/** Overlap between adjacent tiles (fraction) */
const TILE_OVERLAP = 0.25;

/**
 * Run full inference on an image element, using slice inference for large images.
 * Returns detections in original image pixel coordinates.
 */
export async function runInference(
	img: HTMLCanvasElement | HTMLImageElement,
	imgWidth: number,
	imgHeight: number,
	metadata: ModelMetadata,
	onProgress?: (done: number, total: number) => void,
): Promise<Detection[]> {
	const { onnxUrl, inputSize, labels, task } = metadata;
	const session = await loadModel(onnxUrl);
	const handler = getTaskHandler(task);

	// Decide whether to use slice inference
	if (imgWidth <= SLICE_THRESHOLD && imgHeight <= SLICE_THRESHOLD) {
		// Single pass
		const { input, scale, padX, padY } = prepareTile(
			img,
			0,
			0,
			imgWidth,
			imgHeight,
			inputSize,
		);
		onProgress?.(0, 1);
		const dets = await runTile(session, input, inputSize, labels, handler);
		onProgress?.(1, 1);
		return mapDetectionsToOriginal(dets, scale, padX, padY, 0, 0);
	}

	// Slice inference for large images
	const tileSize = inputSize;
	const stride = Math.round(tileSize * (1 - TILE_OVERLAP));

	const tilesX = Math.max(1, Math.ceil((imgWidth - tileSize) / stride) + 1);
	const tilesY = Math.max(1, Math.ceil((imgHeight - tileSize) / stride) + 1);
	const totalTiles = tilesX * tilesY;

	const allDetections: Detection[] = [];
	let done = 0;

	for (let ty = 0; ty < tilesY; ty++) {
		for (let tx = 0; tx < tilesX; tx++) {
			const sx = Math.min(tx * stride, imgWidth - tileSize);
			const sy = Math.min(ty * stride, imgHeight - tileSize);
			const sw = Math.min(tileSize, imgWidth - sx);
			const sh = Math.min(tileSize, imgHeight - sy);

			const { input, scale, padX, padY } = prepareTile(
				img,
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

	// Apply NMS across all tiles to remove duplicate detections in overlapping regions
	return handler.nms(allDetections, 0.45);
}
