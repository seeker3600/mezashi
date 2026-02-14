import * as ort from "onnxruntime-web";
import { prepareTile } from "./imageUtils";
import { CLASS_NAMES, CONFIDENCE_THRESHOLD, MODEL_INPUT_SIZE } from "./labels";
import type { Detection } from "./types";

// Use WASM backend (works in all browsers, no WebGL/WebGPU required)
ort.env.wasm.numThreads = 1;

let sessionPromise: Promise<ort.InferenceSession> | null = null;

/**
 * Load (or return cached) ONNX inference session.
 */
export function loadModel(): Promise<ort.InferenceSession> {
	if (!sessionPromise) {
		sessionPromise = ort.InferenceSession.create("/models/yolo26n-obb.onnx", {
			executionProviders: ["wasm"],
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
): Promise<Detection[]> {
	const tensor = new ort.Tensor("float32", inputData, [
		1,
		3,
		MODEL_INPUT_SIZE,
		MODEL_INPUT_SIZE,
	]);
	const results = await session.run({ images: tensor });
	const output = results.output0;
	if (!output) return [];

	const data = output.data as Float32Array;
	const numDetections = output.dims[1];
	const detections: Detection[] = [];

	for (let i = 0; i < numDetections; i++) {
		const offset = i * 7;
		const cx = data[offset];
		const cy = data[offset + 1];
		const w = data[offset + 2];
		const h = data[offset + 3];
		const angle = data[offset + 4];
		const confidence = data[offset + 5];
		const classId = data[offset + 6];

		if (confidence < CONFIDENCE_THRESHOLD) continue;

		detections.push({
			classId,
			className: CLASS_NAMES[classId] ?? `class_${classId}`,
			confidence,
			cx,
			cy,
			width: w,
			height: h,
			angle,
		});
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
 * NMS for oriented bounding boxes: approximate using axis-aligned IoU of the enclosing rectangle.
 */
function nmsOBB(detections: Detection[], iouThreshold: number): Detection[] {
	if (detections.length === 0) return [];

	// Sort by confidence descending
	const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
	const keep: Detection[] = [];
	const suppressed = new Set<number>();

	for (let i = 0; i < sorted.length; i++) {
		if (suppressed.has(i)) continue;
		keep.push(sorted[i]);
		for (let j = i + 1; j < sorted.length; j++) {
			if (suppressed.has(j)) continue;
			if (sorted[i].classId !== sorted[j].classId) continue;
			if (computeAABBIoU(sorted[i], sorted[j]) > iouThreshold) {
				suppressed.add(j);
			}
		}
	}
	return keep;
}

function computeAABBIoU(a: Detection, b: Detection): number {
	// Use the max dimension to approximate axis-aligned bbox
	const aHalfW = Math.max(a.width, a.height) / 2;
	const aHalfH = aHalfW;
	const bHalfW = Math.max(b.width, b.height) / 2;
	const bHalfH = bHalfW;

	const ax1 = a.cx - aHalfW;
	const ay1 = a.cy - aHalfH;
	const ax2 = a.cx + aHalfW;
	const ay2 = a.cy + aHalfH;
	const bx1 = b.cx - bHalfW;
	const by1 = b.cy - bHalfH;
	const bx2 = b.cx + bHalfW;
	const by2 = b.cy + bHalfH;

	const ix1 = Math.max(ax1, bx1);
	const iy1 = Math.max(ay1, by1);
	const ix2 = Math.min(ax2, bx2);
	const iy2 = Math.min(ay2, by2);

	const iw = Math.max(0, ix2 - ix1);
	const ih = Math.max(0, iy2 - iy1);
	const intersection = iw * ih;

	const aArea = (ax2 - ax1) * (ay2 - ay1);
	const bArea = (bx2 - bx1) * (by2 - by1);
	const union = aArea + bArea - intersection;

	return union > 0 ? intersection / union : 0;
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
	onProgress?: (done: number, total: number) => void,
): Promise<Detection[]> {
	const session = await loadModel();

	// Decide whether to use slice inference
	if (imgWidth <= SLICE_THRESHOLD && imgHeight <= SLICE_THRESHOLD) {
		// Single pass
		const { input, scale, padX, padY } = prepareTile(
			img,
			0,
			0,
			imgWidth,
			imgHeight,
			MODEL_INPUT_SIZE,
		);
		onProgress?.(0, 1);
		const dets = await runTile(session, input);
		onProgress?.(1, 1);
		return mapDetectionsToOriginal(dets, scale, padX, padY, 0, 0);
	}

	// Slice inference for large images
	const tileSize = MODEL_INPUT_SIZE;
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
				MODEL_INPUT_SIZE,
			);

			const tileDets = await runTile(session, input);
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
	return nmsOBB(allDetections, 0.45);
}
