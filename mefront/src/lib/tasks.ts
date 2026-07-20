import { CONFIDENCE_THRESHOLD_MIN } from "./labels";
import { convexPolygonIoU, getOBBCorners } from "./obbUtils";
import type { Detection, DetectionTask } from "./types";

/**
 * Strategy interface for task-specific detection processing.
 * Each task type (obb, detect, ...) implements this to handle
 * its own ONNX output format and NMS logic.
 */
export interface TaskHandler {
	/** Number of columns per detection in the ONNX output tensor */
	readonly columnsPerDetection: number;
	/** Parse a single detection from the raw ONNX output buffer */
	parseDetection(
		data: Float32Array,
		offset: number,
		labels: readonly string[],
	): Detection | null;
}

// ---------------------------------------------------------------------------
// Shared NMS helpers
// ---------------------------------------------------------------------------

export function computeAABBIoU(a: Detection, b: Detection): number {
	const aHalfW = a.width / 2;
	const aHalfH = a.height / 2;
	const bHalfW = b.width / 2;
	const bHalfH = b.height / 2;

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

/**
 * Compute IoU for two oriented bounding boxes using polygon intersection.
 * Uses the 4 corner vertices of each OBB and the Sutherland-Hodgman algorithm
 * to accurately account for rotation (angle).
 */
export function computeOBBIoU(a: Detection, b: Detection): number {
	return convexPolygonIoU(getOBBCorners(a), getOBBCorners(b));
}

// ---------------------------------------------------------------------------
// OBB task handler (YOLOv8-OBB: 7 columns)
// ---------------------------------------------------------------------------

const obbHandler: TaskHandler = {
	columnsPerDetection: 7,

	parseDetection(data, offset, labels) {
		const confidence = data[offset + 4];
		if (confidence < CONFIDENCE_THRESHOLD_MIN) return null;

		const classId = data[offset + 5];
		return {
			classId,
			className: labels[classId] ?? `class_${classId}`,
			confidence,
			cx: data[offset],
			cy: data[offset + 1],
			width: data[offset + 2],
			height: data[offset + 3],
			angle: data[offset + 6],
		};
	},
};

// ---------------------------------------------------------------------------
// Detect task handler (YOLO detect: 6 columns, no rotation)
// ---------------------------------------------------------------------------

const detectHandler: TaskHandler = {
	columnsPerDetection: 6,

	parseDetection(data, offset, labels) {
		const confidence = data[offset + 4];
		if (confidence < CONFIDENCE_THRESHOLD_MIN) return null;

		const classId = data[offset + 5];
		return {
			classId,
			className: labels[classId] ?? `class_${classId}`,
			confidence,
			cx: data[offset],
			cy: data[offset + 1],
			width: data[offset + 2],
			height: data[offset + 3],
			angle: 0,
		};
	},
};

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

const taskRegistry = new Map<DetectionTask, TaskHandler>([
	["obb", obbHandler],
	["detect", detectHandler],
]);

/**
 * Get the TaskHandler for a given detection task type.
 * Throws if the task is unknown (should never happen after metadata validation).
 */
export function getTaskHandler(task: DetectionTask): TaskHandler {
	const handler = taskRegistry.get(task);
	if (!handler) {
		throw new Error(`未知のタスクタイプです: "${task}"`);
	}
	return handler;
}
