import { computeAABBIoU, computeOBBIoU } from "./tasks";
import type { Detection, DetectionTask } from "./types";

/**
 * Build a flattened N×N IoU matrix for the given detections.
 * `matrix[i * n + j]` = IoU between `detections[i]` and `detections[j]`.
 *
 * In class-aware mode, cross-class IoU is always 0.
 * The matrix is symmetric: `matrix[i * n + j] === matrix[j * n + i]`.
 */
export function buildIouMatrix(
	detections: Detection[],
	task: DetectionTask,
	classAgnostic = false,
): Float32Array {
	const n = detections.length;
	const iouFn = task === "obb" ? computeOBBIoU : computeAABBIoU;
	const matrix = new Float32Array(n * n);

	for (let i = 0; i < n; i++) {
		for (let j = i + 1; j < n; j++) {
			if (!classAgnostic && detections[i].classId !== detections[j].classId) {
				continue;
			}
			const iou = iouFn(detections[i], detections[j]);
			matrix[i * n + j] = iou;
			matrix[j * n + i] = iou;
		}
	}

	return matrix;
}

/**
 * Run NMS using a pre-computed IoU matrix.
 *
 * @param rawDetections - The full list of raw detections (same order as used in `buildIouMatrix`).
 * @param iouMatrix - N×N matrix from `buildIouMatrix(rawDetections, task)`.
 * @param confidenceThreshold - Minimum confidence for a detection to be considered.
 * @param iouThreshold - IoU threshold above which a lower-confidence detection is suppressed.
 * @returns Kept detections after filtering by confidence and applying NMS.
 */
export function nmsFromRaw(
	rawDetections: Detection[],
	iouMatrix: Float32Array,
	confidenceThreshold: number,
	iouThreshold: number,
): Detection[] {
	const n = rawDetections.length;
	if (n === 0) return [];

	// Filter by confidence while keeping the original indices for IoU lookup.
	const candidates: Array<{ origIdx: number; det: Detection }> = [];
	for (let i = 0; i < n; i++) {
		if (rawDetections[i].confidence >= confidenceThreshold) {
			candidates.push({ origIdx: i, det: rawDetections[i] });
		}
	}

	// Sort by confidence descending (highest first wins).
	candidates.sort((a, b) => b.det.confidence - a.det.confidence);

	const keep: Detection[] = [];
	const suppressed = new Set<number>(); // indices into `candidates`

	for (let i = 0; i < candidates.length; i++) {
		if (suppressed.has(i)) continue;
		keep.push(candidates[i].det);
		for (let j = i + 1; j < candidates.length; j++) {
			if (suppressed.has(j)) continue;
			const iou = iouMatrix[candidates[i].origIdx * n + candidates[j].origIdx];
			if (iou > iouThreshold) {
				suppressed.add(j);
			}
		}
	}

	return keep;
}
