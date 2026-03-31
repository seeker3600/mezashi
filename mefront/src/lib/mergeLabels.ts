import type { Detection, ModelMetadata } from "./types";

/** Mapping from an original classId to its merged classId and className. */
export interface MergeEntry {
	classId: number;
	className: string;
}

/**
 * Build a lookup map from original classId → merged { classId, className }.
 * Returns null when no merge rules exist or none match any labels.
 *
 * The merged classId is the minimum classId among the source labels so that
 * NMS treats all merged detections as the same class.
 */
export function buildMergeMap(
	metadata: ModelMetadata,
): Map<number, MergeEntry> | null {
	if (!metadata.merge) return null;

	const map = new Map<number, MergeEntry>();

	for (const [mergedName, sourceLabels] of Object.entries(metadata.merge)) {
		const sourceIds = sourceLabels
			.map((label) => metadata.labels.indexOf(label))
			.filter((id) => id >= 0);

		if (sourceIds.length === 0) continue;

		const mergedClassId = Math.min(...sourceIds);

		for (const id of sourceIds) {
			map.set(id, { classId: mergedClassId, className: mergedName });
		}
	}

	return map.size > 0 ? map : null;
}

/**
 * Apply label merge to detections: remap classId and className according to
 * the merge map so that NMS can suppress across the merged classes.
 */
export function applyLabelMerge(
	detections: Detection[],
	mergeMap: Map<number, MergeEntry>,
): Detection[] {
	return detections.map((d) => {
		const entry = mergeMap.get(d.classId);
		if (entry) {
			return { ...d, classId: entry.classId, className: entry.className };
		}
		return d;
	});
}
