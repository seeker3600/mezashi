import { useMemo } from "react";
import { mergeDetectionSets } from "../lib/exportResults";
import { NMS_IOU_THRESHOLD } from "../lib/labels";
import { buildIouMatrix, nmsFromRaw } from "../lib/nms";
import type { DetectionSet, GeoTIFFMeta } from "../lib/types";

interface DetectionResults {
	/** Detections to draw on the canvas (last image only, filtered by threshold). */
	displayDetections: import("../lib/types").Detection[];
	/** Detections for export (merged across all GeoTIFFs if applicable, filtered). */
	exportDetections: import("../lib/types").Detection[];
	/** Whether the current image is a GeoTIFF. */
	isGeoTIFF: boolean;
	/** Geo metadata for the current/merged result. */
	geoMeta: GeoTIFFMeta | undefined;
	/** Whether detections were merged from multiple GeoTIFFs. */
	isMerged: boolean;
}

/**
 * Derive display / export detections from accumulated detection sets.
 *
 * - Display: only the last detection set (what is shown on the canvas).
 * - Export: if multiple GeoTIFF sets exist, merge them; otherwise same as display.
 *
 * NMS is re-applied whenever the confidence threshold changes, using a
 * pre-computed IoU matrix so pairwise IoU values are not recalculated.
 */
export function useDetectionResults(
	detectionSets: DetectionSet[],
	confidenceThreshold: number,
): DetectionResults {
	// Merge is potentially expensive – re-run only when the sets change.
	const mergedResult = useMemo(
		() =>
			detectionSets.length >= 2 ? mergeDetectionSets(detectionSets) : null,
		[detectionSets],
	);

	// Pre-compute IoU matrix for the last detection set.
	// This is the expensive part; it runs only when the set of detections changes.
	const lastIouMatrix = useMemo(() => {
		if (detectionSets.length === 0) return null;
		const last = detectionSets[detectionSets.length - 1];
		return buildIouMatrix(last.detections, last.task);
	}, [detectionSets]);

	// Pre-compute IoU matrix for the merged result.
	const mergedIouMatrix = useMemo(() => {
		if (!mergedResult || detectionSets.length === 0) return null;
		return buildIouMatrix(mergedResult.detections, detectionSets[0].task);
	}, [mergedResult, detectionSets]);

	return useMemo(() => {
		if (detectionSets.length === 0) {
			return {
				displayDetections: [],
				exportDetections: [],
				isGeoTIFF: false,
				geoMeta: undefined,
				isMerged: false,
			};
		}

		const last = detectionSets[detectionSets.length - 1];

		// Re-run NMS on the confidence-filtered subset using the cached IoU matrix.
		const display = lastIouMatrix
			? nmsFromRaw(
					last.detections,
					lastIouMatrix,
					confidenceThreshold,
					NMS_IOU_THRESHOLD,
				)
			: last.detections.filter((d) => d.confidence >= confidenceThreshold);

		const isMerged = mergedResult != null;
		const exportDets =
			isMerged && mergedIouMatrix
				? nmsFromRaw(
						mergedResult.detections,
						mergedIouMatrix,
						confidenceThreshold,
						NMS_IOU_THRESHOLD,
					)
				: isMerged && mergedResult
					? mergedResult.detections.filter(
							(d) => d.confidence >= confidenceThreshold,
						)
					: display;

		return {
			displayDetections: display,
			exportDetections: exportDets,
			isGeoTIFF: last.isGeoTIFF,
			geoMeta: isMerged ? mergedResult.meta : last.geoMeta,
			isMerged,
		};
	}, [
		detectionSets,
		confidenceThreshold,
		mergedResult,
		lastIouMatrix,
		mergedIouMatrix,
	]);
}
