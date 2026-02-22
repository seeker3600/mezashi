import { useMemo } from "react";
import { mergeDetectionSets } from "../lib/exportResults";
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

		const display = last.detections.filter(
			(d) => d.confidence >= confidenceThreshold,
		);

		const isMerged = mergedResult != null;
		const exportDets = isMerged
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
	}, [detectionSets, confidenceThreshold, mergedResult]);
}
