import {
	DIRECTION_DETAIL_MIN_SHAFT_LENGTH,
	DIRECTION_DETAIL_TARGET_RATIO,
	DIRECTION_MODE_HYSTERESIS,
} from "./labels";
import { getOBBLongAxisAngle } from "./obbUtils";
import type { Detection } from "./types";

export type DirectionDisplayMode = "detail" | "overview";

export interface DirectionGroup {
	x: number;
	y: number;
	angle: number;
	count: number;
}

const DETAIL_SHAFT_AXIS_RATIO = 0.4;
const MIN_SCALE = 1;
const MAX_SCALE = 10;

/**
 * Return the zoom scale at which the configured proportion of OBB directions
 * has a readable on-screen shaft length.
 */
export function getDetailScaleThreshold(
	detections: Detection[],
	fitScale: number,
): number {
	if (detections.length === 0 || fitScale <= 0) return MIN_SCALE;

	const requiredScales = detections
		.map((detection) => {
			const shaftLength =
				Math.max(detection.width, detection.height) *
				fitScale *
				DETAIL_SHAFT_AXIS_RATIO;
			return DIRECTION_DETAIL_MIN_SHAFT_LENGTH / shaftLength;
		})
		.sort((first, second) => first - second);
	const percentileIndex = Math.max(
		0,
		Math.ceil(requiredScales.length * DIRECTION_DETAIL_TARGET_RATIO) - 1,
	);

	return Math.min(
		MAX_SCALE,
		Math.max(MIN_SCALE, requiredScales[percentileIndex]),
	);
}

/**
 * Select detail or overview mode with hysteresis around the detail threshold.
 */
export function getDirectionDisplayMode(
	scale: number,
	threshold: number,
	previousMode: DirectionDisplayMode,
): DirectionDisplayMode {
	if (scale >= threshold * (1 + DIRECTION_MODE_HYSTERESIS)) {
		return "detail";
	}
	if (scale < threshold * (1 - DIRECTION_MODE_HYSTERESIS)) {
		return "overview";
	}
	return previousMode;
}

/**
 * Group OBB long-axis orientations into image-coordinate grid cells.
 */
export function aggregateDirectionsByGrid(
	detections: Detection[],
	gridSize: number,
): DirectionGroup[] {
	if (gridSize <= 0) return [];

	const groups = new Map<
		string,
		{
			gridX: number;
			gridY: number;
			sineSum: number;
			cosineSum: number;
			count: number;
		}
	>();

	for (const detection of detections) {
		const gridX = Math.floor(detection.cx / gridSize);
		const gridY = Math.floor(detection.cy / gridSize);
		const key = `${gridX}:${gridY}`;
		const existing = groups.get(key);
		const angle = getOBBLongAxisAngle(detection) * 2;
		const group = existing ?? {
			gridX,
			gridY,
			sineSum: 0,
			cosineSum: 0,
			count: 0,
		};

		group.sineSum += Math.sin(angle);
		group.cosineSum += Math.cos(angle);
		group.count += 1;
		groups.set(key, group);
	}

	return [...groups.values()].map((group) => ({
		x: (group.gridX + 0.5) * gridSize,
		y: (group.gridY + 0.5) * gridSize,
		angle: Math.atan2(group.sineSum, group.cosineSum) / 2,
		count: group.count,
	}));
}
