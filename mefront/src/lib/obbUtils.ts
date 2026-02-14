import type { Detection } from "./types";

/**
 * Calculate the 4 corner points of an oriented bounding box.
 */
export function getOBBCorners(d: Detection): [number, number][] {
	const cos = Math.cos(d.angle);
	const sin = Math.sin(d.angle);
	const hw = d.width / 2;
	const hh = d.height / 2;

	// Corners relative to center, then rotate
	const corners: [number, number][] = [
		[-hw, -hh],
		[hw, -hh],
		[hw, hh],
		[-hw, hh],
	];

	return corners.map(([dx, dy]) => [
		d.cx + dx * cos - dy * sin,
		d.cy + dx * sin + dy * cos,
	]);
}
