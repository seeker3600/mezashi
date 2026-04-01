import type { Detection } from "./types";

type Point = [number, number];

/**
 * Calculate the 4 corner points of an oriented bounding box.
 */
export function getOBBCorners(d: Detection): Point[] {
	const cos = Math.cos(d.angle);
	const sin = Math.sin(d.angle);
	const hw = d.width / 2;
	const hh = d.height / 2;

	// Corners relative to center, then rotate
	const corners: Point[] = [
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

// ---------------------------------------------------------------------------
// Convex polygon intersection & IoU
// ---------------------------------------------------------------------------

/**
 * Compute the area of a simple polygon using the Shoelace formula.
 */
export function polygonArea(vertices: Point[]): number {
	const n = vertices.length;
	if (n < 3) return 0;

	let area = 0;
	for (let i = 0; i < n; i++) {
		const j = (i + 1) % n;
		area += vertices[i][0] * vertices[j][1];
		area -= vertices[j][0] * vertices[i][1];
	}
	return Math.abs(area) / 2;
}

/**
 * Compute the intersection polygon of two convex polygons
 * using the Sutherland-Hodgman clipping algorithm.
 *
 * Both polygons must be convex and share the same winding order.
 * The returned polygon (if non-empty) preserves that winding.
 */
export function intersectConvexPolygons(
	subject: Point[],
	clip: Point[],
): Point[] {
	if (subject.length < 3 || clip.length < 3) return [];

	// Ensure both polygons are in counter-clockwise order (positive signed area)
	const subjectCCW = ensureCCW(subject);
	const clipCCW = ensureCCW(clip);

	let output: Point[] = [...subjectCCW];

	for (let i = 0; i < clipCCW.length; i++) {
		if (output.length === 0) return [];

		const input = output;
		output = [];

		const edgeStart = clipCCW[i];
		const edgeEnd = clipCCW[(i + 1) % clipCCW.length];

		for (let j = 0; j < input.length; j++) {
			const current = input[j];
			const previous = input[(j + input.length - 1) % input.length];

			const currInside = isInsideEdge(current, edgeStart, edgeEnd);
			const prevInside = isInsideEdge(previous, edgeStart, edgeEnd);

			if (currInside) {
				if (!prevInside) {
					const p = lineIntersection(previous, current, edgeStart, edgeEnd);
					if (p) output.push(p);
				}
				output.push(current);
			} else if (prevInside) {
				const p = lineIntersection(previous, current, edgeStart, edgeEnd);
				if (p) output.push(p);
			}
		}
	}

	return output;
}

/**
 * Compute IoU (Intersection over Union) for two convex polygons.
 */
export function convexPolygonIoU(a: Point[], b: Point[]): number {
	const intersection = intersectConvexPolygons(a, b);
	const interArea = polygonArea(intersection);
	if (interArea === 0) return 0;

	const aArea = polygonArea(a);
	const bArea = polygonArea(b);
	const union = aArea + bArea - interArea;
	return union > 0 ? interArea / union : 0;
}

/** Signed area — positive for CCW, negative for CW. */
function signedArea(vertices: Point[]): number {
	const n = vertices.length;
	if (n < 3) return 0;

	let area = 0;
	for (let i = 0; i < n; i++) {
		const j = (i + 1) % n;
		area += vertices[i][0] * vertices[j][1];
		area -= vertices[j][0] * vertices[i][1];
	}
	return area / 2;
}

/** Reverse winding if CW so that the polygon is CCW (positive signed area). */
function ensureCCW(polygon: Point[]): Point[] {
	return signedArea(polygon) < 0 ? [...polygon].reverse() : polygon;
}

/** Return true when `point` is on the *left* (inside) of directed edge start→end. */
function isInsideEdge(point: Point, start: Point, end: Point): boolean {
	return (
		(end[0] - start[0]) * (point[1] - start[1]) -
			(end[1] - start[1]) * (point[0] - start[0]) >=
		0
	);
}

/** Intersection point of two infinite lines (p1–p2) and (p3–p4). */
function lineIntersection(
	p1: Point,
	p2: Point,
	p3: Point,
	p4: Point,
): Point | null {
	const denom =
		(p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0]);
	if (Math.abs(denom) < 1e-10) return null;

	const t =
		((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) /
		denom;

	return [p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])];
}
