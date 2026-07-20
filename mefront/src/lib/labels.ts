/** Confidence threshold for detections */
export const CONFIDENCE_THRESHOLD = 0.25;

/** Minimum confidence threshold for detections */
export const CONFIDENCE_THRESHOLD_MIN = 0.05;

/** Default URL of the model metadata JSON file */
export const DEFAULT_METADATA_URL = "/models/yolo26-obb.json";

/** Overlap between adjacent tiles (fraction) */
export const TILE_OVERLAP = 0.5;

/** IoU threshold for non-maximum suppression */
export const NMS_IOU_THRESHOLD = 0.45;

/** Available marker styles for OBB direction displays */
export const DIRECTION_MARKER_STYLES = {
	ARROW: "arrow",
	LINE: "line",
} as const;

export type DirectionMarkerStyle =
	(typeof DIRECTION_MARKER_STYLES)[keyof typeof DIRECTION_MARKER_STYLES];

/** Marker style used by detail and overview direction displays */
export const DIRECTION_MARKER_STYLE: DirectionMarkerStyle =
	DIRECTION_MARKER_STYLES.ARROW;

/** Minimum on-screen shaft length before an individual direction is readable */
export const DIRECTION_DETAIL_MIN_SHAFT_LENGTH = 14;

/** Proportion of detections that should be readable before using detail mode */
export const DIRECTION_DETAIL_TARGET_RATIO = 0.8;

/** Scale margin around the mode threshold to prevent display flicker */
export const DIRECTION_MODE_HYSTERESIS = 0.1;

/** Target on-screen spacing between direction markers in overview mode */
export const DIRECTION_OVERVIEW_GRID_SPACING = 56;
