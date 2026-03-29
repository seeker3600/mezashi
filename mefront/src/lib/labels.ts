/** Confidence threshold for detections */
export const CONFIDENCE_THRESHOLD = 0.25;

/** Minimum confidence threshold for detections */
export const CONFIDENCE_THRESHOLD_MIN = 0.05;

/** Default URL of the model metadata JSON file */
export const DEFAULT_METADATA_URL = "/models/yolo26n-obb.json";

/**
 * GeoTIFF 画像の縮小時に許容する最小 GSD (Ground Sample Distance)。
 * 1ピクセルあたり 0.5m (50cm) を下限とする。
 */
export const MIN_GSD_METERS = 0.5;

/** Threshold for when to use slice inference (pixels) */
export const SLICE_THRESHOLD = 640;

/** Overlap between adjacent tiles (fraction) */
export const TILE_OVERLAP = 0.5;

/** IoU threshold for non-maximum suppression */
export const NMS_IOU_THRESHOLD = 0.45;
