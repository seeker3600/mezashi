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
