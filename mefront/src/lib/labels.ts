/** DOTA class labels matching the ONNX model metadata */
export const CLASS_NAMES: readonly string[] = [
	"plane",
	"ship",
	"storage tank",
	"baseball diamond",
	"tennis court",
	"basketball court",
	"ground track field",
	"harbor",
	"bridge",
	"large vehicle",
	"small vehicle",
	"helicopter",
	"roundabout",
	"soccer ball field",
	"swimming pool",
] as const;

/** Model input image size (width and height) */
export const MODEL_INPUT_SIZE = 1024;

/** Confidence threshold for detections */
export const CONFIDENCE_THRESHOLD = 0.25;
