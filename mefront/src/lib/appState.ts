import { CONFIDENCE_THRESHOLD, DEFAULT_MODEL_URL } from "./labels";
import type {
	Detection,
	DetectionSet,
	DisplayImage,
	GeoTIFFMeta,
} from "./types";

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

export type AppStatus =
	| { type: "idle" }
	| { type: "loading"; message: string }
	| { type: "processing"; message: string; done: number; total: number }
	| { type: "success"; message: string }
	| { type: "error"; message: string };

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

export interface AppState {
	/** The image currently displayed (only the last one is kept). */
	currentImage: DisplayImage | null;
	/** Accumulated detection sets (one per loaded image). */
	detectionSets: DetectionSet[];
	/** Processing status. */
	status: AppStatus;
	/** Confidence threshold for filtering. */
	confidenceThreshold: number;
	/** URL of the ONNX model file. */
	modelUrl: string;
}

export const initialState: AppState = {
	currentImage: null,
	detectionSets: [],
	status: { type: "idle" },
	confidenceThreshold: CONFIDENCE_THRESHOLD,
	modelUrl: DEFAULT_MODEL_URL,
};

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

export type AppAction =
	| {
			type: "ADD_RESULT";
			image: DisplayImage;
			detections: Detection[];
			isGeoTIFF: boolean;
			geoMeta?: GeoTIFFMeta;
	  }
	| { type: "CLEAR_ALL" }
	| { type: "SET_STATUS"; status: AppStatus }
	| { type: "SET_CONFIDENCE"; value: number }
	| { type: "SET_MODEL_URL"; url: string };

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

export function appReducer(state: AppState, action: AppAction): AppState {
	switch (action.type) {
		case "ADD_RESULT":
			return {
				...state,
				currentImage: action.image,
				detectionSets: [
					...state.detectionSets,
					{
						detections: action.detections,
						isGeoTIFF: action.isGeoTIFF,
						geoMeta: action.geoMeta,
					},
				],
			};

		case "CLEAR_ALL":
			return {
				...initialState,
				confidenceThreshold: state.confidenceThreshold,
			};

		case "SET_STATUS":
			return { ...state, status: action.status };

		case "SET_CONFIDENCE":
			return { ...state, confidenceThreshold: action.value };

		case "SET_MODEL_URL":
			return { ...state, modelUrl: action.url };

		default:
			return state;
	}
}
