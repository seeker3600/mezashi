import { CONFIDENCE_THRESHOLD } from "./labels";
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
}

export const initialState: AppState = {
	currentImage: null,
	detectionSets: [],
	status: { type: "idle" },
	confidenceThreshold: CONFIDENCE_THRESHOLD,
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
	| { type: "SET_CONFIDENCE"; value: number };

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

		default:
			return state;
	}
}
