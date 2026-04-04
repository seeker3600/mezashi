import { CONFIDENCE_THRESHOLD, DEFAULT_METADATA_URL } from "./labels";
import type {
	Detection,
	DetectionSet,
	DetectionTask,
	DisplayImage,
	GeoTIFFMeta,
	ModelMetadata,
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
	/** URL of the model metadata JSON file. */
	metadataUrl: string;
	/** Loaded model metadata (null while loading or on error). */
	modelMetadata: ModelMetadata | null;
	/**
	 * User-specified ground sample distance in metres per pixel.
	 * Used for non-GeoTIFF images. null means "use model default".
	 */
	userGSD: number | null;
}

export const initialState: AppState = {
	currentImage: null,
	detectionSets: [],
	status: { type: "idle" },
	confidenceThreshold: CONFIDENCE_THRESHOLD,
	metadataUrl: DEFAULT_METADATA_URL,
	modelMetadata: null,
	userGSD: null,
};

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

export type AppAction =
	| {
			type: "ADD_RESULT";
			image: DisplayImage;
			detections: Detection[];
			task: DetectionTask;
			isGeoTIFF: boolean;
			geoMeta?: GeoTIFFMeta;
	  }
	| { type: "CLEAR_ALL" }
	| { type: "SET_STATUS"; status: AppStatus }
	| { type: "SET_CONFIDENCE"; value: number }
	| { type: "SET_METADATA_URL"; url: string }
	| { type: "SET_MODEL_METADATA"; metadata: ModelMetadata | null }
	| { type: "SET_USER_GSD"; value: number | null };

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
						task: action.task,
						isGeoTIFF: action.isGeoTIFF,
						geoMeta: action.geoMeta,
					},
				],
			};

		case "CLEAR_ALL":
			return {
				...initialState,
				confidenceThreshold: state.confidenceThreshold,
				metadataUrl: state.metadataUrl,
				modelMetadata: state.modelMetadata,
				userGSD: state.userGSD,
			};

		case "SET_STATUS":
			return { ...state, status: action.status };

		case "SET_CONFIDENCE":
			return { ...state, confidenceThreshold: action.value };

		case "SET_METADATA_URL":
			return { ...state, metadataUrl: action.url };

		case "SET_MODEL_METADATA":
			return { ...state, modelMetadata: action.metadata };

		case "SET_USER_GSD":
			return { ...state, userGSD: action.value };

		default:
			return state;
	}
}
