/** A single oriented bounding box detection */
export interface Detection {
	/** Class index (0-14) */
	classId: number;
	/** Class label */
	className: string;
	/** Detection confidence (0-1) */
	confidence: number;
	/** Center x in pixel coordinates */
	cx: number;
	/** Center y in pixel coordinates */
	cy: number;
	/** Box width in pixels */
	width: number;
	/** Box height in pixels */
	height: number;
	/** Rotation angle in radians */
	angle: number;
}

/** The 4 corner points of an oriented bounding box */
export interface OBBCorners {
	points: [number, number][];
}

/** GeoTIFF metadata for coordinate transformation */
export interface GeoTIFFMeta {
	/** Tie point: pixel (0,0) maps to this geo coordinate */
	tiePoint: { x: number; y: number };
	/** Pixel scale in geo units per pixel */
	pixelScale: { x: number; y: number };
	/** EPSG code of the coordinate reference system */
	epsg: number | null;
}

/** Full inference result */
export interface InferenceResult {
	detections: Detection[];
	/** Original image width */
	imageWidth: number;
	/** Original image height */
	imageHeight: number;
	/** Whether the source was a GeoTIFF */
	isGeoTIFF: boolean;
	/** GeoTIFF metadata (if applicable) */
	geoMeta?: GeoTIFFMeta;
}

/** A set of detections from a single image, with optional geo metadata for merging */
export interface DetectionSet {
	detections: Detection[];
	isGeoTIFF: boolean;
	geoMeta?: GeoTIFFMeta;
}

/** The currently displayed image data (only the last loaded image is retained) */
export interface DisplayImage {
	source: HTMLCanvasElement | HTMLImageElement;
	width: number;
	height: number;
}
