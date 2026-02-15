/**
 * Load an image file (jpg/png) into an HTMLImageElement.
 */
export function loadImageFromFile(file: File): Promise<HTMLImageElement> {
	return new Promise((resolve, reject) => {
		const url = URL.createObjectURL(file);
		const img = new Image();
		img.onload = () => {
			URL.revokeObjectURL(url);
			resolve(img);
		};
		img.onerror = () => {
			URL.revokeObjectURL(url);
			reject(new Error("Failed to load image"));
		};
		img.src = url;
	});
}

/**
 * Extract pixel data from a canvas as Float32Array in CHW format (RGB, 0-1 normalized).
 * The canvas should already be sized to the desired dimensions.
 */
export function canvasToFloat32CHW(
	canvas: HTMLCanvasElement | OffscreenCanvas,
): Float32Array {
	const ctx =
		canvas instanceof HTMLCanvasElement
			? canvas.getContext("2d")
			: canvas.getContext("2d");
	if (!ctx) throw new Error("Cannot get 2d context");

	const { width, height } = canvas;
	const imageData = ctx.getImageData(0, 0, width, height);
	const { data } = imageData;
	const chw = new Float32Array(3 * width * height);
	const planeSize = width * height;

	for (let i = 0; i < planeSize; i++) {
		chw[i] = data[i * 4] / 255; // R
		chw[planeSize + i] = data[i * 4 + 1] / 255; // G
		chw[2 * planeSize + i] = data[i * 4 + 2] / 255; // B
	}
	return chw;
}

/**
 * Prepare a tile from the source image/canvas for the model input.
 * Draws the region (sx, sy, sw, sh) from src onto a MODEL_INPUT_SIZE canvas with letterboxing.
 * Returns the Float32Array in CHW format and the scale/padding info for coordinate mapping.
 */
export function prepareTile(
	src: HTMLCanvasElement | HTMLImageElement,
	sx: number,
	sy: number,
	sw: number,
	sh: number,
	modelSize: number,
): {
	input: Float32Array;
	scale: number;
	padX: number;
	padY: number;
} {
	const canvas = document.createElement("canvas");
	canvas.width = modelSize;
	canvas.height = modelSize;
	const ctx = canvas.getContext("2d");
	if (!ctx) throw new Error("Cannot get 2d context");

	// Letterbox: fit the tile into modelSize x modelSize
	const scale = Math.min(modelSize / sw, modelSize / sh);
	const newW = Math.round(sw * scale);
	const newH = Math.round(sh * scale);
	const padX = (modelSize - newW) / 2;
	const padY = (modelSize - newH) / 2;

	// Fill with gray (114/255 is YOLO's default padding)
	ctx.fillStyle = `rgb(114, 114, 114)`;
	ctx.fillRect(0, 0, modelSize, modelSize);
	ctx.drawImage(src, sx, sy, sw, sh, padX, padY, newW, newH);

	return {
		input: canvasToFloat32CHW(canvas),
		scale,
		padX,
		padY,
	};
}
